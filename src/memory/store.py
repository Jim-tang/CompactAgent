"""SQLite 读写层：建表、L0 会话消息、L1 原子、L2 场景、L3 人设与任务队列的存取与 FTS5 检索，不做业务判断。

对外方法均为同步函数，由调用方用 ``asyncio.to_thread`` 落到线程执行；每个线程持有自己的连接。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from .utils import DDL, fts_match_expr, normalize_message, now_iso, scope_clause, segment

logger = logging.getLogger(__name__)


class MemoryStore:
    """sqlite 访问门面：按线程复用连接，提供各层表的同步读写方法。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._local = threading.local()
        self.fts5_available: bool = False
        self._initialized: bool = False

    def _conn(self) -> sqlite3.Connection:
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _close_thread_conn(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._local.conn = None

    def _fts_search(self, table: str, sql: str, query: str, params: list[Any]) -> list[dict]:
        """按 OR 拼 MATCH 表达式执行检索；查询抛 OperationalError 时记日志返回空列表。"""
        match_expr = fts_match_expr(query)
        if not match_expr:
            return []
        conn = self._conn()
        try:
            rows = conn.execute(sql, [match_expr, *params]).fetchall()
        except sqlite3.OperationalError as exc:  # FTS 语法/损坏 → 非致命
            logger.warning("%s FTS 检索失败: %s", table, exc)
            return []
        return [dict(r) for r in rows]

    def initialize(self) -> None:
        """建目录与表、探测 FTS5 能力，成功后置 _initialized"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._conn()
        conn.executescript(DDL)
        # FTS5 能力探测：创建成功即可用；用完即删，不留痕迹
        try:
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts5_probe USING fts5(x)")
            conn.execute("DROP TABLE IF EXISTS __fts5_probe")
            self.fts5_available = True
        except sqlite3.OperationalError:
            self.fts5_available = False
        conn.commit()
        self._initialized = True

    def close(self) -> None:
        self._close_thread_conn()

    def append_messages(
        self,
        session_id: str,
        scope: str,
        messages: list[dict],
        source: str = "main",
    ) -> int:
        """写入一批 L0 消息（附分词后的 content_seg），提交后返回写入条数，空列表返回 0。"""
        if not messages:
            return 0
        conn = self._conn()
        rows = []
        recorded_at = now_iso()
        for msg in messages:
            role, content, tool_calls_json, tool_call_id, timestamp = normalize_message(msg)
            rows.append(
                (
                    session_id,
                    scope,
                    role,
                    content,
                    segment(content),
                    tool_calls_json,
                    tool_call_id,
                    timestamp,
                    recorded_at,
                    source,
                )
            )
        conn.executemany(
            """
            INSERT INTO l0_conversation
                (session_id, scope, role, content, content_seg, tool_calls_json,
                 tool_call_id, timestamp, recorded_at, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return len(rows)

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> list[dict]:
        conn = self._conn()
        sql = (
            "SELECT id, session_id, role, content, content_seg, tool_calls_json,"
            "       tool_call_id, timestamp, recorded_at, source"
            "  FROM l0_conversation WHERE session_id = ? ORDER BY id"
        )
        params: list[Any] = [session_id]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def count_l0(self, session_id: Optional[str] = None) -> int:
        conn = self._conn()
        if session_id:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM l0_conversation WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) AS c FROM l0_conversation").fetchone()
        return int(row["c"])

    def count_extractable_l0_after(self, session_id: str, after_id: int) -> int:
        """返回该会话中 id 大于 after_id 且 role 为 user 或 assistant 的 L0 行数（不含 tool）。"""
        conn = self._conn()
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM l0_conversation"
            " WHERE session_id = ? AND id > ? AND role IN ('user', 'assistant')",
            (session_id, after_id),
        ).fetchone()
        return int(row["c"])

    def max_l0_id(self, session_id: str) -> int:
        conn = self._conn()
        row = conn.execute(
            "SELECT MAX(id) AS m FROM l0_conversation WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None or row["m"] is None:
            return 0
        return int(row["m"])

    def search_l0_fts(
        self, query: str, session_id: Optional[str] = None, limit: int = 20
    ) -> list[dict]:
        """L0 关键词检索：把 query 分词后按 OR 拼成 MATCH 表达式，按 bm25 升序返回最多 limit 行。"""
        sql = (
            "SELECT c.id, c.session_id, c.role, c.content, c.timestamp,"
            "       bm25(l0_conversation_fts) AS score"
            "  FROM l0_conversation_fts f"
            "  JOIN l0_conversation c ON c.id = f.rowid"
            " WHERE l0_conversation_fts MATCH ?"
        )
        params: list[Any] = []
        if session_id:
            sql += " AND c.session_id = ?"
            params.append(session_id)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        return self._fts_search("L0", sql, query, params)

    def export_session_json(self, session_id: str, out_dir: str | Path) -> Path:
        """L0 快照导出：把该会话的全部 L0 消息写成 ``<out_dir>/<session_id>.json``，返回文件路径。"""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        target = out / f"{session_id}.json"
        payload = {
            "session_id": session_id,
            "exported_at": now_iso(),
            "messages": self.get_messages(session_id),
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    def scalar(self, sql: str, params: tuple = ()) -> Any:
        """ 通用 SQL 查询"""
        conn = self._conn()
        row = conn.execute(sql, params).fetchone()
        if row is None:
            return None
        return row[0]

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        conn = self._conn()
        return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def execute(self, sql: str, params: tuple = ()) -> int:
        """执行写语句并 commit，返回受影响行数。"""
        conn = self._conn()
        cursor = conn.execute(sql, params)
        conn.commit()
        return cursor.rowcount

    def apply_decisions(self, new_atoms: list[dict], decisions: list[dict]) -> dict:
        """在一个事务里按判定结果插入、更新或跳过 L1 原子。

        update / merge 把新内容写到首个目标并提升其 version，其余目标置 superseded；
        返回 ``{stored, updated, merged, skipped}`` 计数；事务内异常回滚后重新抛出。
        """
        conn = self._conn()
        stats = {"stored": 0, "updated": 0, "merged": 0, "skipped": 0}
        to_insert: list[dict] = []
        try:
            for atom, decision in zip(new_atoms, decisions):
                action = decision.get("action", "store")
                targets = [t for t in (decision.get("target_ids") or []) if t]
                merged_content = decision.get("merged_content")
                merged_type = decision.get("merged_type")
                merged_priority = decision.get("merged_priority")

                if action in ("store", "skip") or not targets:
                    if action == "skip":
                        stats["skipped"] += 1
                        continue
                    to_insert.append(atom)
                    stats["stored"] += 1
                    continue

                # update / merge：新内容写入「主」目标，其余目标置 superseded
                primary = targets[0]
                final_content = merged_content or atom["content"]
                final_type = merged_type or atom["type"]
                try:
                    final_priority = int(merged_priority) if merged_priority is not None else atom["priority"]
                except (TypeError, ValueError):
                    final_priority = atom["priority"]

                conn.execute(
                    """
                    UPDATE l1_atom
                       SET content = ?, content_seg = ?, type = ?, priority = ?,
                           version = version + 1, updated_at = ?, status = 'active'
                     WHERE atom_id = ?
                    """,
                    (
                        final_content,
                        segment(final_content),
                        final_type,
                        final_priority,
                        now_iso(),
                        primary,
                    ),
                )
                for other in targets[1:]:
                    conn.execute(
                        "UPDATE l1_atom SET status = 'superseded', updated_at = ? WHERE atom_id = ?",
                        (now_iso(), other),
                    )
                if action == "merge":
                    stats["merged"] += 1
                else:
                    stats["updated"] += 1
            if to_insert:
                conn.executemany(
                    """
                    INSERT INTO l1_atom
                        (atom_id, version, type, content, content_seg, priority,
                         source_message_ids, metadata_json, scene_id, scope, session_id,
                         created_at, updated_at, valid_from, valid_to, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            a["atom_id"],
                            int(a.get("version", 1)),
                            a["type"],
                            a["content"],
                            a.get("content_seg") or segment(a["content"]),
                            int(a["priority"]),
                            json.dumps(a.get("source_message_ids") or [], ensure_ascii=False),
                            json.dumps(a.get("metadata") or {}, ensure_ascii=False),
                            a.get("scene_id"),
                            a["scope"],
                            a["session_id"],
                            a.get("created_at") or now_iso(),
                            a.get("updated_at") or now_iso(),
                            a.get("valid_from"),
                            a.get("valid_to"),
                            a.get("status", "active"),
                        )
                        for a in to_insert
                    ],
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return stats

    def search_atoms_fts(
        self, query: str, scope: Optional[str] = None, limit: int = 20
    ) -> list[dict]:
        """把 query 分词去重后按 OR 拼成 MATCH 表达式检索 active 的 L1 原子，按 bm25 升序返回。

        FTS5 不可用或查询抛 OperationalError 时返回空列表。
        """
        sql = (
            "SELECT a.atom_id, a.content, a.type, a.priority, a.scope, a.session_id,"
            "       a.scene_id, a.created_at, bm25(l1_atom_fts) AS score"
            "  FROM l1_atom_fts f"
            "  JOIN l1_atom a ON a.id = f.rowid"
            " WHERE l1_atom_fts MATCH ? AND a.status = 'active'"
        )
        params: list[Any] = []
        if scope:
            sql += " AND (a.scope = ? OR a.scope = 'global')"
            params.append(scope)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        return self._fts_search("L1", sql, query, params)

    def enqueue_task(
        self, task_id: str, kind: str, scope: str, payload: dict, session_id: Optional[str]
    ) -> str:
        conn = self._conn()
        ts = now_iso()
        conn.execute(
            """
            INSERT INTO memory_task
                (task_id, kind, session_id, scope, status, attempts,
                 payload_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'pending', 0, ?, ?, ?)
            """,
            (task_id, kind, session_id, scope, json.dumps(payload, ensure_ascii=False), ts, ts),
        )
        conn.commit()
        return task_id

    def claim_next_task(self) -> Optional[dict]:
        """认领一条 pending 任务并置为 running、attempts 加一，返回该任务行；无可认领时返回 None。

        跳过尚未到 next_retry_at 的任务，以及同 session_id 已有 running 任务的条目（同会话串行）。
        """
        conn = self._conn()
        ts = now_iso()
        rows = conn.execute(
            """
            SELECT id FROM memory_task
             WHERE status = 'pending'
               AND (next_retry_at IS NULL OR next_retry_at <= ?)
               AND (session_id IS NULL OR session_id NOT IN (
                     SELECT session_id FROM memory_task
                      WHERE status = 'running' AND session_id IS NOT NULL))
             ORDER BY id LIMIT 1
            """,
            (ts,),
        ).fetchall()
        for row in rows:
            row_id = row["id"]
            cursor = conn.execute(
                """
                UPDATE memory_task
                   SET status = 'running', attempts = attempts + 1,
                       started_at = ?, updated_at = ?
                 WHERE id = ? AND status = 'pending'
                """,
                (ts, ts, row_id),
            )
            conn.commit()
            if cursor.rowcount == 1:
                claimed = conn.execute(
                    "SELECT * FROM memory_task WHERE id = ?", (row_id,)
                ).fetchone()
                return dict(claimed) if claimed else None
        return None

    def recover_stale_tasks(self, stale_seconds: int = 600) -> int:
        """把 started_at 早于 stale_seconds 之前仍处于 running 的任务重置为 pending，返回重置条数。"""
        conn = self._conn()
        cutoff = (
            datetime.now() - timedelta(seconds=stale_seconds)
        ).isoformat()
        cursor = conn.execute(
            """
            UPDATE memory_task
               SET status = 'pending', started_at = NULL, updated_at = ?
             WHERE status = 'running' AND (started_at IS NULL OR started_at <= ?)
            """,
            (now_iso(), cutoff),
        )
        conn.commit()
        return cursor.rowcount

    def finish_task(self, task_id: str, status: str, error: Optional[str] = None,
                    next_retry_at: Optional[str] = None,
                    payload_patch: Optional[dict] = None) -> None:
        """写入任务状态、错误、下次重试时间，并把 payload_patch 里各键用 json_set 合并进 payload_json。"""
        sets = ["status = ?", "last_error = ?", "next_retry_at = ?", "updated_at = ?"]
        params: list[Any] = [status, error, next_retry_at, now_iso()]
        for key, value in (payload_patch or {}).items():
            sets.append(f"payload_json = json_set(payload_json, '$.{key}', ?)")
            params.append(value)
        params.append(task_id)
        conn = self._conn()
        conn.execute(f"UPDATE memory_task SET {', '.join(sets)} WHERE task_id = ?", params)
        conn.commit()

    def extract_cursor(self, session_id: str) -> int:
        """返回该会话已完成抽取任务中 payload 的最大 ``to_id``，没有则返回 0。"""
        conn = self._conn()
        row = conn.execute(
            """
            SELECT MAX(CAST(json_extract(payload_json, '$.to_id') AS INTEGER)) AS cursor
              FROM memory_task
             WHERE kind = 'extract_l1' AND status = 'done' AND session_id = ?
               AND json_extract(payload_json, '$.to_id') IS NOT NULL
            """,
            (session_id,),
        ).fetchone()
        if row is None or row["cursor"] is None:
            return 0
        return int(row["cursor"])

    def has_open_task(self, kind: str, session_id: Optional[str] = None) -> bool:
        """返回是否已存在该 kind 的 pending 或 running 任务，session_id 为 None 时按全局判断。"""
        conn = self._conn()
        sql = ("SELECT 1 FROM memory_task"
               " WHERE kind = ? AND status IN ('pending', 'running')")
        params: list = [kind]
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        sql += " LIMIT 1"
        return conn.execute(sql, params).fetchone() is not None

    def idle_extract_candidates(self, cutoff: str, limit: int = 20) -> list[dict]:
        """返回闲置但仍需抽取的会话列表，每项含 ``session_id`` / ``max_id`` / ``pending_rows`` / ``from_id``。

        只取满足全部条件的会话：有 ``recorded_at <= cutoff`` 的行、该行 id 超过抽取游标、
        没有在途的 extract_l1 任务，且该会话没有带 until_id 或 trigger 的已结束抽取任务。
        """
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT l.session_id,
                   MAX(l.id)  AS max_id,
                   COUNT(*)   AS pending_rows
            FROM l0_conversation l
            WHERE l.recorded_at <= ?
              AND l.id > IFNULL(
                    (SELECT MAX(CAST(json_extract(t.payload_json, '$.to_id') AS INTEGER))
                     FROM memory_task t
                     WHERE t.kind = 'extract_l1' AND t.status = 'done'
                       AND t.session_id = l.session_id), 0)
              AND NOT EXISTS (
                    SELECT 1 FROM memory_task t2
                    WHERE t2.kind = 'extract_l1'
                      AND t2.status IN ('pending', 'running')
                      AND t2.session_id = l.session_id)
              AND NOT EXISTS (
                    SELECT 1 FROM memory_task t3
                    WHERE t3.kind = 'extract_l1'
                      AND t3.status = 'done'
                      AND t3.session_id = l.session_id
                      AND (
                        json_extract(t3.payload_json, '$.until_id') IS NOT NULL
                        OR json_extract(t3.payload_json, '$.trigger') IS NOT NULL
                      ))
            GROUP BY l.session_id
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()
        return [
            {**dict(r), "from_id": int(r["session_id"] and self.extract_cursor(r["session_id"]))}
            for r in rows
        ]

    def expired_l0_spans(self, cutoff: str) -> list[dict]:
        """返回每个会话已过期的那一段的 ``{session_id, max_expired_id, rows}``。

        ``max_expired_id`` 是过期行里的最大 id，删除只允许进行到它为止。
        """
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT session_id,
                   MAX(id) AS max_expired_id,
                   COUNT(*) AS rows
              FROM l0_conversation
             WHERE recorded_at < ?
             GROUP BY session_id
            """,
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_l0_before(self, session_id: str, cutoff: str, max_id: int) -> int:
        """删除该会话 recorded_at 早于 cutoff 且 id 不大于 max_id 的 L0 行，返回删除条数。

        只删 L0，L1/L2/L3 与任务表不动。
        """
        conn = self._conn()
        cursor = conn.execute(
            "DELETE FROM l0_conversation"
            " WHERE session_id = ? AND recorded_at < ? AND id <= ?",
            (session_id, cutoff, max_id),
        )
        conn.commit()
        return cursor.rowcount

    def list_scenes(
        self, scope: Optional[str] = None, status: str = "active", limit: int = 200
    ) -> list[dict]:
        conn = self._conn()
        sql = "SELECT * FROM l2_scene WHERE status = ?"
        params: list[Any] = [status]
        clause, extra = scope_clause(scope)
        sql += clause
        params.extend(extra)
        sql += " ORDER BY heat DESC, id DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]

    def get_scene(self, scene_id: str) -> Optional[dict]:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM l2_scene WHERE scene_id = ?", (scene_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_scenes_by_ids(
        self, scene_ids: list[str], scope: Optional[str] = None, status: str = "active"
    ) -> list[dict]:
        """按 scene_id 批量取指定 status 的场景行，scope 非空时附带 global 场景；空列表返回空列表。"""
        if not scene_ids:
            return []
        conn = self._conn()
        marks = ",".join("?" for _ in scene_ids)
        sql = f"SELECT * FROM l2_scene WHERE scene_id IN ({marks}) AND status = ?"
        params: list[Any] = list(scene_ids) + [status]
        clause, extra = scope_clause(scope)
        sql += clause
        params.extend(extra)
        return [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]

    def upsert_scene(
        self, scene_id: str, name: str, summary: str, content_md: str,
        scope: str, path: Optional[str] = None, heat: int = 1,
    ) -> None:
        """按 scene_id 插入或覆盖场景行，status 置回 active，未给 path 时以 ``<scene_id>.md`` 占位。"""
        conn = self._conn()
        ts = now_iso()
        # path 列在 DDL 里是 NOT NULL，未显式给出时按 scene_id 填占位值。
        if not path:
            path = f"{scene_id}.md"
        # 预分词列只拼「名字 + 摘要」，不含正文。
        content_seg = segment(f"{name}\n{summary}")
        conn.execute(
            """
            INSERT INTO l2_scene
                (scene_id, name, path, summary, heat, content_md, content_seg, scope,
                 created_at, updated_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
            ON CONFLICT(scene_id) DO UPDATE SET
                name = excluded.name,
                path = excluded.path,
                summary = excluded.summary,
                heat = excluded.heat,
                content_md = excluded.content_md,
                content_seg = excluded.content_seg,
                updated_at = excluded.updated_at,
                status = 'active'
            """,
            (scene_id, name, path, summary, heat, content_md, content_seg, scope, ts, ts),
        )
        conn.commit()

    def find_scene_by_name(
        self, name: str, scope: Optional[str] = None
    ) -> Optional[dict]:
        """按场景名取 active 场景行，同名多条时取 heat 最高的那条，无匹配返回 None。"""
        conn = self._conn()
        sql = "SELECT * FROM l2_scene WHERE name = ? AND status = 'active'"
        params: list[Any] = [name]
        clause, extra = scope_clause(scope)
        sql += clause
        params.extend(extra)
        sql += " ORDER BY heat DESC, id DESC LIMIT 1"
        row = conn.execute(sql, tuple(params)).fetchone()
        return dict(row) if row else None

    def search_scenes_fts(
        self, query: str, scope: Optional[str] = None, limit: int = 20
    ) -> list[dict]:
        """把 query 分词去重后按 OR 拼成 MATCH 表达式检索 active 的场景，按 bm25 升序返回。

        FTS5 不可用或查询抛 OperationalError 时返回空列表。
        """
        sql = (
            "SELECT s.scene_id, s.name, s.summary, s.content_md, s.heat, s.scope,"
            "       s.created_at, s.updated_at, bm25(l2_scene_fts) AS score"
            "  FROM l2_scene_fts f"
            "  JOIN l2_scene s ON s.id = f.rowid"
            " WHERE l2_scene_fts MATCH ? AND s.status = 'active'"
        )
        params: list[Any] = []
        if scope:
            sql += " AND (s.scope = ? OR s.scope = 'global')"
            params.append(scope)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        return self._fts_search("L2", sql, query, params)

    def mark_scenes_dirty(self, scope: str, names: list[str]) -> None:
        """登记待巩固的 L2 场景名：新名插入并记首次时间，已存在的只刷新 last_dirty_at。"""
        names = [n for n in dict.fromkeys(names or []) if n]
        if not names:
            return
        ts = now_iso()
        conn = self._conn()
        conn.executemany(
            """
            INSERT INTO l2_pending_scene (scope, scene_name, first_dirty_at, last_dirty_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(scope, scene_name) DO UPDATE SET last_dirty_at = excluded.last_dirty_at
            """,
            [(scope, name, ts, ts) for name in names],
        )
        conn.commit()

    def list_pending_scenes(self, scope: str) -> list[dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT scene_name, first_dirty_at, last_dirty_at FROM l2_pending_scene"
            " WHERE scope = ? ORDER BY first_dirty_at",
            (scope,),
        ).fetchall()
        return [dict(r) for r in rows]

    def clear_pending_scenes(self, scope: str, names: list[str]) -> None:
        names = [n for n in dict.fromkeys(names or []) if n]
        if not names:
            return
        conn = self._conn()
        placeholders = ",".join("?" for _ in names)
        conn.execute(
            f"DELETE FROM l2_pending_scene WHERE scope = ? AND scene_name IN ({placeholders})",
            (scope, *names),
        )
        conn.commit()

    def soft_delete_scene(self, scene_id: str) -> None:
        """把该 scene_id 的 status 置为 deleted，不删除行。"""
        conn = self._conn()
        conn.execute(
            "UPDATE l2_scene SET status = 'deleted', updated_at = ? WHERE scene_id = ?",
            (now_iso(), scene_id),
        )
        conn.commit()

    def count_scenes(self, scope: Optional[str] = None) -> int:
        conn = self._conn()
        sql = "SELECT COUNT(*) AS n FROM l2_scene WHERE status = 'active'"
        clause, params = scope_clause(scope)
        sql += clause
        row = conn.execute(sql, tuple(params)).fetchone()
        return int(row["n"]) if row else 0

    def get_persona(self, scope: Optional[str], kind: str) -> Optional[dict]:
        conn = self._conn()
        if scope:
            row = conn.execute(
                "SELECT * FROM l3_persona WHERE scope = ? AND kind = ?", (scope, kind)
            ).fetchone()
            if row:
                return dict(row)
        row = conn.execute(
            "SELECT * FROM l3_persona WHERE scope = 'global' AND kind = ?", (kind,)
        ).fetchone()
        return dict(row) if row else None

    def upsert_persona(self, scope: str, kind: str, content_md: str) -> None:
        conn = self._conn()
        ts = now_iso()
        conn.execute(
            """
            INSERT INTO l3_persona (scope, kind, content_md, version, created_at, updated_at)
            VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(scope, kind) DO UPDATE SET
                content_md = excluded.content_md,
                version = l3_persona.version + 1,
                updated_at = excluded.updated_at
            """,
            (scope, kind, content_md, ts, ts),
        )
        conn.commit()

    def get_atoms_by_ids(
        self, atom_ids: list[str], scope: Optional[str] = None
    ) -> list[dict]:
        if not atom_ids:
            return []
        conn = self._conn()
        placeholders = ",".join("?" for _ in atom_ids)
        sql = f"SELECT * FROM l1_atom WHERE atom_id IN ({placeholders}) AND status = 'active'"
        params: list[Any] = list(atom_ids)
        clause, extra = scope_clause(scope)
        sql += clause
        params.extend(extra)
        return [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]
