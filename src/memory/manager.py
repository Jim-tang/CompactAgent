"""MemoryManager 门面：提供 L0 落盘、后台分层抽取与巩固、分层检索的统一接口。"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from datetime import datetime, timedelta

from ..common_utils import PROJECT_ROOT
from .store import MemoryStore
from .utils import (
    clip_density_window,
    close_memory_logging,
    parse_iso,
    sanitize_l0_message,
    setup_memory_logging,
    workspace_scope,
)
from .worker import MemoryWorker

logger = logging.getLogger(__name__)


class MemoryManager:
    """分层记忆系统门面：持有数据库、任务 worker、检索层与三阶段处理器。"""

    def __init__(self, llm_client: Any, config: dict) -> None:
        self.llm_client = llm_client
        self.config = config or {}

        self.enabled: bool = bool(self.config.get("enabled", False))
        self.scope: str = self.config.get("scope") or workspace_scope(PROJECT_ROOT)

        db_path = Path(self.config.get("db_path", "./memory_db/memory.sqlite3"))
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path
        self.db_path = db_path

        self.store = MemoryStore(self.db_path)
        self.worker: Optional[MemoryWorker] = None
        self._started = False
        self._prewarm: Optional[asyncio.Task] = None
        self._vector_available: bool = False

        # 向量腿与检索层：构造时只建对象、不加载模型，模型加载留给 start() 后的预热。
        from .retrieval import Retriever
        from .vector import VectorIndex

        # 向量腿可选，检索层本身始终构建，以保证纯 FTS5 检索可用。
        self.vector: Optional[VectorIndex] = None
        self.retriever: Optional[Retriever] = None
        if self.config.get("enable_vector", True):
            try:
                self.vector = VectorIndex(self.config)
            except Exception as exc:
                logger.warning("向量腿初始化失败（降级为纯 FTS5）: %s", exc)
                self.vector = None
        try:
            self.retriever = Retriever(self.store, self.vector, self.config)
        except Exception as exc:
            logger.warning("检索层初始化失败（记忆检索将不可用）: %s", exc)
            self.retriever = None

        self._l0_sync_cursor: dict[str, int] = {}  # 原始消息落盘游标
        self._finalized: set[str] = set()
        self._session_id: Optional[str] = None
        self._context_snapshots: dict[str, str] = {}  # 注入块快照（按 session_id 冻结，同一会话内不变）
        self.extract_batch_size = int(self.config.get("extract_batch_size", 10))
        self.mirror_dir = Path(self.config.get("mirror_dir", "./memory_db"))  # L3 镜像目录
        self.l0_export_dir = PROJECT_ROOT / Path(self.config.get("l0_export_dir", "./sessions/l0"))  # L0 快照目录
        self.log_path: Optional[Path] = None

        # L2 巩固的去抖间隔与最长等待兜底
        self.l2_min_interval_seconds = float(
            self.config.get("l2_min_interval_seconds", 900)
        )
        self.l2_max_interval_seconds = float(
            self.config.get("l2_max_interval_seconds", 3600)
        )
        # L3 沉淀阈值（首个场景立即触发；之后累计满 N 次 L2 变更再触发）
        self.persona_trigger_every_n = int(self.config.get("persona_trigger_every_n", 3))
        # L2 累计变更次数（按 scope 分键），只活内存，进程重启后从 0 重新数。
        self._l2_change_counter: dict[str, int] = {}

        # LLM 管线：四个阶段共用 llm_client + config，构造期创建一次。
        from .pipeline import MemoryPipeline

        self.pipeline = MemoryPipeline(llm_client, self.config)

        # 三个后台阶段的任务处理器与共享上下文集中在一个聚合对象里。
        from .handler import Handlers

        self.handlers = Handlers(
            store=self.store,
            scope=self.scope,
            config=self.config,
            pipeline=self.pipeline,
            vector=self.vector,
            mirror_dir=self.mirror_dir,
        )

    @property
    def session_id(self) -> Optional[str]:
        """当前记忆会话 ID；尚未开启或已收尾时为 None。"""
        return self._session_id

    def start_session(self, session_id: str) -> str:
        """开启记忆会话：绑定 session_id、清空该会话的游标与注入块快照，返回该 ID。"""
        self._session_id = session_id
        self._finalized.discard(session_id)
        self._l0_sync_cursor.pop(session_id, None)
        self._context_snapshots.pop(session_id, None)
        return session_id

    async def start(self) -> None:
        """建库（毫秒级，正常 await）+ 启动 worker；模型预热在后台进行。"""
        if self._started:
            return
        try:
            self.log_path = setup_memory_logging(self.config.get("log_dir", "./logs"))
            await asyncio.to_thread(self.store.initialize)
            self.worker = MemoryWorker(
                self.store,
                self.config,
                handlers=self.handlers.as_worker_handlers,
                on_task_settled=self._on_task_settled,
            )
            await self.worker.start()
            self._started = True
            logger.info(
                "MemoryManager started: db=%s fts5=%s scope=%s log=%s",
                self.db_path,
                self.store.fts5_available,
                self.scope,
                self.log_path,
            )
            # 启动共享的模型预热 task
            self._prewarm = asyncio.create_task(self._prewarm_models())
            # L0 TTL 清扫：只删 L0，且未抽取的段落先补抽再删
            asyncio.create_task(self._purge_l0_safe(), name="memory-l0-ttl")
            # 空闲触发：闲置超时仍未满一批的会话补一次抽取
            idle_seconds = float(self.config.get("extract_idle_seconds", 300))
            if idle_seconds > 0:
                asyncio.create_task(
                    self._idle_extract_loop(idle_seconds), name="memory-idle-extract"
                )
            if self.l2_min_interval_seconds > 0:
                asyncio.create_task(
                    self._l2_scheduler_loop(), name="memory-l2-scheduler"
                )
        except Exception as e:
            self.enabled = False
            logger.error("记忆系统启动失败: %s", str(e))
            raise

    async def close(self, flush: bool = True) -> None:
        """停止 worker；flush=True 时先处理完在途任务。"""
        if self._prewarm is not None:
            if flush:
                try:
                    await self._prewarm
                except Exception:
                    pass
            self._prewarm = None
        if self.worker is not None:
            await self.worker.stop(flush=flush)
            self.worker = None
        await asyncio.to_thread(self.store.close)

        if self.log_path is not None:
            logger.info("MemoryManager closed: log=%s", self.log_path)
        close_memory_logging()
        self._started = False

    async def _prewarm_models(self) -> None:
        """并行加载 embedder 与 reranker，再打开向量库并刷新向量可用状态。

        任一模型加载失败只记日志、不抛异常；向量库打开失败时向量检索不可用。
        """
        if self.vector is None:
            return
        results = await asyncio.gather(
            asyncio.to_thread(self.vector.load_embedder),
            asyncio.to_thread(self.vector.load_reranker),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, BaseException):
                logger.warning("模型预热出现异常（已忽略）: %s", r)
        try:
            self.vector.open_collection()
        except Exception as exc:
            logger.warning("向量库打开失败（降级为纯 FTS5）: %s", exc)
        self._vector_available = bool(self.vector.available)
        if self.retriever is not None:
            self.retriever.vector = self.vector
        logger.info(
            "模型预热完成: embedder=%s reranker=%s vector=%s",
            self.vector._embedder is not None,
            self.vector._reranker is not None,
            self._vector_available,
        )

    async def _ensure_models(self) -> None:
        """等待已在进行的模型预热完成，异常忽略。"""
        if self._prewarm is not None:
            try:
                await self._prewarm
            except Exception:
                pass

    async def sync_from(self, raw_messages: list[dict]) -> int:
        """把原始消息流中尚未落盘的部分清洗后写入 L0，返回本次写入条数。"""
        if not self.enabled or not self._session_id or not raw_messages:
            return 0
        try:
            # 用「已落盘条数」作游标增量推进
            cursor = self._l0_sync_cursor.get(self._session_id, 0)
            total = len(raw_messages)
            if cursor > total:
                # 原始流理论上不会变短（append-only），出现说明调用方换了一份流
                cursor = 0
            fresh = raw_messages[cursor:]
            if not fresh:
                return 0
            cleaned = [sanitize_l0_message(m, self.config) for m in fresh]
            if self.enabled:
                await self.record_turn(self._session_id, cleaned)
            self._l0_sync_cursor[self._session_id] = total
            return len(cleaned)

        except Exception as e:
            logger.warning(f"[Agent] Warning: 记忆落盘失败: {e}")
            return 0

    async def end_session(
        self, session_id: Optional[str] = None, notes: str = ""
    ) -> dict:
        """收尾指定会话（缺省为当前绑定会话）：入队剩余抽取、强制一次 L2 巩固、导出 L0 快照。

        同一 session 重复调用只在第一次真正执行。返回
        ``{"session_id", "enqueued_extract", "pending_scenes", "snapshot", "notes"}``。
        """
        session_id = session_id or self._session_id
        result = {
            "session_id": session_id,
            "enqueued_extract": False,
            "pending_scenes": 0,
            "snapshot": None,
            "notes": notes,
        }
        if session_id is None:
            return result
        if session_id in self._finalized:
            return result

        if self.enabled:
            # 会话结束：剩余未抽取消息强制入队（不等批满）；有在途任务时跳过——
            # 在途任务读的是「id > from_id」没有上界，后到的消息本来就会被它吃掉。
            result["enqueued_extract"] = (
                await self._maybe_enqueue_extract(session_id, force=True)
            ) is not None

            # L2：会话结束强制巩固一次，不等去抖间隔
            pending = await asyncio.to_thread(self.store.list_pending_scenes, self.scope)
            result["pending_scenes"] = len(pending)
            if pending:
                try:
                    result["l2_task"] = await self._maybe_enqueue_l2(force=True)
                except Exception as exc:
                    logger.warning("会话结束强制 L2 巩固失败: %s", exc)

            try:
                path = await asyncio.to_thread(
                    self.store.export_session_json, session_id, self.l0_export_dir
                )
                result["snapshot"] = str(path)
                logger.info("L0 快照已导出: %s", path)
            except Exception as exc:  # 导出失败不影响主流程
                logger.warning("L0 快照导出失败: %s", exc)

        self._finalized.add(session_id)
        self._l0_sync_cursor.pop(session_id, None)
        # 会话结束即清掉注入块快照，下个会话重新取一份新鲜的人设/索引。
        self._context_snapshots.pop(session_id, None)
        self._session_id = None
        return result

    async def describe(self) -> str:
        """渲染记忆状态展示文本：开关、计数、队列深度与最近抽取/巩固时间。"""
        if not self.enabled:
            return "[Memory] 未启用（config.json → memory.enabled = false）"
        try:
            data = await self.status_async()
        except Exception as exc:  # noqa: BLE001
            return f"[Memory] 状态读取失败: {exc}"
        counts = data.get("counts") or {}
        queue = data.get("queue") or {}
        queue_text = ", ".join(f"{k}={v}" for k, v in queue.items()) or "空闲"
        return "\n".join([
            "[Memory] 状态",
            f"  enabled={data.get('enabled')} started={data.get('started')}",
            f"  scope={data.get('scope')}",
            f"  db={data.get('db_path')}",
            f"  fts5={data.get('fts5_available')} vector={data.get('vector_available')}",
            f"  L0={counts.get('l0', 0)} L1={counts.get('l1', 0)} "
            f"L2={counts.get('l2', 0)} L3={counts.get('l3', 0)}",
            f"  队列：{queue_text}",
            f"  上次抽取={data.get('last_extract_at')} "
            f"上次巩固={data.get('last_consolidate_at')}",
        ])

    async def record_turn(self, session_id: str, messages: list[dict]) -> None:
        """把本轮消息写入 L0，再按批大小决定是否入队抽取，最后刷新该会话的 L0 JSON 镜像。"""
        if not self.enabled or not messages:
            return
        await asyncio.to_thread(
            self.store.append_messages,
            session_id=session_id, scope=self.scope, messages=messages,
        )
        await self._maybe_enqueue_extract(session_id)
        await self._refresh_l0_mirror(session_id)

    async def _refresh_l0_mirror(self, session_id: str) -> None:
        """把该会话的 L0 导出成 ``l0_export_dir/{sid}.json`` 镜像。"""
        try:
            await asyncio.to_thread(self.store.export_session_json, session_id, self.l0_export_dir)
        except Exception as exc:
            logger.warning("L0 镜像刷新失败: %s", exc)

    async def _remove_l0_mirror(self, session_id: str) -> None:
        """删除该会话的 L0 JSON 镜像，失败只告警。"""
        path = self.l0_export_dir / f"{session_id}.json"
        try:
            path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("L0 镜像删除失败: %s", exc)

    async def _maybe_enqueue_extract(
        self, session_id: str, force: bool = False
    ) -> Optional[str]:
        """待抽取行数达到阈值（``force`` 时只要有就够）且无在途抽取任务时入队，返回 task_id。

        阈值判断只统计会进抽取的行（``role='tool'`` 不进 L1），否则工具调用多的会话
        会在半批时就触发抽取，白跑一次 LLM；已有在途任务时返回 None——它读的是
        「id > from_id」没有上界，后到的消息本来就会被它吃掉。
        """
        if self.worker is None:
            return None
        cursor = await asyncio.to_thread(self.store.extract_cursor, session_id)
        pending = await asyncio.to_thread(
            self.store.count_extractable_l0_after, session_id, cursor
        )
        if pending <= 0 or (not force and pending < self.extract_batch_size):
            return None
        if await asyncio.to_thread(self.store.has_open_task, "extract_l1", session_id):
            return None
        return await self.worker.enqueue_extract(
            session_id,
            from_id=cursor,
            until_id=await asyncio.to_thread(self.store.max_l0_id, session_id),
            scope=self.scope,
        )

    async def _on_task_settled(self, task: dict) -> None:
        """任务收尾回调：``consolidate_l2`` 任务读回 worker 已解出的 ``changed_scenes`` 后考虑级联到 L3。"""
        if task.get("kind") != "consolidate_l2":
            return
        await self._cascade_l2_to_l3(int(task.get("changed_scenes") or 0))

    async def _cascade_l2_to_l3(self, changed_scenes: int) -> bool:
        """按变更场景数判断是否触发一次 L3 沉淀，触发返回 True、跳过返回 False。

        首个 L2 场景创建后立即触发；之后每次 L2 变更让计数器加一，满
        ``persona_trigger_every_n`` 触发一次并归零。已有在途 L3 任务时不触发。
        """
        if not self.enabled or changed_scenes <= 0:
            return False
        if await asyncio.to_thread(self.store.has_open_task, "consolidate_l3"):
            logger.info("L2→L3 级联：已有在途 L3 任务，跳过")
            return False

        every_n = max(int(self.config.get("persona_trigger_every_n", 3)), 1)
        # 首个场景判定：直接数 l2_scene 表，不依赖内存状态（重启后仍准确）
        scene_count = await asyncio.to_thread(self.store.count_scenes, scope=self.scope)
        if scene_count <= 1:
            reason = "first_scene"
            self._l2_change_counter[self.scope] = 0
        else:
            pending = self._l2_change_counter.get(self.scope, 0) + 1
            if pending < every_n:
                self._l2_change_counter[self.scope] = pending
                logger.info(
                    "L2→L3 级联：累计变更 %d，未达阈值 %d，暂不沉淀", pending, every_n
                )
                return False
            reason = "threshold"
            self._l2_change_counter[self.scope] = 0  # 触发即归零

        await self.handlers.l3_handler(
            {"kinds": ["persona", "doctrine"], "trigger": reason},
            {"session_id": self._session_id},
        )
        logger.info(
            "L2→L3 级联：已沉淀 L3（改动 %d 个场景, 原因=%s）", changed_scenes, reason
        )
        return True

    # ------------------------------------------------------------ L0 TTL（只清 L0）
    async def _purge_expired_l0(self, max_batches: int = 20) -> dict:
        """物理删除超出 ``l0_ttl_days`` 的 L0 消息，返回 ``{scanned, deleted, deferred, deleted_rows}``。

        只有 L0 有 TTL，L1/L2/L3 不受影响。某段到期消息若尚未抽取完，本次只入队
        补抽计入 ``deferred``、不删除，等下次调用再删。
        """
        stats = {"scanned": 0, "deleted": 0, "deferred": 0, "deleted_rows": 0}
        if not self.enabled:
            return stats
        ttl_days = float(self.config.get("l0_ttl_days", 30))
        if ttl_days <= 0:
            return stats
        cutoff = (datetime.now() - timedelta(days=ttl_days)).isoformat()

        spans = await asyncio.to_thread(self.store.expired_l0_spans, cutoff)
        for span in spans[:max_batches]:
            session_id = span["session_id"]
            max_expired_id = int(span["max_expired_id"])
            stats["scanned"] += 1
            cursor = await asyncio.to_thread(self.store.extract_cursor, session_id)
            if cursor < max_expired_id:
                # 这一段还没抽完：先补抽，本次不删（下轮再删）
                stats["deferred"] += 1
                has_open = await asyncio.to_thread(
                    self.store.has_open_task, "extract_l1", session_id
                )
                if self.worker is not None and not has_open:
                    await self.worker.enqueue_extract(
                        session_id, from_id=cursor, until_id=max_expired_id,
                        scope=self.scope,
                    )
                    logger.info(
                        "L0 TTL：session=%s 尚有未抽取消息，已入队补抽（cursor=%s/%s）",
                        session_id, cursor, max_expired_id,
                    )
                continue
            removed = await asyncio.to_thread(
                self.store.delete_l0_before, session_id, cutoff, max_expired_id
            )
            if removed:
                stats["deleted"] += 1
                stats["deleted_rows"] += removed
                logger.info("L0 TTL：session=%s 删除 %d 条过期消息", session_id, removed)
                # 会话的 L0 已全部清除 → 同步删掉 JSON 镜像，
                # 否则原文还躺在镜像文件里，TTL 形同虚设
                if await asyncio.to_thread(self.store.count_l0, session_id) == 0:
                    await self._remove_l0_mirror(session_id)
                    logger.info("L0 TTL：session=%s 已清空，JSON 镜像一并删除", session_id)
        return stats

    async def _purge_l0_safe(self) -> None:
        """启动时执行一次 TTL 清扫，任何异常只记日志、不影响启动。"""
        try:
            stats = await self._purge_expired_l0()
            if stats["deleted"] or stats["deferred"]:
                logger.info("L0 TTL 清扫: %s", stats)
        except Exception as exc:
            logger.warning("L0 TTL 清扫失败（已忽略）: %s", exc)

    async def _idle_extract_loop(self, idle_seconds: float) -> None:
        """巡检循环：给最后一次落盘已超过 ``idle_seconds`` 且仍有未抽取消息的会话补一次抽取。

        判定所需状态全部查库，不依赖内存。单次巡检异常只记日志，循环继续。
        """
        interval = max(5.0, min(idle_seconds / 4.0, 60.0))
        while True:
            await asyncio.sleep(interval)
            try:
                if not self.enabled or self.worker is None:
                    continue
                cutoff = (datetime.now() - timedelta(seconds=idle_seconds)).isoformat()
                candidates = await asyncio.to_thread(
                    self.store.idle_extract_candidates, cutoff, 20
                )
                for cand in candidates:
                    session_id = cand["session_id"]
                    has_open = await asyncio.to_thread(
                        self.store.has_open_task, "extract_l1", session_id
                    )
                    if has_open:
                        continue
                    cursor = await asyncio.to_thread(self.store.extract_cursor, session_id)
                    await self.worker.enqueue_extract(
                        session_id, from_id=cursor, until_id=int(cand["max_id"]),
                        scope=self.scope,
                    )
                    logger.info(
                        "空闲触发：session=%s 闲置超 %ds，入队补抽（约 %d 条待抽）",
                        session_id, int(idle_seconds), int(cand["max_id"]) - cursor,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("空闲触发巡检失败（已忽略）: %s", exc)

    async def _l2_scheduler_loop(self) -> None:
        """巡检循环：每 ``l2_min_interval_seconds/4``（上限 60s）判断一次是否需要入队 ``consolidate_l2``。

        单次巡检异常只记日志，循环继续。
        """
        interval = max(1.0, min(self.l2_min_interval_seconds / 4.0, 60.0))
        while True:
            await asyncio.sleep(interval)
            try:
                if not self.enabled or self.worker is None:
                    continue
                await self._maybe_enqueue_l2()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("L2 去抖巡检失败（已忽略）: %s", exc)

    async def _maybe_enqueue_l2(self, force: bool = False) -> Optional[str]:
        """存在脏场景且满足触发条件时入队一次 ``consolidate_l2``，返回 task_id 或 None。

        触发条件为距最近脏事件超过 ``l2_min_interval_seconds``，或距首次脏事件超过
        ``l2_max_interval_seconds``（``<= 0`` 表示不设兜底）；``force=True``
        跳过间隔判断直接入队。已有在途 L2 任务时返回 None。
        """
        if self.worker is None:
            return None
        min_interval = self.l2_min_interval_seconds
        max_interval = self.l2_max_interval_seconds
        pending = await asyncio.to_thread(self.store.list_pending_scenes, self.scope)
        if not pending:
            return None
        if not force:
            if await asyncio.to_thread(self.store.has_open_task, "consolidate_l2"):
                return None
            now = datetime.now()
            first = min(parse_iso(p["first_dirty_at"]) for p in pending)
            last = max(parse_iso(p["last_dirty_at"]) for p in pending)
            idle = (now - last).total_seconds()
            age = (now - first).total_seconds()
            if idle < min_interval and not (
                max_interval > 0 and age >= max_interval
            ):
                return None
            trigger = "max_interval" if idle < min_interval else "debounce"
        else:
            trigger = "session_end"
            if await asyncio.to_thread(self.store.has_open_task, "consolidate_l2"):
                # 在途任务会连带吃到这批脏场景，无需再排一条
                return None

        task_id = await self.worker.enqueue_l2(trigger, scope=self.scope)
        logger.info(
            "L2 去抖调度：%d 个脏场景入队巩固（原因=%s, task=%s）",
            len(pending), trigger, task_id,
        )
        return task_id

    async def search_raw_conversation(
        self, query: str, session_id: Optional[str] = None, top_k: int = 5
    ) -> list[dict]:
        """用 FTS5 在 L0 单条消息正文里做关键词检索，按 ``bm25()`` 排序返回前 ``top_k`` 条。

        返回项保留原行字段，``content`` 超过 ``l0_search_window`` 字时截成关键词最密集
        的窗口并在两侧补 ``…``。检索失败返回空列表。
        """
        if not self.enabled or not query or not query.strip():
            return []
        top_k = max(int(top_k), 1)
        try:
            rows = await asyncio.to_thread(
                self.store.search_l0_fts, query, session_id=session_id, limit=top_k
            )
        except Exception as exc:
            logger.warning("L0 会话检索失败（返回空）: %s", exc)
            return []
        window = int(self.config.get("l0_search_window", 500))
        out = []
        for row in rows[:top_k]:
            item = dict(row)
            item["content"] = clip_density_window(
                item.get("content") or "", query, window
            )
            out.append(item)
        return out

    async def get_scene(self, scene_id_or_name: str) -> Optional[dict]:
        """按 scene_id 或场景名取 L2 场景完整字段，未命中返回 None。"""
        if not self.enabled or not scene_id_or_name:
            return None
        try:
            scene = await asyncio.to_thread(self.store.get_scene, scene_id_or_name)
            if not scene:
                scene = await asyncio.to_thread(
                    self.store.find_scene_by_name, scene_id_or_name, self.scope
                )
        except Exception as exc:
            logger.warning("L2 场景读取失败（返回空）: %s", exc)
            return None
        if not scene:
            return None
        return {
            "scene_id": scene.get("scene_id"),
            "name": scene.get("name"),
            "summary": scene.get("summary") or "",
            "content_md": scene.get("content_md") or "",
            "heat": scene.get("heat", 1),
            "scope": scene.get("scope"),
            "created_at": scene.get("created_at"),
            "updated_at": scene.get("updated_at"),
            "status": scene.get("status"),
        }

    async def retrieve(self, query: str, top_k: int = 4) -> list:
        """对 query 做混合召回并用 reranker 精排，返回前 ``top_k`` 条记忆；失败返回空列表。"""
        if not self.enabled or self.retriever is None or not query:
            return []
        try:
            # 首次检索时如果预热还没跑完就等待
            await self._ensure_models()
            self._vector_available = bool(
                self.vector is not None and getattr(self.vector, "available", False)
            )
            return await self.retriever.retrieve(
                query, top_k=top_k, scope=self.scope
            )
        except Exception as exc:
            logger.warning("记忆检索失败（返回空）: %s", exc)
            return []

    async def retrieval_report(self, query: str, top_k: int = 4) -> str:
        """把 ``retrieve`` 的结果渲染成分层的文本作为工具返回内容"""
        result = await self.retrieve(query, top_k)
        if not result:
            return "No relevant memories found."
        lines = ["检索到的相关记忆（按相关性排序）："]
        for item in result:
            layer = item.get("layer")
            if layer not in ("L1", "L2"):
                logger.warning("检索结果缺少合法的 layer 字段，已跳过: %r", item.get("atom_id"))
                continue
            idx = len(lines)
            content = (item.get("content") or "").strip().replace("\n", " ")
            if layer == "L2":
                # L2 只给索引，正文要按场景名或 scene_id 另行拉取
                lines.append(
                    f"{idx}. [L2 场景索引] [场景: {item.get('name') or '未命名'}]\n   {content}"
                )
            else:
                score = item.get("rerank_score")
                score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
                tail = f" [类型: {item.get('type')}]" if item.get("type") else ""
                lines.append(
                    f"{idx}. [L1 原子记忆]{tail} [相关性: {score_text}]\n   {content}"
                )
        return "\n".join(lines)

    async def raw_conversation_report(self, query: str, top_k: int = 5) -> str:
        """把 L0 原始会话检索结果渲染成带 role/session/时间的工具返回文本。"""
        rows = await self.search_raw_conversation(query, session_id=None, top_k=top_k)
        if not rows:
            return "No matching messages found in the raw conversation log."
        lines = ["原始会话检索结果（按关键词密度排序）："]
        for idx, row in enumerate(rows, 1):
            stamp = row.get("recorded_at") or row.get("timestamp") or ""
            lines.append(
                f"{idx}. [{row.get('role')}] [session: {row.get('session_id')}] [时间: {stamp}]\n"
                f"   {row.get('content')}"
            )
        return "\n".join(lines)

    async def scene_report(self, scene_id_or_name: str) -> str:
        """把 L2 场景渲染成带标题、元信息与正文的工具返回文本，未命中时返回固定提示语。"""
        scene = await self.get_scene(scene_id_or_name)
        if not scene:
            return f"Scene '{scene_id_or_name}' not found."
        header = (
            f"# {scene.get('name')}\n"
            f"> scene_id: {scene.get('scene_id')} · heat: {scene.get('heat')} "
            f"· updated: {scene.get('updated_at')}\n"
            f"> 摘要：{scene.get('summary') or ''}\n"
        )
        return header + "\n" + (scene.get("content_md") or "")

    async def build_context_block(self, session_id: Optional[str] = None) -> str:
        """渲染只含 L3 人设与 L2 场景索引的注入块，同一 session_id 只渲染一次并复用缓存。

        构建失败返回空字符串。
        """
        if not self.enabled or self.retriever is None:
            return ""
        if session_id and session_id in self._context_snapshots:
            return self._context_snapshots[session_id]
        try:
            block = await self.retriever.build_context_block(
                scope=self.scope,
                token_budget=int(self.config.get("inject_token_budget", 300)),
            )
        except Exception as exc:
            logger.warning("注入块构建失败（跳过注入）: %s", exc)
            return ""
        if session_id:
            self._context_snapshots[session_id] = block
        return block

    def status(self) -> dict[str, Any]:
        """返回不查库的状态快照：开关、路径、scope、各腿可用性与最近抽取/巩固时间。"""
        return {
            "enabled": self.enabled,
            "started": self._started,
            "db_path": str(self.db_path),
            "scope": self.scope,
            "fts5_available": self.store.fts5_available,
            "vector_available": self._vector_available,
            "last_extract_at": self.handlers.l1_handler.last_extract_at,
            "last_consolidate_at": self.handlers.l3_handler.last_consolidate_at,
            "mirror_dir": self.mirror_dir,
            "log_path": self.log_path
        }

    async def status_async(self) -> dict[str, Any]:
        """在 ``status()`` 基础上补队列深度与各层计数（需要查库），查询失败时计数为空。"""
        data = self.status()
        if self.worker is not None:
            data["queue"] = await self.worker.depth()
        else:
            data["queue"] = {}
        try:
            data["counts"] = {
                "l0": await asyncio.to_thread(self.store.count_l0),
                "l1": await asyncio.to_thread(
                    self.store.scalar,
                    "SELECT COUNT(*) FROM l1_atom WHERE scope = ? AND status = 'active'",
                    (self.scope,),
                ) or 0,
                "l2": await asyncio.to_thread(self.store.count_scenes, scope=self.scope),
                "l3": 0,
            }
            l3 = await asyncio.to_thread(
                self.store.query,
                "SELECT kind FROM l3_persona WHERE scope = ? OR scope = 'global'", (self.scope,)
            )
            data["counts"]["l3"] = len(l3)
        except Exception as exc:
            logger.warning("统计计数失败: %s", exc)
            data["counts"] = {}
        return data
