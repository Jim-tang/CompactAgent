from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import rjieba

from ..common_utils import PROJECT_ROOT

logger = logging.getLogger(__name__)

TEMPLATE_DIR = PROJECT_ROOT / "template"


def setup_memory_logging(log_dir: str | Path, level: int = logging.INFO) -> Optional[Path]:
    """把记忆模块的全部日志落到 ``{log_dir}/memory_YYYYMMDD_HHMMSS.log``。"""
    root = logging.getLogger("src.memory")
    for handler in root.handlers:
        if getattr(handler, "_memory_file_handler", False):
            # 幂等：重复调用只返回首次安装时创建的日志文件路径
            return getattr(handler, "_memory_log_path", None)

    try:
        directory = Path(log_dir)
        if not directory.is_absolute():
            directory = PROJECT_ROOT / directory
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handler = logging.FileHandler(path, encoding="utf-8", delay=True)
    except OSError as exc:
        print(f"[Memory] 记忆日志文件创建失败（只输出到控制台）: {exc}", flush=True)
        return None

    handler._memory_file_handler = True
    handler._memory_log_path = path
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s:%(lineno)d | %(message)s"
    ))
    root.setLevel(level)
    root.addHandler(handler)
    return path


def close_memory_logging() -> None:
    """摘除并关闭记忆日志 handler（连同此前遗留的 ms 级临时 handler）。"""
    root = logging.getLogger("src.memory")
    for handler in list(root.handlers):
        if not getattr(handler, "_memory_file_handler", False):
            continue
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def now_iso() -> str:
    """返回当前本地时间的 naive ISO 字符串。"""
    return datetime.now().isoformat()


def ulid() -> str:
    """生成 26 字符 ULID：48 位毫秒时间戳 + 80 位随机数，按时间有序。"""
    _CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    ts = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = secrets.randbits(80)
    value = (ts << 80) | rand
    chars = []
    for shift in range(25, -1, -1):
        chars.append(_CROCKFORD[(value >> (shift * 5)) & 31])
    return "".join(chars)


def workspace_scope(project_root: str | Path) -> str:
    """由 project_root 的 sha256 前 8 位生成默认 scope，形如 ``workspace:ab12cd34``。"""
    digest = hashlib.sha256(str(Path(project_root)).encode("utf-8")).hexdigest()
    return f"workspace:{digest[:8]}"


def segment(text: str) -> str:
    """用 rjieba 把文本切成空格分隔的词串，供 sqlite FTS5 检索；分词异常时原样返回。"""
    if not text:
        return ""
    try:
        return " ".join(rjieba.cut(text))
    except Exception:
        return text


def fts_match_expr(query: str) -> str:
    """把查询串转成 FTS5 的 OR MATCH 表达式；没有可用词项时返回空串。"""
    terms: list[str] = []
    for raw in segment(query).split():
        term = raw.strip()
        if not term or not any(ch.isalnum() for ch in term):
            continue
        if term not in terms:
            terms.append(term)
    if not terms:
        return ""
    # 词项一律用双引号包成 FTS5「短语」
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


def clip_middle(text: str, limit: int, head_ratio: float = 0.7) -> str:
    """把超过 limit 的文本截成首 head_ratio 比例加尾部，中间以省略标记替代。"""
    if text is None:
        return ""
    text = str(text)
    if limit <= 0 or len(text) <= limit:
        return text
    omitted = len(text) - limit
    head = int(limit * head_ratio)
    tail = limit - head
    marker = f"…（省略 {omitted} 字符）…"
    # text[-0:] 等于整个字符串，必须显式判空，否则「只留首部」会把原文重拼一遍
    if tail <= 0:
        return f"{text[:head]}{marker}"
    if head <= 0:
        return f"{marker}{text[-tail:]}"
    return f"{text[:head]}{marker}{text[-tail:]}"


def sanitize_tool_calls(tool_calls: Optional[list], arg_limit: int = 200) -> Optional[list]:
    """列表参数超过 arg_limit 时只保留首部，返回清洗后的 tool_calls，空输入返回 None。"""
    if not tool_calls:
        return None
    cleaned = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        func = call.get("function") or {}
        args = func.get("arguments")
        if isinstance(args, (dict, list)):
            args = json.dumps(args, ensure_ascii=False)
        args = "" if args is None else str(args)
        cleaned.append(
            {
                "id": call.get("id"),
                "type": call.get("type", "function"),
                "function": {
                    "name": func.get("name"),
                    "arguments": clip_middle(args, arg_limit, head_ratio=1.0),
                },
            }
        )
    return cleaned or None


def sanitize_l0_message(msg: dict, config: Optional[dict] = None) -> dict:
    """把一条原始消息清洗成可落库的 L0 记忆，工具调用参数和返回结果超长时保留首尾。

    返回字典的属性：role/content/tool_calls/tool_call_id/timestamp。
    """
    config = config or {}
    tool_result_limit = int(config.get("l0_tool_result_limit", 1000))
    arg_limit = int(config.get("l0_tool_arg_limit", 200))

    role = str(msg.get("role", "unknown"))
    content = msg.get("content")
    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        ]
        content = "\n".join(parts)
    content = "" if content is None else str(content)

    cleaned: dict = {"role": role, "content": content}
    if role == "tool":
        cleaned["content"] = clip_middle(content, tool_result_limit)
    tool_calls = sanitize_tool_calls(msg.get("tool_calls"), arg_limit)
    if tool_calls:
        cleaned["tool_calls"] = tool_calls
    if msg.get("tool_call_id"):
        cleaned["tool_call_id"] = msg.get("tool_call_id")
    if msg.get("timestamp"):
        cleaned["timestamp"] = msg.get("timestamp")
    return cleaned


def normalize_message(
    msg: dict,
) -> tuple[str, str, Optional[str], Optional[str], str]:
    """把一条消息字典规整成 ``(role, content, tool_calls_json, tool_call_id, timestamp)``。"""
    role = str(msg.get("role", "unknown"))
    content = msg.get("content")
    content = "" if content is None else str(content)
    tool_calls = msg.get("tool_calls")
    tool_calls_json = (
        json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
    )
    tool_call_id = msg.get("tool_call_id")
    timestamp = msg.get("timestamp") or now_iso()
    return role, content, tool_calls_json, tool_call_id, str(timestamp)


def scope_clause(scope: Optional[str]) -> tuple[str, list]:
    """返回限定 scope 的 SQL 片段与其参数；scope 为空时片段为空、参数为空，表示不限。

    非空时同时命中本 scope 与全局（global）记录，避免业务侧各写一遍拼错的过滤条件。
    """
    if not scope:
        return "", []
    return " AND (scope = ? OR scope = 'global')", [scope]


def density_keywords(query: str) -> list[str]:
    """把 query 切成用于窗口密度统计的词列表，切不出词时返回整条 query。

    用与写入侧相同的分词口径，避免子串匹配把「部署」和「部署了」重复计为两次命中。
    """
    words = [w for w in segment(query).split() if w.strip()]
    return words or [query.strip()]


def clip_density_window(content: str, query: str, window: int) -> str:
    """把长消息截成关键词最密集的一段 window 字窗口，超出部分两侧补 ``…`` 且保持原文顺序。"""
    text = content or ""
    if window <= 0 or len(text) <= window:
        return text
    words = density_keywords(query)
    # 先算出每个字符位置是否落在某个关键词上，再用前缀和做 O(n) 滑窗
    hit = [0] * len(text)
    for word in words:
        start = 0
        while True:
            pos = text.find(word, start)
            if pos < 0:
                break
            # 命中区间 [pos, pos+len(word)) 内每格记 1（重叠不重复累加）
            for i in range(pos, min(pos + len(word), len(text))):
                hit[i] = 1
            start = pos + 1
    prefix = [0] * (len(text) + 1)
    for i, flag in enumerate(hit):
        prefix[i + 1] = prefix[i] + flag
    best_start, best_score = 0, -1
    for start in range(0, len(text) - window + 1):
        score = prefix[start + window] - prefix[start]
        if score > best_score:
            best_start, best_score = start, score
    piece = text[best_start: best_start + window]
    if best_start > 0:
        piece = "…" + piece
    if best_start + window < len(text):
        piece = piece + "…"
    return piece


def parse_json(raw: str, kind: type = list):
    """解析 LLM 输出为 JSON，类型不符或解析失败时返回 None"""
    if not raw:
        return None
    # 剥掉代码围栏后的文本
    fence_re = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S | re.I)
    cleaned = fence_re.sub("", raw.strip()).strip()
    if not cleaned:
        return None
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        logger.warning("LLM 输出无法解析为 JSON: %s", cleaned[:200])
        return None
    if not isinstance(data, kind):
        logger.warning("LLM 输出最外层是 %s 而非 %s: %s", type(data).__name__, kind.__name__, cleaned[:200])
        return None
    return data


def load_template(name: str) -> str:
    """读取并返回 ``template/<name>`` 的文件内容。"""
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8")


def render_template(name: str, variables: dict) -> str:
    """读取模板并填入 variables 占位符后返回；文件缺失或占位符不解析时抛异常。"""
    from ..common_utils import render_prompt

    return render_prompt(load_template(name), variables)


def rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = 60,
    weights: Optional[list[float]] = None,
) -> dict[str, float]:
    """用 RRF 把多路有序 id 列表融合成一个 ``{doc_id: 分数}`` 表。

    每路按 ``w / (k + 名次 + 1)`` 累加，空列表与空 id 跳过；weights 为 None 时全路等权，
    长度与 ranked_lists 不一致时抛 ValueError。调用方自行按分数降序排。
    """
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError("weights 必须与 ranked_lists 等长")
    scores: dict[str, float] = {}
    for idx, ranked in enumerate(ranked_lists):
        w = 1.0 if weights is None else float(weights[idx])
        if w <= 0:
            continue
        for rank, doc_id in enumerate(ranked):
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank + 1)
    return scores


def estimate_tokens(text: str) -> int:
    """按约 4 字符一个 token 向上取整估算文本长度。"""
    return (len(text) + 3) // 4


def parse_iso(text: str) -> datetime:
    """解析 naive ISO 时间串；解析失败返回 ``datetime.min``。"""
    try:
        return datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return datetime.min


DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---- L0 原始会话（content_seg 为预分词文本，供 FTS5 检索） ----
CREATE TABLE IF NOT EXISTS l0_conversation (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL,
    scope           TEXT    NOT NULL,
    role            TEXT    NOT NULL,
    content         TEXT,
    content_seg     TEXT    NOT NULL DEFAULT '',
    tool_calls_json TEXT,
    tool_call_id    TEXT,
    timestamp       TEXT    NOT NULL,
    recorded_at     TEXT    NOT NULL,
    source          TEXT    NOT NULL DEFAULT 'main'
);
CREATE INDEX IF NOT EXISTS idx_l0_session ON l0_conversation(session_id, id);
CREATE INDEX IF NOT EXISTS idx_l0_scope   ON l0_conversation(scope, id);

CREATE VIRTUAL TABLE IF NOT EXISTS l0_conversation_fts USING fts5(
    content_seg,
    content='l0_conversation',
    content_rowid='id',
    tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS trg_l0_ai AFTER INSERT ON l0_conversation BEGIN
    INSERT INTO l0_conversation_fts(rowid, content_seg) VALUES (new.id, new.content_seg);
END;
CREATE TRIGGER IF NOT EXISTS trg_l0_ad AFTER DELETE ON l0_conversation BEGIN
    INSERT INTO l0_conversation_fts(l0_conversation_fts, rowid, content_seg)
    VALUES ('delete', old.id, old.content_seg);
END;
CREATE TRIGGER IF NOT EXISTS trg_l0_au AFTER UPDATE ON l0_conversation BEGIN
    INSERT INTO l0_conversation_fts(l0_conversation_fts, rowid, content_seg)
    VALUES ('delete', old.id, old.content_seg);
    INSERT INTO l0_conversation_fts(rowid, content_seg) VALUES (new.id, new.content_seg);
END;

-- ---- L1 原子记忆 ----
CREATE TABLE IF NOT EXISTS l1_atom (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    atom_id             TEXT    NOT NULL UNIQUE,
    version             INTEGER NOT NULL DEFAULT 1,
    type                TEXT    NOT NULL,
    content             TEXT    NOT NULL,
    content_seg         TEXT    NOT NULL DEFAULT '',
    priority            INTEGER NOT NULL,
    source_message_ids  TEXT,
    metadata_json       TEXT,
    scene_id            TEXT,
    scope               TEXT    NOT NULL,
    session_id          TEXT    NOT NULL,
    created_at          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL,
    valid_from          TEXT,
    valid_to            TEXT,
    status              TEXT    NOT NULL DEFAULT 'active'
);
CREATE INDEX IF NOT EXISTS idx_l1_scope_status ON l1_atom(scope, status, priority DESC);
CREATE INDEX IF NOT EXISTS idx_l1_session      ON l1_atom(session_id);
CREATE INDEX IF NOT EXISTS idx_l1_scene        ON l1_atom(scene_id);

CREATE VIRTUAL TABLE IF NOT EXISTS l1_atom_fts USING fts5(
    content_seg,
    content='l1_atom',
    content_rowid='id',
    tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS trg_l1_ai AFTER INSERT ON l1_atom BEGIN
    INSERT INTO l1_atom_fts(rowid, content_seg) VALUES (new.id, new.content_seg);
END;
CREATE TRIGGER IF NOT EXISTS trg_l1_ad AFTER DELETE ON l1_atom BEGIN
    INSERT INTO l1_atom_fts(l1_atom_fts, rowid, content_seg)
    VALUES ('delete', old.id, old.content_seg);
END;
CREATE TRIGGER IF NOT EXISTS trg_l1_au AFTER UPDATE ON l1_atom BEGIN
    INSERT INTO l1_atom_fts(l1_atom_fts, rowid, content_seg)
    VALUES ('delete', old.id, old.content_seg);
    INSERT INTO l1_atom_fts(rowid, content_seg) VALUES (new.id, new.content_seg);
END;

-- ---- L2 场景（content_seg 只含 name+summary 的预分词） ----
-- 正文 content_md 不进 FTS：命中正文会让 bm25 虚高、挤掉真正的索引命中，
-- 而精排只看 name+summary 又评不出分。正文一律按 id 单独取。
CREATE TABLE IF NOT EXISTS l2_scene (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_id    TEXT    NOT NULL UNIQUE,
    name        TEXT    NOT NULL,
    path        TEXT    NOT NULL,
    summary     TEXT    NOT NULL DEFAULT '',
    heat        INTEGER NOT NULL DEFAULT 1,
    content_md  TEXT    NOT NULL DEFAULT '',
    content_seg TEXT    NOT NULL DEFAULT '',   -- 只含 name+summary 的预分词（不含正文）
    scope       TEXT    NOT NULL,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'active'
);
CREATE INDEX IF NOT EXISTS idx_l2_scope ON l2_scene(scope, status, heat DESC);

-- L2 场景的 FTS 索引（向量索引只覆盖 L1）
CREATE VIRTUAL TABLE IF NOT EXISTS l2_scene_fts USING fts5(
    content_seg,
    content='l2_scene',
    content_rowid='id',
    tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS trg_l2_ai AFTER INSERT ON l2_scene BEGIN
    INSERT INTO l2_scene_fts(rowid, content_seg) VALUES (new.id, new.content_seg);
END;
CREATE TRIGGER IF NOT EXISTS trg_l2_ad AFTER DELETE ON l2_scene BEGIN
    INSERT INTO l2_scene_fts(l2_scene_fts, rowid, content_seg)
    VALUES ('delete', old.id, old.content_seg);
END;
CREATE TRIGGER IF NOT EXISTS trg_l2_au AFTER UPDATE ON l2_scene BEGIN
    INSERT INTO l2_scene_fts(l2_scene_fts, rowid, content_seg)
    VALUES ('delete', old.id, old.content_seg);
    INSERT INTO l2_scene_fts(rowid, content_seg) VALUES (new.id, new.content_seg);
END;

-- ---- L2 待巩固标记（只登记脏场景名，由调度器去抖后入队） ----
CREATE TABLE IF NOT EXISTS l2_pending_scene (
    scope        TEXT NOT NULL,
    scene_name   TEXT NOT NULL,
    first_dirty_at TEXT NOT NULL,
    last_dirty_at  TEXT NOT NULL,
    PRIMARY KEY (scope, scene_name)
);

-- ---- L3 画像 / 准则 ----
CREATE TABLE IF NOT EXISTS l3_persona (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scope       TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    content_md  TEXT    NOT NULL DEFAULT '',
    version     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL,
    UNIQUE(scope, kind)
);

-- ---- 持久化任务队列 ----
CREATE TABLE IF NOT EXISTS memory_task (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id       TEXT    NOT NULL UNIQUE,
    kind          TEXT    NOT NULL,
    session_id    TEXT,
    scope         TEXT    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'pending',
    attempts      INTEGER NOT NULL DEFAULT 0,
    next_retry_at TEXT,
    started_at    TEXT,
    payload_json  TEXT    NOT NULL DEFAULT '{}',
    last_error    TEXT,
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_ready ON memory_task(status, next_retry_at, id);
CREATE INDEX IF NOT EXISTS idx_task_sess  ON memory_task(session_id, status);
-- 抽取游标 = 该会话所有 done 的 extract_l1 任务中 MAX(payload_json.to_id)
CREATE INDEX IF NOT EXISTS idx_task_cursor ON memory_task(kind, session_id, status, id);
"""