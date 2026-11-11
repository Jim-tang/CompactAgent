"""检索层：把 query 经 FTS5 与向量双路召回、RRF 融合、reranker 精排后，输出注入块文本。

向量不可用时只走 FTS5，FTS5 不可用时返回空结果。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .utils import estimate_tokens, rrf_fuse

logger = logging.getLogger(__name__)


class Retriever:
    """L1 与 L2 各自独立做 RRF + 精排 + 阈值（两套分数互不调和），并渲染 persona 与场景索引注入块。"""

    def __init__(self, store, vector, config: dict | None = None) -> None:
        self.store = store
        self.vector = vector
        self.config = config or {}
        self.rrf_k = int(self.config.get("rrf_k", 60))
        self.recall_top_k = int(self.config.get("recall_top_k", 20))
        self.final_top_k = int(self.config.get("final_top_k", 4))
        self.rerank_candidates = int(self.config.get("rerank_candidates", 20))
        self.l2_recall_enabled = bool(self.config.get("l2_recall_enabled", True))  # L2 场景召回开关，关闭后只检索 L1。
        self.l1_rerank_threshold = float(self.config.get("l1_rerank_threshold", 0.2))
        self.l2_rerank_threshold = float(self.config.get("l2_rerank_threshold", 0.2))
        self.rerank_relative_ratio = float(self.config.get("rerank_relative_ratio", 0.5)) # 相对阈值（与第一名分数的差距）
        self.last_mode = "none"

    @property
    def fts5_available(self) -> bool:
        """每次调用时读取 store 是否支持 FTS5，不在构造期缓存。"""
        return bool(getattr(self.store, "fts5_available", False))

    # ------------------------------------------------------------ 召回
    async def _recall_scenes(self, query: str, scope: Optional[str], limit: int) -> list[dict]:
        """用 FTS5 召回 L2 场景行；L2 召回关闭时返回空列表，异常只记日志。"""
        if not self.l2_recall_enabled:
            return []
        rows: list[dict] = []
        try:
            rows = await asyncio.to_thread(
                self.store.search_scenes_fts, query, scope=scope, limit=limit
            )
        except Exception as exc:
            logger.warning("L2 关键词召回失败（不影响其它腿）: %s", exc)
            rows = []
        return rows

    async def _recall_scenes_vector(self, query: str, limit: int) -> list[dict]:
        """用向量库召回 L2 场景 id；L2 召回关闭、向量不可用或异常时返回空列表。"""
        if not self.l2_recall_enabled:
            return []
        if not bool(getattr(self.vector, "available", False)):
            return []

        try:
            raw = await asyncio.to_thread(self.vector.search_scenes, query, limit)
        except Exception as exc:
            logger.warning("L2 向量召回失败（降级为纯 FTS5）: %s", exc)
            return []
        out = []
        for row in raw or []:
            scene_id = row.get("scene_id")
            if not scene_id:
                continue
            out.append({"scene_id": scene_id})
        return out

    async def recall(
        self, query: str, scope: Optional[str] = None
    ) -> tuple[list[dict], list[dict], str]:
        """对 L1 与 L2 各做一次 RRF 融合召回，返回 ``(L1 原子候选, L2 场景候选, 模式)``。

        两组候选都带 ``layer`` 字段（``L1`` / ``L2``），便于调用方按层渲染。
        模式取 ``hybrid`` / ``vector`` / ``fts`` / ``none``，空 query 直接返回两个空列表与 ``none``；
        每一路召回失败只记日志并按空结果处理。
        """
        if not query or not query.strip():
            self.last_mode = "none"
            return [], [], "none"

        limit = max(self.recall_top_k, self.final_top_k)

        # ---- L1：FTS5 + 向量
        fts_rows: list[dict] = []
        try:
            fts_rows = await asyncio.to_thread(
                self.store.search_atoms_fts, query, scope=scope, limit=limit
            )
        except Exception as exc:
            logger.warning("FTS5 召回失败（不影响向量路）: %s", exc)
            fts_rows = []

        vec_rows: list[dict] = []
        vec_ok = bool(getattr(self.vector, "available", False))
        if vec_ok:
            try:
                vec_rows = await self._vector_search(query, limit)
            except Exception as exc:
                logger.warning("向量召回失败（降级为纯 FTS5）: %s", exc)
                vec_rows = []

        fts_ids = [r["atom_id"] for r in fts_rows]
        vec_ids = [r["atom_id"] for r in vec_rows]
        fused = rrf_fuse([fts_ids, vec_ids], k=self.rrf_k)

        if fts_ids and vec_ids:
            mode = "hybrid"
        elif vec_ids:
            mode = "vector"
        elif fts_ids:
            mode = "fts"
        else:
            mode = "fts" if self.fts5_available else "none"
        self.last_mode = mode

        atoms: list[dict] = []
        if fused:
            by_id = {r["atom_id"]: r for r in fts_rows + vec_rows}
            missing = [aid for aid in fused if aid not in by_id]
            if missing:
                for row in await asyncio.to_thread(
                    self.store.get_atoms_by_ids, missing, scope=scope
                ):
                    by_id.setdefault(row["atom_id"], row)
            for atom_id, score in sorted(fused.items(), key=lambda x: x[1], reverse=True):
                row = by_id.get(atom_id)
                if not row:
                    continue
                item = dict(row)
                item["layer"] = "L1"
                item["rrf_score"] = score
                item["recall_sources"] = (
                    ("fts" if atom_id in fts_ids else "")
                    + ("+vec" if atom_id in vec_ids else "")
                ).strip("+")
                atoms.append(item)

        # ---- L2：FTS5 + 向量（独立融合，不掺 L1 名次）
        scenes = await self._recall_scene_candidates(query, scope, limit)
        return atoms, scenes, mode

    async def _recall_scene_candidates(
        self, query: str, scope: Optional[str], limit: int
    ) -> list[dict]:
        """把 L2 的 FTS5 与向量两路结果独立 RRF 融合，补齐正文后返回候选项列表。"""
        fts_rows = await self._recall_scenes(query, scope, limit)
        vec_rows = await self._recall_scenes_vector(query, limit)
        if not fts_rows and not vec_rows:
            return []
        fts_ids = [r["scene_id"] for r in fts_rows]
        vec_ids = [r["scene_id"] for r in vec_rows]
        fused = rrf_fuse([fts_ids, vec_ids], k=self.rrf_k)
        if not fused:
            return []
        by_id = {r["scene_id"]: r for r in fts_rows}
        missing = [sid for sid in fused if sid not in by_id]
        if missing:
            try:
                for row in await asyncio.to_thread(
                    self.store.get_scenes_by_ids, missing, scope=scope
                ):
                    by_id.setdefault(row["scene_id"], row)
            except Exception as exc:
                logger.warning("L2 场景补全失败: %s", exc)
        out = []
        for scene_id, score in sorted(fused.items(), key=lambda x: x[1], reverse=True):
            row = by_id.get(scene_id)
            if not row:
                continue
            out.append(self._scene_candidate(row, score, scene_id, fts_ids, vec_ids))
        return out

    @staticmethod
    def _scene_candidate(
        row: dict, score: float, scene_id: str,
        fts_ids: list[str], vec_ids: list[str],
    ) -> dict:
        """把场景行转成与原子同形的候选项，``layer`` 标为 L2。

        ``content`` 只放 name + summary 的索引视图，并提示正文需另行按 id 拉取。
        """
        name = row.get("name") or ""
        summary = row.get("summary") or ""
        return {
            "atom_id": scene_id,
            "layer": "L2",
            "content": f"[场景索引] {name}：{summary}（正文请用 get_scene 拉取）",
            "summary": summary,
            "name": name,
            "type": "scene",
            "priority": None,
            "scope": row.get("scope"),
            "session_id": None,
            "scene_id": scene_id,
            "created_at": row.get("created_at"),
            "status": "active",
            "rrf_score": score,
            "recall_sources": (
                ("l2-fts" if scene_id in fts_ids else "")
                + ("+l2-vec" if scene_id in vec_ids else "")
            ).strip("+"),
        }

    def _apply_threshold(self, items: list[dict], absolute: float) -> list[dict]:
        """精排按绝对阈值与相对阈值（第一名分数的一定比例）过滤条目，只用绝对阈值时弱 query 会放进大量噪声。

        ``rerank_score`` 缺失的条目视为通过；全部缺分数时原样返回。
        """
        if not items:
            return []
        scored = [i for i in items if isinstance(i.get("rerank_score"), float)]
        if not scored:
            return items
        top = max(i["rerank_score"] for i in scored)
        cut = max(absolute, top * self.rerank_relative_ratio)
        return [i for i in items if i.get("rerank_score") is None or i["rerank_score"] >= cut]

    async def _vector_search(self, query: str, limit: int) -> list[dict]:
        """在线程池里跑向量检索，返回只含 atom_id 与元数据的候选项列表（不含 score）。"""

        raw = await asyncio.to_thread(self.vector.search, query, limit)
        out = []
        for row in raw or []:
            atom_id = row.get("atom_id")
            if not atom_id:
                continue
            meta = row.get("metadata") or {}
            out.append(
                {
                    "atom_id": atom_id,
                    "content": row.get("content") or meta.get("content", ""),
                    "type": meta.get("type", ""),
                    "priority": meta.get("priority"),
                    "scope": meta.get("scope"),
                    "session_id": meta.get("session_id"),
                    "created_at": meta.get("created_at"),
                    "scene_id": meta.get("scene_id"),
                    "status": "active",
                }
            )
        return out

    async def retrieve(
        self, query: str, top_k: Optional[int] = None, scope: Optional[str] = None
    ) -> list[dict]:
        """检索主入口：对 L1 与 L2 分别做 RRF、精排与阈值筛选，不足 top_k 时用 L2 结果补位。"""
        top_k = int(top_k or self.final_top_k)
        atoms, scenes, mode = await self.recall(query, scope=scope)

        l1_ok = await self._rank_and_filter(
            query, atoms, self.l1_rerank_threshold, max(top_k, 1), mode
        )
        if len(l1_ok) >= top_k or not scenes:
            return l1_ok[:top_k]

        l2_ok = await self._rank_and_filter(
            query, scenes, self.l2_rerank_threshold, top_k, mode
        )
        merged = list(l1_ok)
        seen = {c.get("scene_id") for c in merged}
        for item in l2_ok:
            if len(merged) >= top_k:
                break
            if item.get("scene_id") in seen:
                continue
            merged.append(item)
        return merged

    async def _rank_and_filter(
        self, query: str, items: list[dict], threshold: float,
        want: int, mode: str,
    ) -> list[dict]:
        """把一个层的候选精排后返回，并保证每条都带 ``layer`` 与 ``retrieval_mode``。"""
        if not items:
            return []
        head = items[: self.rerank_candidates]
        if self.vector is None:
            return self._finalize(head, want, mode)

        try:
            ranked = await asyncio.to_thread(self.vector.rerank, query, head, len(head))
        except Exception as exc:
            logger.warning("精排失败（保持 RRF 顺序）: %s", exc)
            ranked = [dict(i) for i in head]
        kept = self._apply_threshold(ranked, threshold)
        return self._finalize(kept[:want], want, mode)

    @staticmethod
    def _finalize(items: list[dict], want: int, mode: str) -> list[dict]:
        """给结果条目补上 ``retrieval_mode``，并在此处兜底补齐 ``layer``。"""
        out = []
        for item in items[:want]:
            item.setdefault("layer", "L1")
            item.setdefault("retrieval_mode", mode)
            out.append(item)
        return out

    async def build_persona_block(self, scope: Optional[str] = None) -> str:
        """读取该 scope 的 L3 人设正文，读取失败或无内容时返回空字符串。"""
        try:
            row = await asyncio.to_thread(self.store.get_persona, scope, "persona")
        except Exception as exc:
            logger.warning("L3 人设读取失败: %s", exc)
            return ""
        return (row or {}).get("content_md", "") if row else ""

    async def build_scene_index(self, scope: Optional[str] = None) -> str:
        """返回 ``- 场景名：摘要`` 形式的场景索引文本，摘要截到 80 字；无场景或读取失败返回空串。"""
        try:
            scenes = await asyncio.to_thread(
                self.store.list_scenes, scope=scope, status="active"
            )
        except Exception:
            return ""
        if not scenes:
            return ""
        lines = []
        for scene in scenes:
            name = scene.get("name") or ""
            summary = (scene.get("summary") or "").strip().replace("\n", " ")
            if len(summary) > 80:
                summary = summary[:80] + "…"
            lines.append(f"- {name}：{summary}" if summary else f"- {name}")
        return "\n".join(lines)

    def render_injection(
        self,
        persona: str,
        scene_index: str,
        memories: list[dict],
        token_budget: int = 300,
    ) -> str:
        """把人设、场景索引与 priority>=80 的记忆渲染进 ``<memory>`` 块，超出 token_budget 即截断。"""
        sections: list[str] = []

        if persona.strip():
            sections.append("### 长期人设\n" + persona.strip())

        if scene_index.strip():
            sections.append("### 情境索引\n" + scene_index.strip())

        if memories:
            high = [
                m for m in memories
                if isinstance(m.get("priority"), int) and m["priority"] >= 80
            ]
            lines = []
            for m in high:
                content = (m.get("content") or "").strip().replace("\n", " ")
                if content:
                    lines.append(f"- {content}")
            if lines:
                sections.append("### 相关长期记忆\n" + "\n".join(lines))

        if not sections:
            return ""

        header = "<memory>"
        footer = "</memory>"
        body = "\n\n".join(sections)
        block = f"{header}\n{body}\n{footer}"

        if estimate_tokens(block) <= token_budget:
            return block

        # 超预算时按节从后往前砍，保留最靠前的几节
        while sections and estimate_tokens(
            f"{header}\n" + "\n\n".join(sections) + f"\n{footer}"
        ) > token_budget:
            sections.pop()
        if not sections:
            return ""
        return f"{header}\n" + "\n\n".join(sections) + f"\n{footer}"

    async def build_context_block(
        self, scope: Optional[str] = None, token_budget: int = 300
    ) -> str:
        """返回只含 L3 人设与 L2 场景索引的固定上下文块，内容不随 query 变化，避免影响 prompt 缓存命中"""
        persona = await self.build_persona_block(scope)
        scene_index = await self.build_scene_index(scope)
        return self.render_injection(persona, scene_index, [], token_budget)
