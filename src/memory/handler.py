from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

from .pipeline import MemoryPipeline
from .store import MemoryStore
from .utils import now_iso, ulid

logger = logging.getLogger(__name__)

__all__ = [
    "MemoryHandlerContext",
    "MemoryHandler",
    "L1AtomHandler",
    "L2SceneHandler",
    "L3PersonaHandler",
]


@dataclass(frozen=True)
class MemoryHandlerContext:
    """三个处理器共用的只读依赖：store / scope / config / pipeline。"""

    store: MemoryStore
    scope: str
    config: dict
    pipeline: MemoryPipeline


class MemoryHandler:
    """后台处理器基类：持有共享上下文，并以属性暴露其四个字段。"""

    def __init__(self, ctx: MemoryHandlerContext) -> None:
        self.ctx = ctx

    # ---- 共享依赖的短别名
    @property
    def scope(self) -> str:
        return self.ctx.scope

    @property
    def config(self) -> dict:
        return self.ctx.config

    @property
    def pipeline(self) -> MemoryPipeline:
        return self.ctx.pipeline

    @property
    def store(self) -> MemoryStore:
        return self.ctx.store

    @staticmethod
    def async_call(fn, *args, **kwargs):
        """把 store 的同步读写挪到线程执行。"""
        return asyncio.to_thread(fn, *args, **kwargs)


class L1AtomHandler(MemoryHandler):
    """L1 原子记忆抽取任务：取 L0 未抽取消息，抽取 L1 原子，去重后落库并登记脏场景。"""

    def __init__(self, ctx: MemoryHandlerContext, vector: Any = None) -> None:
        super().__init__(ctx)
        self.vector = vector
        self.last_extract_at: Optional[str] = None

    async def __call__(self, payload: dict, task: dict) -> Optional[dict]:
        return await self.handle(payload, task)

    async def handle(self, payload: dict, task: dict) -> Optional[dict]:
        """取 payload 中 session_id 之后未抽取的 user/assistant 消息，抽取并落库 L1 原子。

        成功时返回 ``{"to_id": 本次消费到的最大 L0 行 id}``，worker 据此推进游标；
        失败返回 None，任务仍算完成但游标不前进，这批消息留待下次重抽。
        """
        session_id = payload.get("session_id")
        if not session_id:
            return None
        from_id = int(payload.get("from_id", 0))
        # 工具结果只作为 L0 事实留存，不进入 L1 抽取：
        rows = await self.async_call(
            self.store.query,
            "SELECT id, role, content FROM l0_conversation"
            " WHERE session_id = ? AND id > ? AND role IN ('user', 'assistant')"
            " ORDER BY id",
            (session_id, from_id),
        )
        if not rows:
            return None

        # 游标推进到「本次真正消费掉的最后一行」，而不是入队时的快照。
        # 由于 tool 行不参与抽取，rows 已排除 role='tool'，所以 tool 行 id 更大时也会被跳过
        to_id = max(int(r["id"]) for r in rows)
        messages = [{"role": r["role"], "content": r["content"]} for r in rows]

        # 已有场景名
        existing = [
            r["name"]
            for r in await self.async_call(
                self.store.query,
                "SELECT name FROM l2_scene WHERE scope = ? AND status = 'active'",
                (self.scope,),
            )
        ]

        scenes = await self.pipeline.extract_l1(messages, existing)
        if scenes is None:
            # LLM/解析失败：不落库、不推进游标，让这批消息留待下次重抽
            logger.warning("L1 抽取失败，游标保持不变（session=%s to_id=%s）",
                           session_id, to_id)
            return None
        if not scenes:
            # 成功但没有可存记忆：游标可以安全前进
            return {"to_id": to_id}

        # 摊平成待写原子
        new_atoms: list[dict] = []
        for scene in scenes:
            for mem in scene["memories"]:
                new_atoms.append(
                    {
                        "atom_id": ulid(),
                        "type": mem["type"],
                        "content": mem["content"],
                        "priority": mem["priority"],
                        "source_message_ids": mem["source_message_ids"],
                        "metadata": mem["metadata"],
                        "scope": self.scope,
                        "session_id": session_id,
                        "scene_name": scene["scene_name"],
                        # l1_atom.scene_id 存的是场景名（抽取阶段的天然分组键），
                        # 供 L2 分组查询使用；与 l2_scene.scene_id(ULID) 不同命名空间。
                        "scene_id": scene["scene_name"],
                    }
                )
        if not new_atoms:
            return {"to_id": to_id}

        # 阶段 1：候选召回（不用 LLM）
        candidates = await self._recall_dedup_candidates(new_atoms)

        # 阶段 2：批量判定（LLM；失败 → 全部 store）
        decisions = await self.pipeline.judge_dedup(new_atoms, candidates)

        stats = await self.async_call(self.store.apply_decisions, new_atoms, decisions)
        await self._index_atoms(new_atoms, decisions)

        self.last_extract_at = now_iso()
        logger.info("L1 抽取完成: %s", stats)

        # L1 落库后只登记脏场景，真正的 L2 入队交给调度循环或会话结束的强制巩固。
        scene_names = sorted({a["scene_name"] for a in new_atoms})
        if scene_names:
            await self.async_call(self.store.mark_scenes_dirty, self.scope, scene_names)
        return {"to_id": to_id}

    async def _recall_dedup_candidates(
        self, new_atoms: list[dict]
    ) -> dict[str, list[dict]]:
        """为每条新记忆召回候选旧记忆（只用检索、不调 LLM，不做判断）。"""
        top_k = int(self.config.get("dedup_candidate_top_k", 5))
        batch_ids = {a["atom_id"] for a in new_atoms}
        result: dict[str, list[dict]] = {}
        for atom in new_atoms:
            pool: list[dict] = []
            try:
                hits = await self.async_call(
                    self.store.search_atoms_fts,
                    atom["content"],
                    scope=self.scope,
                    limit=top_k + len(batch_ids),
                )
                for hit in hits:
                    if hit["atom_id"] in batch_ids:
                        continue  # 排除本批次自身，否则新记忆会和自己判重
                    pool.append(hit)
                    if len(pool) >= top_k:
                        break
            except Exception as exc:
                logger.warning("候选召回失败（将全部 store）: %s", exc)
                pool = []
            result[atom["atom_id"]] = pool
        return result

    async def _index_atoms(self, new_atoms: list[dict], decisions: list[dict]) -> None:
        """把未跳过的新原子（幂等 upsert）写进向量索引，失败只记日志。

        向量索引不可用或异常时直接返回，FTS5 检索仍可用。
        """
        if self.vector is None or not self.vector.available:
            return
        try:
            store_now = [a for a, d in zip(new_atoms, decisions) if d.get("action") != "skip"]
            if not store_now:
                return
            ids = [a["atom_id"] for a in store_now]
            texts = [a["content"] for a in store_now]
            metas = [
                {
                    "type": a.get("type", ""),
                    "priority": int(a.get("priority") or 0),
                    "scope": a.get("scope", self.scope),
                    "session_id": a.get("session_id", ""),
                    "created_at": now_iso(),
                }
                for a in store_now
            ]
            ok = await asyncio.to_thread(self.vector.upsert, ids, texts, metas)
            if ok:
                logger.info("向量索引已更新: %d 条", len(ids))
        except Exception as exc:
            logger.warning("向量索引入库失败（sqlite 仍是权威源）: %s", exc)


class L2SceneHandler(MemoryHandler):
    """L2 场景记忆巩固任务：取脏场景对应的 L1 原子，经一次 LLM 规划后归并成 L2 场景。

    有实际改动时返回 `{"changed_scenes": N}`，由 worker 写回任务 payload。
    """

    def __init__(
        self,
        ctx: MemoryHandlerContext,
        vector: Any = None,
        mirror_dir: Any = None,
    ) -> None:
        super().__init__(ctx)
        self.vector = vector
        self.mirror_dir = mirror_dir
        self.last_consolidate_at: Optional[str] = None

    async def __call__(self, payload: dict, task: dict) -> Optional[dict]:
        return await self.handle(payload, task)

    async def handle(self, payload: dict, task: dict) -> Optional[dict]:
        """按场景名取 L1 原子，经一次 LLM 规划后落库，返回 ``{"changed_scenes": N}`` 或 None。

        场景名默认在执行期从脏场景登记表读取，payload 显式给了 ``scene_names`` 时以它为准；
        没有可巩固的场景、方案为空或无实际改动时返回 None。
        """
        claimed = await self._resolve_dirty_scenes(payload)
        if not claimed:
            return None

        # 按场景名收集本批原子（scene_id 先留空，一律以「场景名」为对外键）
        grouped: dict[str, list[dict]] = {}
        for name in claimed:
            rows = await self.async_call(
                self.store.query,
                "SELECT atom_id, content, type, priority FROM l1_atom"
                " WHERE scope = ? AND status = 'active' AND scene_id = ?",
                (self.scope, name),
            )
            if rows:
                grouped[name] = [dict(r) for r in rows]

        if not grouped:
            # 原子在排队期间被软删：脏标记已无对应素材，摘掉后才不会每轮空转
            await self.async_call(self.store.clear_pending_scenes, self.scope, claimed)
            return None

        existing = await self.async_call(self.store.list_scenes, scope=self.scope)
        max_scenes = int(self.config.get("max_scenes", 50))
        strategy = self.pipeline.scene_cap_strategy(len(existing), max_scenes)
        logger.info("L2 巩固: %d 个场景待归并, 现有 %d, 策略=%s",
                    len(grouped), len(existing), strategy)

        plan = await self.pipeline.plan_scene_merge(grouped, existing, strategy)
        if not plan:
            # 方案为空时（这批材料不值得入库），摘掉脏标记
            await self.async_call(self.store.clear_pending_scenes, self.scope, claimed)
            logger.info("L2 巩固: 方案为空，场景库不变")
            return None

        changed = 0
        retry_names: set[str] = set()
        for item in plan:
            if item.get("action") == "skip":
                continue
            try:
                await self._apply_scene_plan(item, existing)
            except Exception as exc:
                # 若发生 LLM 超时/输出不合规等异常，单条失败不拖垮整批，去抖到期后要再试一次，避免这批原子被丢弃
                logger.warning("L2 巩固: 落库失败（保留脏标记）: %s", exc)
                # 把 item 里的 target_ids 映射回场景名
                retry_names.update(self._target_names(item, existing))
                continue
            changed += 1
        self.last_consolidate_at = now_iso()

        # 正常消费（含空方案、全 skip）的素材摘掉脏标记；
        consumed = [n for n in claimed if n not in retry_names]
        if consumed:
            await self.async_call(self.store.clear_pending_scenes, self.scope, consumed)
        if retry_names:
            logger.warning("L2 巩固: %d 个场景落库失败，保留脏标记以便重试", len(retry_names))

        return {"changed_scenes": changed} if changed > 0 else None

    @staticmethod
    def _target_names(item: dict, existing: list[dict]) -> list[str]:
        """把方案条目的目标场景 ID 映射回场景名；指向新场景（create）时返回空列表。"""
        by_id = {s["scene_id"]: s["name"] for s in existing}
        return [by_id[t] for t in (item.get("target_ids") or []) if t in by_id]

    async def _resolve_dirty_scenes(self, payload: dict) -> list[str]:
        """确定本次要巩固的场景名：显式给定优先，否则取当前全部脏场景。"""
        explicit = [str(n) for n in (payload.get("scene_names") or []) if n]
        if explicit:
            return explicit
        pending = await self.async_call(self.store.list_pending_scenes, self.scope)
        return [row["scene_name"] for row in pending]

    async def _apply_scene_plan(self, item: dict, existing: list[dict]) -> None:
        """落库一条 L2 方案条目，action 为 skip 时什么都不做。"""
        action = item["action"]
        if action == "skip":
            return
        by_id = {s["scene_id"]: s for s in existing}

        if action == "update":
            target = by_id.get(item["target_ids"][0])
            if target is None:
                return
            await self.async_call(
                self.store.upsert_scene,
                scene_id=target["scene_id"], name=item["name"], summary=item["summary"],
                content_md=item["content_md"], scope=target.get("scope") or self.scope,
                heat=int(target.get("heat") or 1) + 1,
            )
            self._index_scene(target["scene_id"], item["name"], item["summary"])
        elif action == "merge":
            targets = [by_id[t] for t in item["target_ids"] if t in by_id]
            if len(targets) < 2:
                return
            # 保留热度最高的那个 ID
            keeper = max(targets, key=lambda s: int(s.get("heat") or 1))
            total_heat = sum(int(s.get("heat") or 1) for s in targets) + 1
            await self.async_call(
                self.store.upsert_scene,
                scene_id=keeper["scene_id"], name=item["name"], summary=item["summary"],
                content_md=item["content_md"], scope=keeper.get("scope") or self.scope,
                heat=total_heat,
            )
            for s in targets:
                if s["scene_id"] != keeper["scene_id"]:
                    await self.async_call(self.store.soft_delete_scene, s["scene_id"])
                    self._delete_scene_vector(s["scene_id"])
            self._index_scene(keeper["scene_id"], item["name"], item["summary"])
        elif action == "create":
            # `l1_atom.scene_id` 存的是场景名（抽取阶段的天然分组键），
            # 与 l2_scene.scene_id(ULID) 不是同一命名空间，因此这里不需要回填。
            scene_id = ulid()
            await self.async_call(
                self.store.upsert_scene,
                scene_id=scene_id, name=item["name"], summary=item["summary"],
                content_md=item["content_md"], scope=self.scope, heat=1,
            )
            self._index_scene(scene_id, item["name"], item["summary"])

    def _index_scene(self, scene_id: str, name: str, summary: str) -> None:
        """以 ``name + summary``（不含正文）为嵌入文本，把场景写进向量库，失败只记日志。"""
        if self.vector is None or not self.vector.available:
            return
        try:
            self.vector.upsert_scenes(
                ids=[scene_id],
                texts=[f"{name}\n{summary}"],
                metadatas=[{"scene_id": scene_id, "name": name,
                            "layer": "L2", "scope": self.scope}],
            )
        except Exception as exc:
            logger.warning("L2 向量写入失败（不影响 sqlite）: %s", exc)

    def _delete_scene_vector(self, scene_id: str) -> None:
        if self.vector is None or not self.vector.available:
            return
        try:
            self.vector.delete_scenes([scene_id])
        except Exception as exc:
            logger.warning("L2 向量删除失败: %s", exc)


class L3PersonaHandler(MemoryHandler):
    """L3画像沉淀任务：把 L2 场景摘要沉淀成 L3 persona / doctrine 并刷新 Markdown 镜像。"""

    def __init__(self, ctx: MemoryHandlerContext, mirror_dir: Any = None) -> None:
        super().__init__(ctx)
        self.mirror_dir = mirror_dir
        self.last_consolidate_at: Optional[str] = None

    async def __call__(self, payload: dict, task: dict) -> Optional[dict]:
        return await self.handle(payload, task)

    async def handle(self, payload: dict, task: dict) -> Optional[dict]:
        """把 L2 场景摘要压成 L3 persona / doctrine，无场景或内容无变化时不动库存。

        payload 支持 ``{"kind": "persona"}`` 与 ``{"kinds": [...]}`` 两种形状，
        每个 kind 独立调用一次 LLM 并独立写镜像，一个失败不影响其它。总是返回 None。
        """
        if payload.get("kinds"):
            kinds = [str(k) for k in payload["kinds"]]
        else:
            kinds = [payload.get("kind") or "persona"]
        scenes = await self.async_call(self.store.list_scenes, scope=self.scope)
        if not scenes:
            return None
        digest = "\n\n".join(
            f"## {s['name']}\n（heat={s.get('heat', 1)}）\n{s.get('content_md') or ''}"
            for s in scenes
        )
        for kind in kinds:
            current_row = await self.async_call(self.store.get_persona, self.scope, kind)
            current = (current_row or {}).get("content_md") or ""
            new_content = await self.pipeline.consolidate_persona(kind, current, digest)
            if not new_content:
                logger.info("L3 %s: 无变化，保持原样", kind)
                continue
            await self.async_call(self.store.upsert_persona, self.scope, kind, new_content)
            await self._sync_mirror(kind)
            self.last_consolidate_at = now_iso()
            logger.info("L3 %s 已更新 (%d 字符)", kind, len(new_content))
        return None

    async def _sync_mirror(self, kind: str) -> None:
        """把该 kind 的 L3 内容写成 ``{mirror_dir}/{kind}.md``，内容为空时删除该文件。

        镜像随时可重建，写入失败只记日志。
        """
        try:
            row = await self.async_call(self.store.get_persona, self.scope, kind)
            path = self.mirror_dir / f"{kind}.md"
            if not row or not (row.get("content_md") or "").strip():
                path.unlink(missing_ok=True)  # 档案为空 → 镜像同步为空
                return
            self.mirror_dir.mkdir(parents=True, exist_ok=True)
            label = "长期画像" if kind == "persona" else "行为准则"
            meta = (
                f"> scope: {self.scope} · version: {row.get('version', 1)}"
                f" · updated: {row.get('updated_at') or ''}"
            )
            body = (
                f"# {kind}（L3 {label}）\n\n"
                f"{meta}\n"
                f"> sqlite 是权威源，本文件只是可读镜像，可随时删除重建。\n\n"
                f"{row['content_md']}\n"
            )
            await asyncio.to_thread(path.write_text, body, encoding="utf-8")
        except Exception as exc:
            logger.warning("L3 镜像写入失败（sqlite 仍是权威源）: %s", exc)


class Handlers:
    """构建 L1/L2/L3 三个处理器并持有其共享上下文，通过 ``l1_handler`` 等属性暴露。"""

    def __init__(
        self,
        store: MemoryStore,
        scope: str,
        config: dict,
        pipeline: MemoryPipeline,
        vector: Any = None,
        mirror_dir: Any = None,
    ) -> None:
        ctx = MemoryHandlerContext(
            store=store, scope=scope, config=config, pipeline=pipeline
        )
        self.context = ctx
        self.l1_handler = L1AtomHandler(ctx, vector=vector)
        self.l2_handler = L2SceneHandler(ctx, vector=vector, mirror_dir=mirror_dir)
        self.l3_handler = L3PersonaHandler(ctx, mirror_dir=mirror_dir)

    @property
    def as_worker_handlers(self) -> dict[str, Any]:
        """返回 worker 入队时使用的 ``{任务 kind: 处理器}`` 映射。"""
        return {
            "extract_l1": self.l1_handler,
            "consolidate_l2": self.l2_handler,
            "consolidate_l3": self.l3_handler,
        }
