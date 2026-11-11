"""记忆管线：L1 抽取、去重判定、L2 场景规划、L3 人格沉淀四个 LLM 阶段及其输出归一化，本模块不直接读写数据库。"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .utils import parse_json, render_template

logger = logging.getLogger(__name__)

MIN_STORED_PRIORITY = 70  # priority 小于该值的记忆直接丢弃、不入库
GLOBAL_INSTRUCTION_PRIORITY = -1  # priority == 该值表示「全局死命令」

VALID_ATOM_TYPES = {"work_fact", "work_task", "work_method", "work_artifact"}
VALID_ACTIONS = {"store", "update", "merge", "skip"}
VALID_SCENE_ACTIONS = {"update", "merge", "create", "skip"}


class MemoryPipeline:
    """持有 llm_client 与 config，提供四个 LLM 阶段及各自的输出归一化。

    不缓存跨阶段数据，同一实例可被多个后台任务并发调用。
    """

    def __init__(self, llm_client: Any, config: dict | None = None) -> None:
        self.llm_client = llm_client
        self.config = config or {}

    async def invoke(self, prompt: str, temperature: float = 0.3) -> str:
        """把 prompt 作为单条 user 消息发给 LLM（关闭推理），返回回复正文，响应非 dict 时返回空串。"""
        response = await self.llm_client.invoke(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            reasoning_enabled=False,
        )
        if not isinstance(response, dict):
            return ""
        return response.get("content", "") or ""

    @staticmethod
    def normalize_extraction(scenes: Any) -> list[dict]:
        """把 L1 抽取结果格式化成 ``[{scene_name, message_ids, memories}]``。

        丢弃缺场景名、缺正文与低优先级的项，非法 type 归一为 work_fact；入参不是列表时返回空列表。
        """
        if not isinstance(scenes, list):
            return []
        result: list[dict] = []
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            scene_name = str(scene.get("scene_name") or "").strip()
            if not scene_name:
                continue
            raw_ids = scene.get("message_ids")
            scene_msg_ids = (
                [int(i) for i in raw_ids if isinstance(i, (int, float))]
                if isinstance(raw_ids, list)
                else []
            )

            memories: list[dict] = []
            for mem in scene.get("memories") or []:
                if not isinstance(mem, dict):
                    continue
                content = str(mem.get("content") or "").strip()
                if not content:  # 空内容视为幻觉
                    continue
                atom_type = str(mem.get("type") or "").strip()
                if atom_type not in VALID_ATOM_TYPES:
                    atom_type = "work_fact"  # 非法 type 降级

                try:
                    priority = int(mem.get("priority", 0))
                except (TypeError, ValueError):
                    priority = 0
                if priority != GLOBAL_INSTRUCTION_PRIORITY and priority < MIN_STORED_PRIORITY:
                    continue  # 低优先级直接丢弃

                src = mem.get("source_message_ids")
                source_ids = (
                    [int(i) for i in src if isinstance(i, (int, float))]
                    if isinstance(src, list)
                    else []
                )
                metadata = mem.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}

                memories.append(
                    {
                        "content": content,
                        "type": atom_type,
                        "priority": priority,
                        "source_message_ids": source_ids,
                        "metadata": metadata,
                    }
                )
            if memories:
                result.append(
                    {"scene_name": scene_name, "message_ids": scene_msg_ids, "memories": memories}
                )
        return result

    async def extract_l1(
        self,
        messages: list[dict],
        existing_scenes: list[str] | None = None,
    ) -> Optional[list[dict]]:
        """渲染 L1 抽取模板、调用 LLM 同时完成情境切分与记忆提取，返回格式化场景列表。

        返回 None 表示模板渲染、调用或解析失败；返回 ``[]`` 表示成功但没有可存记忆。
        """
        prompt_mode = self.config.get("prompt_mode", "code")
        template_name = f"memory_extract_{prompt_mode}.md"

        lines = []
        for idx, msg in enumerate(messages):
            role = str(msg.get("role", "unknown")).upper()
            content = str(msg.get("content") or "")
            if content:
                lines.append(f"[{idx}] {role}: {content}")
        messages_text = "\n".join(lines) if lines else "(空)"

        scene_index = "（无）"
        if existing_scenes:
            scene_index = "\n".join(f"- {name}" for name in existing_scenes)

        try:
            prompt = render_template(
                template_name, {"messages": messages_text, "existing_scenes": scene_index}
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("抽取模板渲染失败: %s", exc)
            return None

        try:
            raw = await self.invoke(prompt)
        except Exception as exc:  # LLM 故障不得影响主流程
            logger.warning("L1 抽取调用失败: %s", exc)
            return None

        parsed = parse_json(raw, list)
        if parsed is None:
            return None
        return self.normalize_extraction(parsed)

    @staticmethod
    def normalize_dedup(decisions: Any, new_atoms: list[dict]) -> list[dict]:
        """把去重判定规整成与 new_atoms 等长的列表，缺失项与非法 action 一律按 store 兜底。"""
        fallback = [
            {
                "new_id": a["atom_id"],
                "action": "store",
                "target_ids": [],
                "merged_content": None,
                "merged_type": None,
                "merged_priority": None,
            }
            for a in new_atoms
        ]
        if not isinstance(decisions, list):
            return fallback

        by_id: dict[str, dict] = {}
        for item in decisions:
            if not isinstance(item, dict):
                continue
            new_id = str(item.get("new_id") or "").strip()
            if not new_id:
                continue
            action = str(item.get("action") or "").strip().lower()
            if action not in VALID_ACTIONS:
                action = "store"  # 非法 action 一律当作 store
            merged_type = item.get("merged_type")
            if merged_type not in VALID_ATOM_TYPES:
                merged_type = None
            by_id[new_id] = {
                "action": action,
                "target_ids": [str(t) for t in (item.get("target_ids") or []) if t],
                "merged_content": item.get("merged_content"),
                "merged_type": merged_type,
                "merged_priority": item.get("merged_priority"),
            }

        return [by_id.get(a["atom_id"], fallback[i]) for i, a in enumerate(new_atoms)]

    async def judge_dedup(
        self,
        new_atoms: list[dict],
        candidates: dict[str, list[dict]],
    ) -> list[dict]:
        """去重判定：把新原子与各自候选池组成一份 payload 交 LLM 批量判定，返回与 new_atoms 等长的判定列表。

        没有候选、模板渲染失败、调用失败或解析失败时，全部按 store 兜底。
        """
        if not new_atoms:
            return []
        if not any(candidates.get(a["atom_id"]) for a in new_atoms):
            # 候选池全空 → 无需 LLM，全部 store
            return self.normalize_dedup(None, new_atoms)

        payload = []
        for atom in new_atoms:
            pool = candidates.get(atom["atom_id"]) or []
            payload.append(
                {
                    "new_id": atom["atom_id"],
                    "content": atom["content"],
                    "type": atom["type"],
                    "priority": atom["priority"],
                    "candidates": [
                        {
                            "id": c.get("atom_id"),
                            "content": c.get("content"),
                            "type": c.get("type"),
                            "priority": c.get("priority"),
                        }
                        for c in pool
                    ],
                }
            )

        try:
            prompt = render_template(
                "memory_dedup.md", {"payload": json.dumps(payload, ensure_ascii=False, indent=2)}
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("去重模板渲染失败: %s", exc)
            return self.normalize_dedup(None, new_atoms)

        try:
            raw = await self.invoke(prompt, temperature=0.1)
        except Exception as exc:
            logger.warning("去重调用失败，降级为全部 store: %s", exc)
            return self.normalize_dedup(None, new_atoms)

        parsed = parse_json(raw, list)
        if parsed is None:
            return self.normalize_dedup(None, new_atoms)  # fallbackStoreAll
        return self.normalize_dedup(parsed, new_atoms)

    # ------------------------------------------------------------ L2 场景巩固
    @staticmethod
    def scene_cap_strategy(count: int, max_scenes: int) -> str:
        """按场景数与上限返回 ``red``（已达上限）/ ``orange``（差一个）/ ``yellow``。"""
        if count >= max_scenes:
            return "red"
        if count >= max_scenes - 1:
            return "orange"
        return "yellow"

    @staticmethod
    def normalize_scene_plan(plan: Any, known_ids: set[str], strategy: str) -> list[dict]:
        """格式化 L2 合并方案，非法 action、无效目标与超限的 create 一律改为 skip。"""
        if not isinstance(plan, list):
            return []
        result: list[dict] = []
        for item in plan:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action") or "").strip().lower()
            if action not in VALID_SCENE_ACTIONS:
                action = "skip"
            targets = [
                str(t) for t in (item.get("target_ids") or item.get("target_scene_ids") or []) if t
            ]
            # 只保留真实存在的目标，防止 LLM 编造 ID
            targets = [t for t in targets if t in known_ids]
            name = str(item.get("name") or "").strip()
            summary = str(item.get("summary") or "").strip()
            content_md = str(item.get("content_md") or "").strip()

            if action == "update" and not targets:
                action = "skip"
            if action == "merge" and len(targets) < 2:
                action = "skip"
            if action == "create" and strategy in ("red", "orange"):
                # 已达上限：禁止新建
                action = "skip"
            if action != "skip" and (not content_md or not name):
                action = "skip"

            result.append(
                {
                    "action": action,
                    "target_ids": targets,
                    "name": name,
                    "summary": summary,
                    "content_md": content_md,
                }
            )
        return result

    async def plan_scene_merge(
        self,
        grouped: dict[str, list[dict]],
        existing: list[dict],
        strategy: str,
    ) -> list[dict]:
        """把待巩固原子与现有场景概况交给 LLM，返回规整后的 L2 方案；渲染、调用或解析失败返回空列表。"""
        max_chars = int(self.config.get("max_scene_chars", 1500))
        known_ids = {s["scene_id"] for s in existing}
        scene_stats = (
            "\n".join(
                f"- id={s['scene_id']} heat={s.get('heat', 1)} summary={s.get('summary') or ''}"
                f" | name={s.get('name')}"
                for s in existing
            )
            or "（当前场景库为空）"
        )
        stats_block = f"策略：**{strategy}**（当前 {len(existing)} 个场景）\n{scene_stats}"
        payload = json.dumps(grouped, ensure_ascii=False, indent=2)

        try:
            prompt = render_template(
                "memory_scene_merge.md",
                {"scene_stats": stats_block, "payload": payload, "max_scene_chars": max_chars},
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("场景合并模板渲染失败: %s", exc)
            return []
        try:
            raw = await self.invoke(prompt, temperature=0.2)
        except Exception as exc:
            logger.warning("场景合并调用失败（不修改场景库）: %s", exc)
            return []
        return self.normalize_scene_plan(parse_json(raw, list), known_ids, strategy)

    # ------------------------------------------------------------ L3 人格沉淀
    @staticmethod
    def normalize_persona(data: Any, max_chars: int) -> Optional[str]:
        """从 L3 模型输出取新正文，action 为 keep、内容为空或入参非 dict 时返回 None，超长硬截到 max_chars。"""
        if not isinstance(data, dict):
            return None
        action = str(data.get("action") or "rewrite").strip().lower()
        if action == "keep":
            return None
        content = str(data.get("content_md") or "").strip()
        if not content:
            return None
        if len(content) > max_chars:
            content = content[:max_chars]  # 超限就硬截，绝不放过
        return content

    async def consolidate_persona(
        self, kind: str, current: str, scene_digest: str
    ) -> Optional[str]:
        """用现有正文与场景摘要调一次 LLM 增量更新 L3，返回新正文；无需修改或失败返回 None。

        doctrine 的长度上限取 ``max_persona_chars`` 与 ``max_doctrine_chars`` 中较小者。
        """
        max_chars = int(self.config.get("max_persona_chars", 2000))
        if kind == "doctrine":
            max_chars = min(max_chars, int(self.config.get("max_doctrine_chars", 1200)))
        try:
            prompt = render_template(
                "memory_persona.md",
                {
                    "current_persona": current or "（尚无档案，本批将是首次沉淀）",
                    "scene_digest": scene_digest or "（无）",
                    "max_persona_chars": max_chars,
                },
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("人设模板渲染失败: %s", exc)
            return None
        try:
            raw = await self.invoke(prompt, temperature=0.2)
        except Exception as exc:
            logger.warning("人设调用失败（保持原样）: %s", exc)
            return None
        return self.normalize_persona(parse_json(raw, dict), max_chars)
