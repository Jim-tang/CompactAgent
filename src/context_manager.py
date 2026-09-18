import json
import os
import re
import yaml
import asyncio
import tiktoken
import httpx
from pathlib import Path
from typing import Dict, List, Optional
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from common_utils import get_workspace_tree, get_messages_text

COMPRESSION_PROMPT = """
You are a conversation summarization expert. Please compress the message list provided by the user into a dialogue summary written from the Assistant's perspective. The compression requirements are as follows:

1. **Perspective**: Use the first-person perspective of the Assistant. Describe what request you (the Assistant) received in the conversation, what actions you performed, what results you obtained, and what judgments you made.

2. **Concise and to the point**: Remove repetitions, pleasantries, or irrelevant details. Extract the following key information:
   - The user's core request or task objective
   - All tool names and key parameters invoked by the Assistant
   - Important data from tool call return results
   - Decisions, reasoning, or final responses made by the Assistant based on the above
   - Any problems encountered, exceptions, or content requiring clarification from the user

3. **Length**: Keep within 150 words. If there are many dialogue turns or key pieces of information, the length may be appropriately increased but should not exceed 300 words.
"""


class AgentContextManager:
    """
    Agent 上下文管理器

    管理 Agent 对话历史和 Skill 的渐进式披露
    支持根据 token 阈值或消息数量自动压缩历史，避免超出上下文窗口限制
    """
    def __init__(self, llm_client: AsyncOpenAI, compress_config: Optional[Dict] = None, skills_root: str = "skills"):
        self.llm_client = llm_client            # 用于生成摘要的 AsyncOpenAI 客户端
        self.compress_config = compress_config  # 上下文压缩配置字典
        self.skills_root = Path(skills_root)    # 技能文件夹根路径
        self.metadata_cache = {}                # 存储所有技能元数据的字典
        self.active_instructions = {}           # 存储当前已激活技能的完整指令的字典
        self.loaded_resources = {}              # 存储已加载的技能资源文件的字典
        self.history_messages = []              # 历史对话（不含 system）放到管理器内部维护
        self._semaphore = asyncio.Semaphore(3)  # 限制对 LLM 的并发调用数量
        self._load_all_metadata()

    def _load_all_metadata(self):
        """
        第一层：启动时加载所有技能的元数据（仅 name + description）

        遍历 skills_root 下的每个子目录，读取 SKILL.md 文件并解析 YAML 头，提取 name 和 description
        """
        if not self.skills_root.exists():
            return
        for skill_dir in self.skills_root.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            content = skill_md.read_text(encoding="utf-8")
            match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
            if match:
                try:
                    meta = yaml.safe_load(match.group(1))
                    self.metadata_cache[skill_dir.name] = {
                        "name": meta.get("name", skill_dir.name),
                        "description": meta.get("description", ""),
                        "path": skill_dir
                    }
                except:
                    pass

    def add_history(self, message: dict):
        """往管理器维护的历史对话列表中追加新的消息"""
        self.history_messages.append(message)

    async def prepare_message(self, system_prompt: str):
        """上下文处理的核心方法：先扩展 -> 后压缩，然后拼接历史记录返回完整消息列表"""
        # 动态扩展系统提示（渐进式披露）
        extended_system_msg = self.extend_system_prompt(system_prompt)
        full_messages = extended_system_msg + self.history_messages
        # 扩展后的系统提示只参与阈值计算但不进行压缩，从而让模型看到 Skills 的完整指令
        if self._need_compact(full_messages):
            full_messages = extended_system_msg + await self.compact_history()
        # 实际发送给 LLM 的消息列表：完整的系统提示 + 压缩的对话历史
        return full_messages

    def get_metadata_prompt(self) -> str:
        """生成注入到系统提示的技能元数据列表"""
        if not self.metadata_cache:
            return ""
        lines = ["\n[Available Skills] (activate via `active_skill` tool if needed)"]
        for name, description in self.metadata_cache.items():
            lines.append(f"- {name}: {description}")
        return "\n".join(lines)

    def get_active_context_extension(self) -> str:
        """返回当前活跃技能指令和已加载资源的文本，用于附加到上下文"""
        parts = []
        for skill_name, instruction in self.active_instructions.items():
            parts.append(f"\n[Active Skill] (name: {skill_name})\n{instruction}")
        for res_path, content in self.loaded_resources.items():
            parts.append(f"\n[Loaded Resource] (path: {res_path})\n{content}")
        return "\n".join(parts) if parts else ""

    def extend_system_prompt(self, base_prompt: str) -> List[dict]:
        """根据 Skills 的激活状态，在系统消息后动态追加对应的元数据、指令以及资源"""
        extended_content = base_prompt + self.get_metadata_prompt()
        active_ext = self.get_active_context_extension()
        if active_ext:
            extended_content += "\n" + active_ext
        ws_tree = get_workspace_tree(os.getcwd(), excludes=["mcp_service", "skills", "chroma_db", "session"])
        extended_content += f"\n[Workspace Tree] (depth=3):\n{ws_tree}"
        system_message = [{"role": "system", "content": extended_content}]
        return system_message

    def activate_skill(self, skill_name: str) -> str:
        """第二层：注入完整技能指令"""
        if skill_name not in self.metadata_cache:
            return f"Error: Skill '{skill_name}' not found"
        if skill_name in self.active_instructions:
            return f"Skill '{skill_name}' already active"
        skill_path = self.metadata_cache[skill_name]['path']
        skill_md = skill_path / "SKILL.md"
        content = skill_md.read_text(encoding="utf-8")
        # 移除 YAML 头，保留 Markdown 正文
        body = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL).strip()
        self.active_instructions[skill_name] = body
        return f"Skill '{skill_name}' activated. Instructions injected into context."

    def load_resource(self, skill_name: str, resource_path: str) -> str:
        """第三层：按需加载资源文件（references 或 scripts）"""
        if skill_name not in self.metadata_cache:
            return f"Error: Skill '{skill_name}' not found"
        full_path = self.metadata_cache[skill_name]['path'] / resource_path
        if not full_path.exists():
            return f"Error: Resource '{resource_path}' not found"
        key = f"{skill_name}/{resource_path}"
        if key in self.loaded_resources:
            return f"Resource already loaded. Preview: {self.loaded_resources[key][:200]}..."
        content = full_path.read_text(encoding="utf-8")
        self.loaded_resources[key] = content
        return f"Resource loaded. Content preview:\n{content[:200]}"

    async def summarize_with_llm(self, history_msg: list[dict]) -> str:
        """调用 LLM 接口生成历史对话的摘要"""
        messages = [
            {"role": "system", "content": COMPRESSION_PROMPT},
            {"role": "user", "content": get_messages_text(history_msg)},
        ]
        async with self._semaphore:
            response = await self.llm_client.chat.completions.create(
            model=os.environ.get("MODEL"),
            messages=messages,
            temperature=0.3
        )
        if isinstance(response, str) and response.startswith("data:"):
            # 转换SSE格式的接口响应
            parsed = json.loads(response[5:])
            if "error" in parsed:
                raise Exception(f"API返回错误: {parsed.get('error')}")
            response = ChatCompletion.model_validate(parsed)
        summary = response.choices[0].message.content.strip()
        print("[Summary]", summary.replace('\n', ' ').replace('\r', ' ')[:200])
        return summary

    @staticmethod
    def _count_messages_tokens(messages: List[dict]) -> int:
        """近似计算消息列表的总 token"""
        prompt_tokens = 0
        try:
            encoder = tiktoken.get_encoding(os.environ.get("MODEL"))
        except ValueError:
            # 更灵活的方案： encoder = AutoTokenizer.from_pretrained(os.environ.get("MODEL"))
            encoder = tiktoken.get_encoding("cl100k_base")

        for msg in messages:
            # 序列化为紧凑 JSON
            json_str = json.dumps(msg, ensure_ascii=False, separators=(',', ':'))
            prompt_tokens += len(encoder.encode(json_str, disallowed_special=()))

        return prompt_tokens

    def _need_compact(self, messages: List[dict]) -> bool:
        """判断消息列表是否需要压缩"""
        if not self.compress_config:
            return False
        if len(messages) <= self.compress_config.get("keep_recent"):
            return False  # 消息数量不足 keep_recent，无需压缩
        mode = self.compress_config.get("mode")
        threshold = self.compress_config.get("threshold")
        if mode == "message_count":
            return len(messages) >= threshold
        elif mode == "token_ratio":
            max_token = self.compress_config.get("max_context_length", 0)
            reserved_tokens = self.compress_config.get("reserved_tokens", 0)
            if max_token == 0:
                return False
            message_tokens = self._count_messages_tokens(messages)
            token_ratio = (reserved_tokens + message_tokens) / max_token
            return token_ratio >= threshold
        else:
            return False

    @staticmethod
    def _split_into_rounds(messages: List[dict]) -> List[List[dict]]:
        """将消息列表按 user 角色分割成多个对话轮次。每个轮次以一个 user 消息开始，直到下一个 user 消息结束"""
        rounds = []
        current_round = []
        for msg in messages:
            if msg.get("role") == "user":
                if current_round:  # 上一个轮次结束
                    rounds.append(current_round)
                current_round = [msg]
            else:
                current_round.append(msg)
        if current_round:
            rounds.append(current_round)
        return rounds

    @staticmethod
    def _find_safe_split(messages: List[dict], keep_recent: int) -> int:
        """找到压缩消息列表的安全切分点，避免要保留的部分出现“孤儿”工具消息"""
        split_idx = max(0, len(messages) - keep_recent)

        # 切分点不在 tool 上，直接返回
        if split_idx >= len(messages) or messages[split_idx].get("role") != "tool":
            return split_idx

        # 场景 1：尝试回退到最近的 assistant(tool_calls) 之前，整组保留
        max_lookback = keep_recent
        for offset in range(1, max_lookback + 1):
            i = split_idx - offset
            if i < 0:
                break
            msg = messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                return i

        # 场景 2：回溯 max 步还是 tool 消息（超大工具组）那就向前移，让整个工具组进入 to_compress
        while split_idx < len(messages) and messages[split_idx].get("role") == "tool":
            split_idx += 1
        return split_idx

    async def compact_history(self, max_workers: int = 5) -> List[dict]:
        """
        压缩历史消息，具体策略：
        - 根据配置的 keep_recent，保留最近 N 条消息，将剩余待压缩部分按 user 消息切分为多个轮次
        - 每个轮次的所有消息压缩为一条摘要，同时保留 user 消息原文
        - 压缩后的轮次按原序排列，再拼接上尾部 recent 消息，替换内部维护的对话历史列表
        """
        # 分割待压缩部分 + 保留部分
        keep_recent = self.compress_config.get("keep_recent", 10)
        split_idx = self._find_safe_split(self.history_messages, keep_recent)
        to_compress = self.history_messages[:split_idx]
        recent = self.history_messages[split_idx:]

        print(f"⏳ 正在执行上下文压缩 (compress_msg={len(to_compress)}, keep_msg={len(recent)})")

        # 分割待压缩的对话轮次
        rounds = self._split_into_rounds(to_compress)
        tasks = [self.summarize_with_llm(round_msgs) for round_msgs in rounds]
        thread_results = await asyncio.gather(*tasks)

        # 按原始轮次顺序构建压缩后的消息列表
        compressed_rounds = []
        for idx, summary_content in enumerate(thread_results):
            user_msg = rounds[idx][0]
            summary_msg = {"role": "assistant", "content": summary_content}
            compressed_rounds.extend([user_msg, summary_msg])
        self.history_messages = compressed_rounds + recent

        return self.history_messages

    def reset_session(self):
        """新会话时重置历史对话和激活的技能栈"""
        self.history_messages.clear()
        self.active_instructions.clear()
        self.loaded_resources.clear()