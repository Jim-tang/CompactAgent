import json
import os
import re
import yaml
import asyncio
from typing import Dict, List, Optional
from datetime import datetime

from src.llm_client import OpenAIClient
from src.terminal_ui import display
import src.common_utils as util


class AgentContextManager:
    """
    Agent 上下文管理器

    管理 Agent 对话历史和 Skill 的渐进式披露
    支持根据 token 阈值或消息数量自动压缩历史，避免超出上下文窗口限制
    """
    def __init__(self, llm_client: OpenAIClient, compress_config: Optional[Dict] = None):
        self.llm_client = llm_client                       # 用于生成摘要的 OpenAIClient 客户端
        self.compress_config = compress_config             # 上下文压缩配置字典
        self.metadata_cache = {}                           # 存储所有技能元数据的字典
        self.active_instructions = {}                      # 存储当前已激活技能的完整指令的字典
        self.loaded_resources = {}                         # 存储已加载的技能资源文件的字典
        self.system_message = []                           # 系统消息（静态层+动态层）
        self.history_messages = []                         # 历史对话（不含 system）放到管理器内部维护
        self._compaction_task = None                       # 后台压缩任务
        self.session_dir = os.path.join(util.PROJECT_ROOT, "sessions")
        self._load_system_prompt()
        self._load_all_metadata()

    def _load_system_prompt(self):
        """加载系统提示的静态层"""
        prompt_template = util.PROJECT_ROOT / "template" / "system_prompt.md"
        with open(prompt_template, 'r', encoding='utf-8') as f:
            content = f.read()
        runtime_info = util.collect_runtime()
        # 将运行时信息渲染至系统提示的模板
        self.system_prompt = util.render_prompt(content, runtime_info)

    def _load_all_metadata(self):
        """
        第一层：启动时加载所有技能的元数据（仅 name + description）

        遍历 skills 下的每个子目录，读取 SKILL.md 文件并解析 YAML 头，提取 name 和 description
        """
        skills_root = util.PROJECT_ROOT / "skills"
        if not skills_root.exists():
            return
        for skill_dir in skills_root.iterdir():
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

    async def prepare_message(self):
        """上下文处理的核心方法：动态扩展 -> 压缩，然后拼接历史记录返回完整消息列表"""
        # 渐进式披露
        self.system_message = self.extend_system_prompt()
        # 消费后台压缩结果
        self._apply_pending_compaction()
        full_messages = self.system_message + self.history_messages
        strategy = self._compact_strategy(full_messages)

        if strategy == "blocking":
            display("[Agent] 上下文超出硬限制，等待后台压缩完成...")
            await self._compact_blocking()

        if strategy == "background":
            if not self._is_compacting():
                # 启动后台压缩，本轮立即返回
                self._start_background_compaction()

        return self.system_message + self.history_messages

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

    def extend_system_prompt(self) -> List[dict]:
        """根据 Skills 的激活状态，在系统消息后动态追加对应的元数据、指令以及资源"""
        extended_content = self.system_prompt + self.get_metadata_prompt()
        active_ext = self.get_active_context_extension()
        if active_ext:
            extended_content += "\n" + active_ext
        ws_tree = util.get_workspace_tree(os.getcwd(), excludes=["mcp_service", "skills", "chroma_db"])
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

    async def summarize_with_llm(self, history_msg: list[dict], instruction_type: str) -> str:
        """调用 LLM 接口生成历史对话的摘要"""
        template_file = util.PROJECT_ROOT / "template" / f"{instruction_type}.md"
        with open(template_file, 'r', encoding='utf-8') as f:
            instruction = f.read()
        messages = [
            {"role": "system", "content": self.system_prompt},
            *history_msg,
            {"role": "user", "content": instruction}
        ]
        response = await self.llm_client.invoke(messages, temperature=0.3)
        summary = response.get("content", "").strip()
        return summary

    def _compact_strategy(self, messages: List[dict]) -> str:
        """判断消息列表是否需要压缩以及对应的压缩策略: skip | background | blocking"""
        if not self.compress_config:
            return "skip"
        if len(messages) <= self.compress_config.get("keep_recent"):
            return "skip"  # 消息数量不足 keep_recent，无需压缩

        mode = self.compress_config.get("mode")
        threshold = self.compress_config.get("threshold")
        if mode == "message_count":
            return "background"  if len(messages) >= threshold else "skip"

        elif mode == "token_ratio":
            max_token = self.compress_config.get("max_context_length", 128 * 1024)
            reserved_tokens = self.compress_config.get("reserved_tokens", 8 * 1024)
            blocking_threshold = self.compress_config.get("blocking_threshold", 0.9)
            message_tokens = util.estimate_token_count(messages)
            token_ratio = (reserved_tokens + message_tokens) / max_token
            if token_ratio >= blocking_threshold:
                return "blocking"
            elif token_ratio >= threshold:
                return "background"
            else:
                return "skip"

        else:
            return "skip"

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

    def _is_compacting(self) -> bool:
        return self._compaction_task is not None and not self._compaction_task.done()

    def _form_compacted_history(self, summary: str, split_idx: int) -> List[dict]:
        summary_msg = {
            "role": "user",
            "content": f"The following is a summary of the earlier conversation, before the compact boundary. Use it as background context.\n\n{summary}"
        }
        boundary = {"role": "system", "content": "[CompactBoundary]"}

        # 用切片保留压缩期间新追加的尾部消息
        recent = self.history_messages[split_idx:]
        for msg in recent:
            # 压缩改变了前缀，旧的 token 锚点已失效
            msg.pop("usage", None)

        compacted_history = [summary_msg, boundary] + recent
        return compacted_history

    def _apply_pending_compaction(self) -> bool:
        """消费已完成的压缩结果，同步执行避免竞态"""
        if self._compaction_task is None or self._is_compacting():
            return False

        try:
            split_idx, summary = self._compaction_task.result()
        except (Exception, asyncio.CancelledError) as e:
            display(f"[Agent] Error: 上下文压缩失败: {e}")
            return False
        finally:
            self._compaction_task = None

        # history 被外部重置，丢弃本次结果
        if split_idx > len(self.history_messages):
            return False

        system_prompt_tokens = util.estimate_token_count(self.system_message)
        before_tokens = system_prompt_tokens + util.estimate_token_count(self.history_messages)
        self.history_messages = self._form_compacted_history(summary, split_idx)
        after_tokens = system_prompt_tokens + util.estimate_token_count(self.history_messages)
        display(f"[Agent] 上下文压缩完成: {before_tokens} Tokens -> {after_tokens} Tokens")
        return True

    async def _run_compaction(self, snapshot: List[dict]):
        summary = await self.summarize_with_llm(snapshot, "context_compression")
        return len(snapshot), summary

    def _start_background_compaction(self) -> bool:
        """通过创建一个独立的协程在后台并发执行压缩任务（不阻塞网络IO）"""
        keep_recent = self.compress_config.get("keep_recent", 10)
        split_idx = self._find_safe_split(self.history_messages, keep_recent)
        if split_idx <= 0:
            return False

        # 快照压缩区间，避免后台任务期间被外部修改
        snapshot = [dict(m) for m in self.history_messages[:split_idx]]
        recent = self.history_messages[split_idx:]

        display(f"[Agent] 启动后台上下文压缩 (compress_msg={len(snapshot)}, keep_msg={len(recent)})")
        self._compaction_task = asyncio.create_task(self._run_compaction(snapshot))
        return True

    async def _compact_blocking(self):
        if not self._is_compacting():
            if not self._start_background_compaction():
                display("[Agent] Warning: 无可压缩历史")
                return
        try:
            await self._compaction_task
        except asyncio.CancelledError:
            self._apply_pending_compaction()
            raise
        except Exception as e:
            display(f"[Agent] Error：上下文压缩失败 {e}")

        self._apply_pending_compaction()

    async def manual_compact(self) -> List[dict]:
        await self._compact_blocking()
        return self.history_messages

    def reset_session(self):
        """新会话时重置历史对话和激活的技能栈"""
        self.history_messages.clear()
        self.active_instructions.clear()
        self.loaded_resources.clear()

    async def archive_session(self):
        """提炼并存储本次会话的对话历史并归档至 Sessions.md"""
        if not self.history_messages:
            display("[Agent] Warning: Empty session history, skip recording")
            return
        os.makedirs(self.session_dir, exist_ok=True)
        try:
            raw = await self.summarize_with_llm(self.history_messages, "archive_session")
            session_note = util.parse_session_summary(raw)
            topic = session_note.get("topic")
            summary = session_note.get("content")
            keywords = session_note.get("keywords")
            timestamp = datetime.now().isoformat()
            self.update_session_md(topic, summary, timestamp, keywords)
            archive_path = self.save_session_json(topic, timestamp)
            display(f"[Agent] 已归档会话：{archive_path}")
        except Exception as e:
            display(f"[Agent] Error: 会话归档失败: {e}")

    def update_session_md(self, topic: str, content: str, timestamp: str, keywords:list[str]) -> None:
        """将新的会话条目增量追加到 Sessions.md（插入到文件开头）"""
        entry_lines = [
            f"### {topic}\n",
            f"> **{datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S')}**\n",
            f"> **Key Word: {' | '.join(keywords)}**\n"
            f"> {content}\n",
            "---\n",
        ]
        entry_text = "".join(entry_lines)
        session_file_path = os.path.join(self.session_dir, "Sessions.md")
        if os.path.exists(session_file_path):
            with open(session_file_path, "r", encoding="utf-8") as f:
                existing = f.read()
            new_content = entry_text + existing
        else:
            new_content = entry_text

        with open(session_file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

    def save_session_json(self, topic: str, timestamp: str) -> str:
        """将完整对话历史保存为 JSON 文件"""
        safe_topic = re.sub(r'[<>:"/\\|?*]', '_', topic)
        safe_timestamp = datetime.fromisoformat(timestamp).strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.session_dir, f"{safe_timestamp}_{safe_topic}.json")
        session_data = {
            "topic": topic,
            "messages": self.system_message + self.history_messages,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        return filepath