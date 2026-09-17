import re
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict


# 声明式安全配置（Openclaw 风格）
SAFETY_CONFIG = {
    "input_blacklist": ["忽略之前的所有指令", "ignore previous instructions"],
    "output_blacklist": [
        r"sk-[-A-Za-z0-9]+",
        r"AKIA[0-9A-Z]{16}",
        r"-----BEGIN RSA PRIVATE KEY-----"
    ],
    "sensitive_strategy": "redact",
    "allowed_tools": [],
    "tool_config": {
        "write_file": {"sandbox_root": "D:\\NoteBooks\\Agent", "require_approval": True},
    },
    "truncation_config": {
        "max_output_chars": 20480,          # 最大输出字符数
        "preserve_head_ratio": 0.7,         # 头部占截断比例
        "persist_threshold_chars": 102400   # 持久化输出阈值
    },
    "runtime_approval": True
}


class HookReject:
    """安全策略否决，可被序列化为 JSON 传递给 LLM"""
    def __init__(self, reason: str):
        self.reason = reason

    def to_json(self) -> str:
        return json.dumps({
            "vetoed": True,
            "reason": self.reason
        }, ensure_ascii=False)

    def __str__(self):
        return f"[HookReject] {self.reason}"

class AgentHookManager:
    """
    微内核 Hook 管理器，一个简化的 Hook 基础设施。
    """
    def __init__(self, config: Dict = SAFETY_CONFIG):
        """注册基础钩子"""
        self._hooks = [
            InputSensitiveHook(config),
            ToolWhitelistHook(config),
            CommandSafetyHook(config),
            OutputSensitiveHook(config),
            OutputTruncationHook(config),
        ]
        self.state = {"tool_call_count": 0}  # 共享状态

    def add(self, hook):
        """允许添加自定义钩子"""
        self._hooks.append(hook)

    def emit(self, event: str, *args):
        """触发 Agent 生命周期事件（支持数据流的管道传递与链式加工），返回事件的核心数据"""
        for hook in self._hooks:
            # 依次获取所有 Hook 实例对应该事件的方法，比如 InputSensitiveHook.on_agent_start
            func = getattr(hook, event, None)
            if func is None:
                continue
            # 传递 state 来实现状态共享和钩子间通信
            result = func(self.state, *args)
            if isinstance(result, HookReject):
                return result
            # 如果没有被钩子拦截，则用返回值来替换第一个参数，实现核心数据的链式加工
            if result is not None:
                args = (result,) + args[1:] if len(args) > 1 else (result,)
        return args[0]


class IOSensitiveBase:
    """I/O 清洗与拦截钩子基类"""
    def __init__(self, blacklists: List[str]) -> None:
        self.patterns = [re.compile(p, re.IGNORECASE) for p in blacklists]

    def _detect(self, text: str):
        hits = []
        for p in self.patterns:
            if p.search(text):
                hits.append(p.pattern)
        return hits

    def _redact(self, text: str):
        for pattern in self.patterns:
            text = pattern.sub('[REDACT]', text)
        return text

class InputSensitiveHook(IOSensitiveBase):
    """
    输入清洗与拦截钩子
    策略：
    - detect_only: 仅记录，不修改（调试用）
    - redact（默认）：使用 [REDACT] 替换黑名单
    - veto：抛出 HookRejectException 直接拒绝
    """
    def __init__(self, hook_config: Dict):
        super().__init__(hook_config.get("input_blacklist", []))
        self.mode = hook_config.get("sensitive_strategy", "redact")

    def on_agent_start(self, state: dict, user_input: str):
        hits = self._detect(user_input)
        if not hits:
            return user_input
        if self.mode == "detect_only":
            return user_input
        elif self.mode == "redact":
            return self._redact(user_input)
        else:
            return HookReject(f"输入包含禁止指令: {', '.join(hits)}")

class OutputSensitiveHook(IOSensitiveBase):
    """输出护栏：正则匹配敏感信息"""
    def __init__(self, hook_config: Dict):
        super().__init__(hook_config.get("output_blacklist", []))
        self.mode = hook_config.get("sensitive_strategy", "redact")

    def on_tool_end(self, state: dict, tool_output: str, tool_name: str):
        hits = self._detect(tool_output)
        if not hits:
            return tool_output
        if self.mode == "redact":
            return f"工具输出包含敏感信息，过滤后：{self._redact(tool_output)}"
        else:
            return HookReject(f"工具输出包含敏感信息: {', '.join(hits)}")

    def on_agent_end(self, state: dict, final_output: str):
        hits = self._detect(final_output)
        if not hits:
            return final_output
        if self.mode == "redact":
            return self._redact(final_output)
        else:
            return HookReject(f"最终输出包含敏感信息: {', '.join(hits)}")

class ToolWhitelistHook:
    """工具护栏：白名单、路径沙箱，风险审批"""
    def __init__(self, hook_config):
        self.allowed = hook_config.get("allowed_tools", [])
        self.tool_config = hook_config.get("tool_config")
        self.runtime_approval = hook_config.get("runtime_approval", False)

    def on_tool_start(self, state: dict, tool_args: dict, tool_name: str):
        state["tool_call_count"] += 1

        if self.allowed and tool_name not in self.allowed:
            return HookReject(f"工具 {tool_name} 不在白名单中")
        cfg = self.tool_config.get(tool_name, {})

        # 路径沙箱检查
        if "path" in tool_args and cfg.get("sandbox_root"):
            if not str(Path(tool_args["path"]).resolve()).startswith(cfg.get("sandbox_root")):
                return HookReject("路径越界")

        # 高风险审批
        if cfg.get("require_approval") and self.runtime_approval:
            truncated_args = ', '.join([f"{k}={str(v).replace('\n', ' ')[:200]}" for k,v in tool_args.items()])
            print(f"⚠️ 高风险操作需审批：{tool_name}({truncated_args})")
            if input("输入 'yes(y)' 批准，其他键拒绝: ").strip().lower() not in  ('yes', 'y'):
                return HookReject("人工审批未通过")

class CommandSafetyHook:
    """危险命令拦截钩子"""

    DANGEROUS_PATTERNS = [
        # rm 危险标志
        r'\brm\s+(?:.*?(?:-rf|--recursive\s+--force)|-rf\s+/(?:etc|boot|lib|sys|proc|dev)\b|.*?--no-preserve-root)',
        # 格式化 / 清空磁盘
        r'\bmkfs\.\S+',
        r'\bdd\s+if=/dev/zero\s+of=/dev/',
        # 高危 chmod
        r'\bchmod\s+(-R\s+)?777\s+/',
        # 高危 wget/curl 管道执行
        r'(wget|curl)\s+.*\s*\|\s*(sh|bash|zsh)',
        # CMD 高危命令
        r'\b(?:del|erase|rmdir|rd|sudo|su)\b',
    ]

    def __init__(self, hook_config: Dict):
        self.command_tools = hook_config.get("command_tools", [])
        self.command_blacklist = hook_config.get("command_blacklist", [])

    def on_tool_start(self, state: dict, tool_args: dict, tool_name: str):
        if not self.command_tools:
            self.command_tools = ["run_command", "shell", "exec", "bash"]
        if tool_name not in self.command_tools or not tool_args:
            return  # 非命令类工具，忽略

        command_blacklist = self.command_blacklist + CommandSafetyHook.DANGEROUS_PATTERNS
        patterns = [re.compile(p, re.IGNORECASE) for p in command_blacklist]
        command = tool_args.get("command") or list(tool_args.values())[0]
        for p in patterns:
            if p.search(command):
                return HookReject(f"危险命令拦截: {command}")

class OutputTruncationHook:
    """工具输出截断钩子，超大输出持久化为文件并返回引用"""
    def __init__(self, hook_config: Dict):
        config = hook_config.get("truncation_config", {})
        self.max_output_chars = config.get("max_output_chars", 8000)
        self.head_ratio = config.get("head_ratio", 0.7)
        self.persist_threshold_chars = config.get("persist_threshold_chars", self.max_output_chars * 10)

    def on_tool_end(self, state: dict, tool_output: str, tool_name: str) -> str:
        """工具执行后，检查并截断输出"""
        output_len = len(tool_output)
        if output_len <= self.max_output_chars:
            return tool_output

        truncated = self._truncate_output(tool_output)
        if output_len >= self.persist_threshold_chars:
            ref_msg = self._persist_large_output(tool_name, tool_output)
            truncated += ref_msg

        return truncated

    def _truncate_output(self, tool_output: str) -> str:
        """超过 max_chars 则保留 head 和 tail，中间替换为截断说明"""
        output_len = len(tool_output)
        # 计算头尾长度
        head_len = int(self.max_output_chars * self.head_ratio)
        tail_len = self.max_output_chars - head_len

        head = tool_output[:head_len]
        tail = tool_output[-tail_len:]
        # 输出截断后插入可读提示 (OpenCode风格)
        truncated = (
            f"[Truncated|original_size={output_len} chars, limited_to={self.max_output_chars} chars]\n"
            f"{head}\n\n[... {output_len - self.max_output_chars} characters omitted ...]\n\n{tail}"
        )
        return truncated

    def _persist_large_output(self, tool_name: str, tool_output: str) -> str:
        """超大输出持久化为文件并返回引用"""
        filename = f"{tool_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_path = Path.cwd() / filename

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(tool_output)

        ref_msg = (
            f"\n[LargeOutputStored] Tool '{tool_name}' output size: {len(tool_output)} chars\n"
            f"Output persisted to: {file_path}\n"
            f"You can inspect it using: read_file('{file_path}', offset=0, limit=2000)"
        )

        return ref_msg
