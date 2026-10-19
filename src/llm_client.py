import json
import os
import asyncio
import threading
from typing import List, Union, Optional, Literal
from urllib.parse import urlparse

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel
from openai.types.chat import ChatCompletion

from src.terminal_ui import display_stream, end_stream, display


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-flash"

# 允许出现在请求 messages 中的字段；usage / finish_reason 等本地附加信息不会发往 API
_MESSAGE_KEYS = {"role", "content", "name", "tool_calls", "tool_call_id"}
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


class OpenAIClient:
    """
    兼容 OpenAI 接口的 LLM 客户端，默认适配 DeepSeek API，使用单例模式确保全局只有一个实例。

    配置优先级：显式参数 > 环境变量 > config.json > 内置默认值。
    - API Key：DEEPSEEK_API_KEY（兼容 API_KEY）/ llm.api_key
    - 服务地址：DEEPSEEK_BASE_URL（兼容 BASE_URL）/ llm.base_url，默认 https://api.deepseek.com
    - 模型：DEEPSEEK_MODEL（兼容 MODEL）/ llm.model，默认 deepseek-flash

    DeepSeek 相关内容：
    - provider 为 deepseek 时通过 extra_body 下发 thinking 开关，并可指定 reasoning_effort；
    - thinking 模式在流式/非流式响应中返回 reasoning_content，会一并保存；
    - thinking 模式下请求历史中的每条 assistant 消息都必须回传 reasoning_content，
      没有可用推理时用空串占位，否则 API 返回 400。
    """

    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        # 线程安全的单例，防止两个线程同时创建对象
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]

    def __init__(self, config: dict = None, model: str = None, api_key: str = None, base_url: str = None, max_concurrency: int = 3):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        config = config or {}

        self.model = (
            model
            or os.getenv("DEEPSEEK_MODEL")
            or os.getenv("MODEL")
            or config.get("model")
            or DEEPSEEK_DEFAULT_MODEL
        )
        api_key = (
            api_key
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("API_KEY")
            or config.get("api_key")
        )
        base_url = (
            base_url
            or os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("BASE_URL")
            or config.get("base_url")
            or DEEPSEEK_BASE_URL
        )

        if not api_key:
            raise ValueError(
                "缺少 DeepSeek API Key：请设置环境变量 DEEPSEEK_API_KEY，"
                "或在 config.json 的 llm.api_key 中填写。"
            )

        self.base_url = base_url.rstrip("/")
        self.provider = (
            config.get("provider")
            or ("deepseek" if "deepseek" in self.base_url.lower() else "openai")
        ).lower()

        # DeepSeek 默认开启 thinking；其他 OpenAI 兼容服务默认不发送该参数
        if "thinking" in config:
            self.thinking = bool(config["thinking"])
        else:
            self.thinking = True if self.provider == "deepseek" else None
        self.reasoning_effort = config.get("reasoning_effort")
        self.max_tokens = config.get("max_tokens")
        self.timeout = int(config.get("timeout") or os.getenv("TIMEOUT") or 600)
        self.max_retries = int(config.get("max_retries", 2))
        self.proxy = config.get("proxy")

        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client=self._build_http_client(),
        )

    def _build_http_client(self) -> Optional[httpx.AsyncClient]:
        """构建 httpx 客户端：本地服务绕过系统代理，公网服务（如 DeepSeek）默认继承代理环境变量。"""
        if self.proxy:
            return httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(proxy=self.proxy))
        if self._is_local_endpoint():
            return httpx.AsyncClient(
                trust_env=False,
                transport=httpx.AsyncHTTPTransport(proxy=None),
            )
        return None

    def _is_local_endpoint(self) -> bool:
        host = (urlparse(self.base_url).hostname or "").lower()
        return host in _LOCAL_HOSTS or host.endswith(".local")

    @staticmethod
    def _to_dict(message):
        """把 Pydantic 消息模型转成 dict，普通 dict 原样返回。"""
        if isinstance(message, BaseModel):
            return message.model_dump(exclude_none=True)
        return message

    @classmethod
    def _normalize_tool_call(cls, tool_call) -> dict:
        """把流式分片或 SDK 模型中的 tool_call 统一成请求/历史消息格式。"""
        tool_call = cls._to_dict(tool_call) or {}
        function = tool_call.get("function") or {}
        normalized = {
            "type": tool_call.get("type") or "function",
            "function": {
                "name": function.get("name") or "",
                "arguments": function.get("arguments") or "",
            },
        }
        if tool_call.get("id"):
            normalized["id"] = tool_call["id"]
        return normalized

    def _normalize_messages(self, messages: List[Union[dict, BaseModel]], inject_reasoning: bool) -> List[dict]:
        """清理历史消息中仅用于本地记录的多余字段，并按 DeepSeek 要求回传 reasoning_content。"""
        normalized = []
        for raw in messages:
            message = self._to_dict(raw)
            if not isinstance(message, dict):
                normalized.append(message)
                continue

            msg = {key: value for key, value in message.items() if key in _MESSAGE_KEYS}
            if msg.get("role") == "assistant":
                if msg.get("tool_calls"):
                    msg["tool_calls"] = [self._normalize_tool_call(tc) for tc in msg["tool_calls"]]
                # DeepSeek 思考模式下，请求历史中的每条 assistant 消息都必须回传 reasoning_content，
                # 包括没有发起工具调用的轮次；缺失会导致 400。本地压缩生成的摘要消息补空串。
                if inject_reasoning:
                    msg["reasoning_content"] = message.get("reasoning_content") or ""
            normalized.append(msg)
        return normalized

    @staticmethod
    def _delta_field(delta, field: str):
        """读取 SDK 未显式声明的扩展字段（如 reasoning_content）。"""
        value = getattr(delta, field, None)
        if value is not None:
            return value
        extra = getattr(delta, "model_extra", None) or {}
        return extra.get(field)

    @staticmethod
    def _merge_tool_call_delta(slots: dict, tool_call) -> None:
        """按 index 合并流式 tool_call 分片：id/name 通常只在首片出现，arguments 逐片拼接。"""
        index = getattr(tool_call, "index", None)
        if index is None:
            index = 0
        slot = slots.setdefault(
            index,
            {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if getattr(tool_call, "id", None):
            slot["id"] = tool_call.id
        if getattr(tool_call, "type", None):
            slot["type"] = tool_call.type
        function = getattr(tool_call, "function", None)
        if function is not None:
            name = getattr(function, "name", None)
            if name:
                slot["function"]["name"] += name
            arguments = getattr(function, "arguments", None)
            if arguments:
                slot["function"]["arguments"] += arguments

    @staticmethod
    def _format_usage(usage) -> dict:
        if usage is None:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "cached_tokens": 0,
            }
        details = getattr(usage, "completion_tokens_details", None)
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
            "cached_tokens": getattr(usage, "prompt_cache_hit_tokens", 0) or 0,
        }

    @staticmethod
    def _parse_sse_response(response: str) -> ChatCompletion:
        """将 SSE 格式的接口响应转换为 OpenAI 格式的 ChatCompletion 对象。"""
        parsed = json.loads(response[5:])
        if "error" in parsed:
            raise Exception(f"API返回错误: {parsed.get('error')}")
        return ChatCompletion.model_validate(parsed)

    async def _consume_stream(self, response, result: dict, stream_mode: str):
        """消费流式响应：打印 reasoning/content，并按 index 聚合 tool_call 分片。"""
        collected_content = []
        collected_reasoning = []
        tool_call_slots = {}
        usage = None
        finish_reason = None

        try:
            async for chunk in response:
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage

                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason

                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                reasoning = self._delta_field(delta, "reasoning_content")
                if reasoning:
                    display_stream(reasoning, prefix="[reasoning]")
                    collected_reasoning.append(reasoning)

                content = getattr(delta, "content", None)
                if content:
                    if stream_mode == "full":
                        display_stream(content)
                    collected_content.append(content)

                for tool_call in getattr(delta, "tool_calls", None) or []:
                    self._merge_tool_call_delta(tool_call_slots, tool_call)
        finally:
            # 即使流中途异常，也把已收到的内容写回结果，避免整轮响应丢失
            result["content"] = "".join(collected_content)
            result["reasoning_content"] = "".join(collected_reasoning)
            result["tool_calls"] = [slot for _, slot in sorted(tool_call_slots.items())] or None
            result["finish_reason"] = finish_reason

        return usage

    async def invoke(
        self,
        messages: List[Union[dict, BaseModel]],
        tools: Optional[List[dict]] = None,
        temperature: float = 0,
        stream_mode: Literal[None, "full", "reasoning_only"] = None,
        thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """
        调用大语言模型推理接口，支持流式响应

        stream_mode:
        - None: 非流式，返回完整结果，不打印任何内容
        - "full": 流式打印 reasoning_content 和 content
        - "reasoning_only": 流式，只打印 reasoning_content

        thinking:
        - None: 使用 config.json 中 llm.thinking 的配置（DeepSeek 默认开启）
        - True/False: 为本次调用显式开启/关闭思考模式

        reasoning_effort:
        - None: 使用 config.json 中 llm.reasoning_effort
        - "low" / "high" / "max": 本次调用的思考深度
        """
        result = {"role": "assistant"}

        if stream_mode not in ("full", "reasoning_only"):
            stream_mode = None
        is_streaming = stream_mode is not None
        tools = tools or None

        thinking_enabled = None
        if self.provider == "deepseek":
            thinking_enabled = self.thinking if thinking is None else thinking

        request = {
            "model": self.model,
            "messages": self._normalize_messages(
                messages,
                inject_reasoning=self.provider == "deepseek" and bool(thinking_enabled),
            ),
            "temperature": temperature,
        }
        if tools:
            request["tools"] = tools

        if self.provider == "deepseek":
            if thinking_enabled is not None:
                request["extra_body"] = {
                    "thinking": {"type": "enabled" if thinking_enabled else "disabled"}
                }
            effort = reasoning_effort if reasoning_effort is not None else self.reasoning_effort
            if thinking_enabled and effort:
                request["reasoning_effort"] = effort

        limit = max_tokens if max_tokens is not None else self.max_tokens
        if limit:
            request["max_tokens"] = int(limit)

        if is_streaming:
            request["stream"] = True
            request["stream_options"] = {"include_usage": True}

        request.update(kwargs)

        async with self._semaphore:
            try:
                response = await self.client.chat.completions.create(**request)
                if isinstance(response, str) and response.startswith("data:"):
                    response = self._parse_sse_response(response)
            except Exception as e:
                display(f"❌ 调用LLM API时发生错误: {e}")
                result["content"] = str(e)
                result["error"] = True
                return result

            if is_streaming:
                try:
                    usage = await self._consume_stream(response, result, stream_mode)
                except Exception as e:
                    display(f"❌ 读取LLM流式响应时发生错误: {e}")
                    result["error"] = True
                    usage = None
                finally:
                    end_stream()
            else:
                try:
                    choice = response.choices[0]
                    message = choice.message
                    result["content"] = message.content or ""
                    result["reasoning_content"] = self._delta_field(message, "reasoning_content") or ""
                    result["tool_calls"] = [
                        self._normalize_tool_call(tc) for tc in (message.tool_calls or [])
                    ] or None
                    result["finish_reason"] = choice.finish_reason
                    usage = getattr(response, "usage", None)
                except Exception as e:
                    display(f"❌ 解析LLM响应时发生错误: {e}")
                    result["content"] = str(e)
                    result["error"] = True
                    usage = None

        result["usage"] = self._format_usage(usage)
        return result
