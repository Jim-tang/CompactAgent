import os
import json
import asyncio
import threading
import time
import httpx
from openai import AsyncOpenAI
from typing import List, Optional, Literal

from src.terminal_ui import display_stream, end_stream, launch_ui, display


class OpenAIClient:
    """
    兼容 OpenAI 接口的 LLM 客户端，使用单例模式确保全局只有一个实例。
    """
    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        # 线程安全的单例，防止两个线程同时创建对象
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]

    def __init__(self, config: dict = None, model: str = None, api_key: str = None, base_url: str = None,
                 max_concurrency: int = 3):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        if config is None:
            config = {}

        self.model = model if model is not None else config.get("model") or os.getenv("MODEL")
        api_key = api_key if api_key is not None else config.get("api_key") or os.getenv("API_KEY")
        base_url = base_url if base_url is not None else config.get("base_url") or os.getenv("BASE_URL")

        if not all([self.model, api_key, base_url]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供。")

        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=int(os.getenv("TIMEOUT", 120)),
            http_client=httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(proxy=None)),
        )

    async def invoke(
            self,
            messages: List[dict],
            tools: Optional[List[dict]] = None,
            temperature: float = 0,
            stream_mode: Literal[None, "full", "reasoning_only"] = None,
            reasoning_effort: str = "high",
            reasoning_enabled: bool = True,
            **kwargs,
    ) -> dict:
        """
        调用大语言模型推理接口，支持流式响应

        stream_mode:
        - None: 非流式，返回完整结果，不打印任何内容
        - "full": 流式打印 reasoning_content 和 content
        - "reasoning_only": 流式，只打印 reasoning_content
        """
        result = {"role": "assistant"}
        # 浅拷贝消息列表，避免篡改调用方传入的原始对象
        messages = [dict(msg) for msg in messages]
        for msg in messages:
            # usage 不属于标准 API 消息格式需要剥离
            msg.pop("usage", None)
            # DeepSeek 思考模式要求 assistant 消息必须回传 reasoning_content 否则 API 返回 400，需要加一层对摘要等消息的防护
            if msg.get("role") == "assistant":
                msg.setdefault("reasoning_content", None)

        if stream_mode not in ("full", "reasoning_only"):
            stream_mode = None
        is_streaming = stream_mode is not None

        async with self._semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    stream=is_streaming,
                    reasoning_effort=reasoning_effort,
                    extra_body={"thinking": {"type": "enabled" if reasoning_enabled else "disabled"}},
                    **kwargs,
                )
            except Exception as e:
                display(f"❌ 调用LLM API时发生错误: {e}")
                result["content"] = str(e)
                return result

            if is_streaming:
                # 流式接口返回的是 openai.Stream 对象，其本质是将客户端底层的 HTTP 响应流封装成了一个迭代器
                # 每一次迭代都是从这个已经建立好的响应流中读取下一个可用的数据块
                # 如果服务端还没生成完，迭代器会阻塞等待下一个数据块到来，直到最后收到 [DONE] 标记
                collected_content = []
                collected_reasoning = []
                collected_tool_calls = {}
                usage = None

                try:
                    async for chunk in response:
                        usage = getattr(chunk, "usage", usage)
                        if not chunk.choices:
                            continue

                        delta = chunk.choices[0].delta
                        reasoning = delta.model_extra.get("reasoning_content", "")
                        if reasoning:
                            display_stream(reasoning, prefix="[reasoning]")
                            collected_reasoning.append(reasoning)

                        content = delta.content or ""
                        if content:
                            if stream_mode == "full":
                                display_stream(content)
                            collected_content.append(content)

                        # tool_calls 会按 index 分片下发，id / name 通常只在首片出现，arguments 需要逐片拼接
                        tool_calls = delta.tool_calls or []
                        for tc in tool_calls:
                            slot = collected_tool_calls.setdefault(tc.index or 0, {
                                "id": None,
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            })
                            if tc.id:
                                slot["id"] = tc.id
                            if tc.type:
                                slot["type"] = tc.type
                            if tc.function:
                                if tc.function.name:
                                    slot["function"]["name"] += tc.function.name
                                if tc.function.arguments:
                                    slot["function"]["arguments"] += tc.function.arguments
                finally:
                    # 无论正常结束还是异常，都必须显式收尾流式输出，避免UI状态错乱
                    end_stream()

                result["content"] = "".join(collected_content)
                result["reasoning_content"] = "".join(collected_reasoning)
                result["tool_calls"] = list(collected_tool_calls.values())

            else:
                message = response.choices[0].message
                result["content"] = message.content
                result["reasoning_content"] = message.model_extra.get("reasoning_content", "")
                result["tool_calls"] = [tc.model_dump() for tc in message.tool_calls] if message.tool_calls else None
                usage = response.usage

            result["usage"] = {
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }

        return result