import os
import json
import asyncio
import threading
import time
import httpx
from openai import AsyncOpenAI
from typing import List, Union, Optional, Literal
from pydantic import BaseModel
from openai.types.chat import ChatCompletion

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

    def __init__(self, config: dict = None, model: str = None, api_key: str = None, base_url: str = None, max_concurrency: int = 3):
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

        self.base_url = base_url

        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=int(os.getenv("TIMEOUT", 120)),
            http_client=httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(proxy=None)),
        )

    async def invoke(
        self,
        messages: List[Union[dict, BaseModel]],
        tools: Optional[List[dict]] = None,
        temperature: float = 0,
        stream_mode: Literal[None, "full", "reasoning_only"] = None,
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

        if stream_mode not in ("full", "reasoning_only"):
            stream_mode = None
        is_streaming = stream_mode is not None

        # usage 是本地统计字段，不属于 API 消息格式；DeepSeek 思考模式要求历史 assistant
        # 消息回传 reasoning_content，压缩摘要等缺失该字段的消息补空串，否则 API 返回 400。
        messages = [
            {k: v for k, v in m.items() if k != "usage"} if isinstance(m, dict) else m
            for m in messages
        ]
        if "deepseek" in self.base_url.lower():
            for m in messages:
                if isinstance(m, dict) and m.get("role") == "assistant":
                    m.setdefault("reasoning_content", "")

        async with self._semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    stream=is_streaming,
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
                        # 开启 include_usage 时，最后一个统计块的 choices 为空，
                        # 直接取 choices[0] 会抛 IndexError 并截断整段输出
                        if not chunk.choices:
                            if getattr(chunk, "usage", None) is not None:
                                usage = chunk.usage
                            continue

                        delta = chunk.choices[0].delta
                        extra = delta.model_extra or {}
                        # reasoning_content 属于非标准扩展字段，SDK 可能放在
                        # model_extra 里，也可能已提升为属性；model_extra 为
                        # None 时直接 .get() 会抛 AttributeError
                        reasoning = (
                            extra.get("reasoning_content")
                            or getattr(delta, "reasoning_content", None)
                            or ""
                        )
                        if reasoning:
                            display_stream(reasoning, prefix="[reasoning]")
                            collected_reasoning.append(reasoning)

                        content = delta.content or ""
                        if content:
                            if stream_mode == "full":
                                display_stream(content)
                            collected_content.append(content)

                        # DeepSeek 等服务的 tool_calls 会按 index 分片下发：
                        # id / name 通常只在首片出现，arguments 需要逐片拼接
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
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

                        if getattr(chunk, "usage", None) is not None:
                            usage = chunk.usage
                finally:
                    # 无论正常结束还是中途异常，都必须按序收尾流式输出，
                    # 否则 UI 的流式行状态和日志会残留半截内容
                    end_stream()

                result["content"] = "".join(collected_content)
                result["reasoning_content"] = "".join(collected_reasoning)
                result["tool_calls"] = [collected_tool_calls[i] for i in sorted(collected_tool_calls)] or None
            else:
                if isinstance(response, str) and response.startswith("data:"):
                    # 将SSE格式的接口响应转换为 OpenAI 格式的 ChatCompletion 对象
                    parsed = json.loads(response[5:])
                    if "error" in parsed:
                        raise Exception(f"API返回错误: {parsed.get('error')}")
                    response = ChatCompletion.model_validate(parsed)
                message = response.choices[0].message
                result["content"] = message.content
                result["reasoning_content"] = (message.model_extra or {}).get("reasoning_content", "")
                result["tool_calls"] = [tc.model_dump() for tc in message.tool_calls] if message.tool_calls else None
                usage = response.usage

            result["usage"] = {
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }

        return result


if __name__ == "__main__":
    # 运行方式（项目根目录下）：python -m src.llm_client
    from pathlib import Path

    _config_path = Path(__file__).resolve().parent.parent / "config.json"
    with open(_config_path, "r", encoding="utf-8") as f:
        _llm_config = json.load(f).get("llm", {})

    launch_ui()

    async def _smoke_test():
        client = OpenAIClient(config=_llm_config)
        messages = [{"role": "user", "content": "用一句话介绍你自己。"}]
        result = await client.invoke(messages, temperature=0, stream_mode="reasoning_only")
        display(result.get("content"))
        # UI 在后台线程按刷新周期绘制，留出一帧时间再退出进程，
        # 避免最后一段流式内容已进入缓冲却还没画到屏幕上
        await asyncio.sleep(0.3)

    try:
        asyncio.run(_smoke_test())
    except ValueError as error:
        print(f"配置错误: {error}")
