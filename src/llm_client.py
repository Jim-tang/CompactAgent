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

from terminal_ui import display_stream, end_stream, launch_ui, display

os.environ['MODEL'] = 'MiniMax-M2.7'
os.environ['API_KEY'] = '123'
os.environ['BASE_URL'] = 'http://127.0.0.1:8080/v1'


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

    def __init__(self, model: str = "", api_key: str = "", base_url: str = "", max_concurrency: int = 3):
        # 防止单例重复初始化
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        
        self.model = model or os.getenv("MODEL")
        api_key = api_key or os.getenv("API_KEY")
        base_url = base_url or os.getenv("BASE_URL")
        if not all([self.model, api_key, base_url]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

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
                collected_tool_calls = []
                async for chunk in response:
                    reasoning = chunk.choices[0].delta.model_extra.get("reasoning_content", "")
                    if reasoning:
                        display_stream(reasoning, prefix="[reasoning]")
                        collected_reasoning.append(reasoning)

                    content = chunk.choices[0].delta.content or ""
                    if content:
                        if stream_mode == "full":
                            display_stream(content)
                        collected_content.append(content)

                    if chunk.choices[0].delta.tool_calls:
                        for tc in chunk.choices[0].delta.tool_calls:
                            collected_tool_calls.append(tc.model_dump())
                end_stream()

                result["content"] = "".join(collected_content)
                result["reasoning_content"] = "".join(collected_reasoning)
                result["tool_calls"] = collected_tool_calls if collected_tool_calls else None
                usage = chunk.usage
            else:
                if isinstance(response, str) and response.startswith("data:"):
                    # 将SSE格式的接口响应转换为 OpenAI 格式的 ChatCompletion 对象
                    parsed = json.loads(response[5:])
                    if "error" in parsed:
                        raise Exception(f"API返回错误: {parsed.get('error')}")
                    response = ChatCompletion.model_validate(parsed)
                message = response.choices[0].message
                result["content"] = message.content
                result["reasoning_content"] = response.choices[0].message.model_extra.get("reasoning_content", "")
                result["tool_calls"] = [tc.model_dump() for tc in message.tool_calls] if message.tool_calls else None
                usage = response.usage

            result["usage"] = {
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }

        return result