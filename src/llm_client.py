import os
import json
import asyncio
import threading
import httpx
from openai import AsyncOpenAI
from typing import List, Union, Optional
from pydantic import BaseModel
from openai.types.chat import ChatCompletion

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
        stream: bool = False,
        **kwargs,
    ) -> dict:
        """
        调用大语言模型推理接口，支持流式响应
        """
        result = {"role": "assistant"}

        async with self._semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    stream=stream,
                    **kwargs,
                )
            except Exception as e:
                print(f"❌ 调用LLM API时发生错误: {e}")
                result["content"] = str(e)
                return result

            if stream:
                # 流式接口返回的是 openai.Stream 对象，其本质是将客户端底层的 HTTP 响应流封装成了一个迭代器
                # 每一次迭代都是从这个已经建立好的响应流中读取下一个可用的数据块
                # 如果服务端还没生成完，迭代器会阻塞等待下一个数据块到来，直到最后收到 [DONE] 标记
                collected_content = []
                collected_reasoning = []
                first_reasoning = True
                collected_tool_calls = []
                async for chunk in response:
                    reasoning = chunk.choices[0].delta.model_extra.get("reasoning_content", "")
                    if reasoning:
                        if first_reasoning:
                            print(f"[思考] {reasoning}", end="", flush=True)
                            first_reasoning = False
                        else:
                            print(reasoning, end="", flush=True)
                        collected_reasoning.append(reasoning)
                    content = chunk.choices[0].delta.content or ""
                    if content:
                        print(content, end="", flush=True)
                    collected_content.append(content)
                    if chunk.choices[0].delta.tool_calls:
                        for tc in chunk.choices[0].delta.tool_calls:
                            collected_tool_calls.append(tc.model_dump())
                print()
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

async def test_tool_calls():
    llmClient = OpenAIClient()
    exampleMessages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "帮我查询一下今天北京的天气，并且计算 25 * 37 等于多少"}
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "计算数学表达式",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "数学表达式"}
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    print("\n=== 测试 tool_calls 输出 ===")
    result = await llmClient.invoke(exampleMessages, tools=tools, stream=False)
    print("\n--- 完整模型响应 ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))

async def test_stream_invoke():
    print("=== 测试流式对话 ===")
    llmClient = OpenAIClient()
    exampleMessages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "请帮我计算 165 * (43 + 10) 的结果，并展示你的计算步骤"}
    ]
    result = await llmClient.invoke(exampleMessages, stream=True)
    print("\n\n--- 完整模型响应 ---")
    print(json.dumps(result, indent=2, ensure_ascii=True))

if __name__ == '__main__':
    try:
        asyncio.run(test_stream_invoke())
        asyncio.run(test_tool_calls())
    except ValueError as e:
        print(e)