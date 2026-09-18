import json
import os
import httpx
import asyncio
from typing import Annotated
from pydantic import Field
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from context_manager import AgentContextManager
from hook_manager import AgentHookManager
from tool_manager import ToolManager
from memory_manager import MemoryManager
from common_utils import subagent_exclude

os.environ['MODEL'] = 'MiniMax-M2.7'
client = AsyncOpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="123",
    http_client=httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(proxy=None)),
)

# 全局上下文压缩配置
COMPRESSION_CONFIG = {
    "mode": "message_count",  # message_count 或 token_ratio
    "threshold": 30,  # 触发压缩的阈值（消息数或比例0~1）
    "max_context_length": 64000,  # token_ratio 模式的最大上下文窗口
    "reserved_tokens": 8000,  # token_ratio 模式为 Tool Schema 预留的空间
    "keep_recent": 6,  # 压缩时保留最近的消息条数
}

# 全局管理器实例
memory_manager = MemoryManager()
hook_manager = AgentHookManager()
tool_manager = ToolManager()
context_manager = AgentContextManager(client, COMPRESSION_CONFIG)  # 管理主Agent上下文

@subagent_exclude
def active_skill(
    name: Annotated[str, Field(description="Name of the skill to activate")]
) -> str:
    """Activate a specialized skill by name. Use this when user request matches an available skill."""
    return context_manager.activate_skill(name)

@subagent_exclude
def load_skill_resource(
    skill: Annotated[str, Field(description="Name of the active skill")],
    path: Annotated[str, Field(description="Relative path within skill directory (e.g., references/api.md)")]
) -> str:
    """Load a reference document belonging to an active skill."""
    return context_manager.load_resource(skill, path)

def retrieve_memory(
    query: Annotated[str, Field(description="A complete natural language query to retrieve relevant memories from the agent's memory store")]
) -> str:
    """Retrieve relevant memories (both short-term and long-term) based on the inputted query."""
    formatted_memory = "检索到的相关记忆（按相关性排序）:\n"
    result = memory_manager.retrieve(query, top_k=4)
    if not result:
        return "No relevant memories found."
    for idx, (doc, score) in enumerate(result):
        mem_type = doc.metadata.get("memory_type")
        content = doc.page_content if mem_type == "long_term" else doc.metadata.get("content")
        timestamp = doc.metadata.get("timestamp")
        mem_str = f"{idx}. [记忆类型: {mem_type}] [时间: {timestamp}] [相关性: {score:.2f}]\n   {content}"
        formatted_memory += f"\n{mem_str}\n"
    return formatted_memory

@subagent_exclude
async def run_subagent(
        task: Annotated[str, Field(description="The task assigned to the subagent")],
        skill: Annotated[str, Field(description="The available Skill required to complete the task", default="")],
        label: Annotated[str, Field(description="Label to identify this subagent", default="Sub-Agent")]
) -> str:
    """Run a subagent with fresh conversation context and one matched skill (if necessary), then return its final response."""
    # 子Agent独立的上下文管理器
    sub_ctx_manager = AgentContextManager(client, COMPRESSION_CONFIG)
    if skill:
        # 子Agent不做渐进式加载，而是直接激活最多一个与任务相关的 Skill
        sub_ctx_manager.activate_skill(skill)
    response = await run_agent(task, sub_ctx_manager, is_sub=True, agent_label=label)
    return f"[subagent({label})] {response}"

tool_manager.register(active_skill, load_skill_resource, retrieve_memory, run_subagent)

async def run_agent(user_input: str, ctx_manager: AgentContextManager, max_iterations: int = 30, is_sub: bool = False, agent_label: str = "") -> str:
    system_prompt = "You are a helpful assistant. Be concise."
    # ★用户输入护栏★
    user_input = hook_manager.emit("on_agent_start", user_input)
    ctx_manager.add_history({"role": "user", "content": user_input})

    for _ in range(max_iterations):
        messages = await ctx_manager.prepare_message(system_prompt)  # 一站式上下文处理：扩展 + 压缩
        response = await client.chat.completions.create(
            model=os.environ.get("MODEL"),
            messages=messages,
            tools=tool_manager.generate_openai_tool_schema(is_sub),
        )
        parsed = json.loads(response[5:])
        if "error" in parsed:
            raise Exception(f"API返回错误: {parsed.get('error')}")

        response = ChatCompletion.model_validate(parsed)
        assist_message = response.choices[0].message
        ctx_manager.add_history(assist_message.model_dump())

        if not assist_message.tool_calls:
            # ★最终输出护栏★
            return hook_manager.emit("on_agent_end", assist_message.content)

        for tool_call in assist_message.tool_calls:
            function_payload = getattr(tool_call, "function", None)
            if function_payload is None:
                continue
            function_response = await tool_manager.exec_tool_call(function_payload.model_dump(), hook_manager, agent_label)
            tool_message = {"role": "tool", "tool_call_id": tool_call.id, "content": function_response}
            ctx_manager.add_history(tool_message)

        # 催更机制（Nag Reminer）：连续 5 轮没有调用 todo_write 的话自动注入提醒
        if tool_manager.rounds_since_todo >= 5:
            ctx_manager.add_history({"role": "system", "content": "<reminder>Update your todos.</reminder>"})

    # 超出循环上限时返回对话历史的梗概
    full_history = ctx_manager.get_messages_text(ctx_manager.history_messages)
    summary = await ctx_manager.summarize_with_llm(full_history)
    return f"Max iterations reached, summary:\n{summary}"

async def main():
    print("交互式 CLI Agent 已启动。可用命令：\n/exit 退出程序\n/reset 结束当前会话并开始新会话\n/compact 触发上下文压缩")
    while True:
        try:
            user_msg = await asyncio.to_thread(input, "\nUSER: ")
        except (EOFError, KeyboardInterrupt):
            print("\n退出程序")
            break

        if not user_msg.strip():
            continue

        # Command Router
        if user_msg.startswith('/'):
            cmd = user_msg.lower().strip()
            if cmd in ('/exit', '/quit'):
                break
            elif cmd == '/reset':
                # 记录当前会话历史
                full_history = context_manager.get_messages_text(context_manager.history_messages)
                await memory_manager.record_session(full_history)
                # 清空上下文历史，开始新会话
                context_manager.reset_session()
                print("🔄 会话已重置，可以开始新的对话。")
                continue
            elif cmd == '/compact':
                # 主动触发上下文压缩
                try:
                    await context_manager.compact_history()
                except Exception as e:
                    print(f"❌ 压缩时发生错误: {e}")
                continue
            else:
                print(f"未知命令: {user_msg}，可用命令：/exit, /reset, /compact")
                continue

        try:
            response = await run_agent(user_msg, context_manager, agent_label="main")
            print(f"\nAgent: {response}")
        except Exception as e:
            print(f"发生错误: {e}")

    # 程序退出前，记录本次会话
    full_history = context_manager.get_messages_text(context_manager.history_messages)
    await memory_manager.record_session(full_history)

    print("程序已退出。")


if __name__ == "__main__":
    asyncio.run(main())
