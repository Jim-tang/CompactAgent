import asyncio
import json
from pathlib import Path
from typing import Annotated
from pydantic import Field

from src.llm_client import OpenAIClient
from src.context_manager import AgentContextManager
from src.hook_manager import AgentHookManager
from src.tool_manager import ToolManager
from src.memory_manager import MemoryManager
from src.common_utils import subagent_exclude
from src.terminal_ui import launch_ui, display


PROJECT_ROOT = Path(__file__).resolve().parent
with open(PROJECT_ROOT / "config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

ui = launch_ui()
llm_client = OpenAIClient(config=CONFIG.get("llm", {}))
context_manager = AgentContextManager(llm_client, CONFIG.get("compression", {}))
memory_manager = MemoryManager(llm_client, config=CONFIG.get("memory", {}))
hook_manager = AgentHookManager(config=CONFIG.get("hooks", {}))
tool_manager = ToolManager(mcp_config=CONFIG.get("mcp"))


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
    sub_ctx_manager = AgentContextManager(llm_client, CONFIG.get("compression", {}))
    if skill:
        # 子Agent不做渐进式加载，而是直接激活最多一个与任务相关的 Skill
        sub_ctx_manager.activate_skill(skill)
    response = await run_agent(task, sub_ctx_manager, is_sub=True, agent_label=label)
    return f"[subagent({label})] {response}"

tool_manager.register(active_skill, load_skill_resource, retrieve_memory, run_subagent)

async def run_agent(user_input: str, ctx_manager: AgentContextManager, max_iterations: int = 30, is_sub: bool = False, agent_label: str = "") -> str:
    # ★用户输入护栏★
    user_input = await hook_manager.emit("on_agent_start", user_input)
    ctx_manager.add_history({"role": "user", "content": user_input})

    for _ in range(max_iterations):
        messages = await ctx_manager.prepare_message()
        response = await llm_client.invoke(
            messages=messages,
            tools=tool_manager.generate_openai_tool_schema(is_sub),
            stream_mode="reasoning_only",
        )
        ctx_manager.add_history(response)

        if response.get("tool_calls"):
            tool_call_tasks = []
            for tc in response.get("tool_calls"):
                if not tc.get("function"):
                    continue
                task = asyncio.create_task(tool_manager.exec_tool_call(tc, hook_manager, agent_label))
                tool_call_tasks.append(task)
            for coro in asyncio.as_completed(tool_call_tasks):
                tool_call_id, function_response = await coro
                tool_message = {"role": "tool", "tool_call_id": tool_call_id, "content": function_response}
                ctx_manager.add_history(tool_message)
        else:
            return await hook_manager.emit("on_agent_end", response.get("content"))

        # 催更机制（System Reminer）：连续 5 轮没有调用 todo_write 的话自动注入提醒
        if tool_manager.rounds_since_todo >= 5:
            ctx_manager.add_history({"role": "system", "content": "<system-reminder>Update your todos.</system-reminder>"})

    # 超出循环上限时返回对话历史的梗概
    summary = await ctx_manager.summarize_with_llm(ctx_manager.history_messages)
    return f"Max iterations reached, summary:\n{summary}"

async def main():
    ui_error = ui.get_error()
    if ui_error:
        display(f"\n=== UI 启动失败 ===\n{ui_error}\n")
        return

    display(
        f"\n交互式 CLI Agent 已启动",
        "可用命令：",
        "  /exit   退出程序",
        "  /reset  结束当前会话并清空输出",
        "  /compact 触发上下文压缩\n"
    )

    while True:
        user_msg = await asyncio.to_thread(ui.get_input, timeout=0.1)
        if user_msg is None:
            if not ui.is_alive():
                break
            continue

        if not user_msg:
            continue

        if user_msg.startswith('/'):
            cmd = user_msg.lower()
            if cmd in ('/exit', '/quit'):
                display("\n📦 会话结束，正在清理临时文件并归档会话历史")
                await asyncio.sleep(2)
                break
            elif cmd == '/reset':
                await memory_manager.record_session(context_manager.history_messages)
                context_manager.reset_session()
                display("\n🔄 会话已重置，可以开始新的对话。")
                continue
            elif cmd == '/compact':
                try:
                    await context_manager.compact_history()
                except Exception as e:
                    display(f"❌ 压缩时发生错误: {e}")
                continue
            else:
                display(f"未知命令: {user_msg}，可用命令：/exit, /reset, /compact")
                continue

        try:
            response = await run_agent(user_msg, context_manager, agent_label="main")
            display(response)
        except Exception as e:
            display(f"发生错误: {e}")

    await memory_manager.record_session(context_manager.history_messages)

    # ★会话结束护栏★
    await hook_manager.emit("on_session_end")
    ui.request_exit()

if __name__ == "__main__":
    asyncio.run(main())
