import asyncio
import json
from typing import Annotated
from pydantic import Field

from src.llm_client import OpenAIClient
from src.context_manager import AgentContextManager
from src.hook_manager import AgentHookManager
from src.session_manager import SessionManager
from src.tool_manager import ToolManager
from src.terminal_ui import launch_ui, display
from src.memory import MemoryManager
from src.common_utils import PROJECT_ROOT, subagent_exclude


with open(PROJECT_ROOT / "config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

ui = launch_ui()
llm_client = OpenAIClient(config=CONFIG.get("llm"))
context_manager = AgentContextManager(llm_client, CONFIG.get("compression"))
hook_manager = AgentHookManager(config=CONFIG.get("hooks"))
tool_manager = ToolManager(mcp_config=CONFIG.get("mcp"))
memory_manager = MemoryManager(llm_client, CONFIG.get("memory"))
session_manager = SessionManager()


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


async def retrieve_memory(
    query: Annotated[str, Field(description="A complete natural language query describing what to look up in long-term memory")],
    top_k: Annotated[int, Field(description="How many memories to return (default 4)")] = 4
) -> str:
    """Search the agent's long-term memory (L1 atomic memories and L2 scene summaries) with hybrid retrieval."""
    return await memory_manager.retrieval_report(query, top_k=top_k)


async def search_raw_conversation(
    query: Annotated[str, Field(description="Keywords to look for in the original conversation logs")],
    top_k: Annotated[int, Field(description="How many raw messages to return (default 5)")] = 5
) -> str:
    """Search the raw conversation log by keyword.

    Results are single raw messages, each truncated to a 500-character window. Use it as the ground truth only in these circumstances:
    - When memories from other sources conflict
    - When you are unsure whether a memory is accurate
    - When you need the exact original content
    """
    return await memory_manager.raw_conversation_report(query, top_k=top_k)


async def read_scene(
    scene_id_or_name: Annotated[str, Field(description="The L2 scene id (ULID) or the scene name shown in the memory index")]
) -> str:
    """Fetch the full narrative text of one long-term memory scene (L2) by its id or name."""
    return await memory_manager.scene_report(scene_id_or_name)


@subagent_exclude
async def run_subagent(
    task: Annotated[str, Field(description="A short (3-5 word) description of the task for the subagent to perform")],
    skill: Annotated[str, Field(description="The available Skill required to complete the task")] = "",
    label: Annotated[str, Field(description="Label to identify this subagent")] = "General-Agent"
) -> str:
    """
    Run a subagent to handle tasks autonomously with fresh conversation context and one matched skill (if necessary), then return its final response.

    Use this when you are researching complex questions, searching for code, or executing other complex and multistep tasks.
    """
    # 子Agent独立的上下文管理器
    sub_ctx_manager = AgentContextManager(llm_client, CONFIG.get("compression", {}))
    if skill:
        # 子Agent不做渐进式加载，而是直接激活最多一个与任务相关的 Skill
        sub_ctx_manager.activate_skill(skill)
    response = await run_agent(task, sub_ctx_manager, is_sub=True, agent_label=label)
    return f"[subagent({label})] {response}"


tool_manager.register(active_skill, load_skill_resource, run_subagent)
if memory_manager.enabled:
    tool_manager.register(retrieve_memory, search_raw_conversation, read_scene)


async def run_agent(user_input: str, ctx_manager: AgentContextManager, max_iterations: int = 30, is_sub: bool = False, agent_label: str = "") -> str:
    # ★用户输入护栏★
    user_input = await hook_manager.emit("on_agent_start", user_input)
    ctx_manager.add_message({"role": "user", "content": user_input})

    for _ in range(max_iterations):
        messages = await ctx_manager.prepare_message()
        response = await llm_client.invoke(
            messages=messages,
            tools=tool_manager.generate_openai_tool_schema(is_sub),
            stream_mode="reasoning_only",
        )
        ctx_manager.add_message(response)

        # 只落盘主 Agent 的 L0 记忆
        if not is_sub:
            raw_messages = context_manager.get_raw_messages()
            await memory_manager.sync_from(raw_messages)

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
                ctx_manager.add_message(tool_message)
        else:
            return await hook_manager.emit("on_agent_end", response.get("content"))

        # 催更机制（System Reminer）：连续 5 轮没有调用 todo_write 的话自动注入提醒
        if tool_manager.rounds_since_todo >= 5:
            ctx_manager.add_message({"role": "system", "content": "<system-reminder>Update your todos.</system-reminder>"})

    # 超出循环上限时返回对话历史的梗概
    summary = await ctx_manager.summarize_with_llm(ctx_manager.history_messages, "archive_session")
    return f"Max iterations reached, summary:\n{summary}"


async def start_new_session():
    session_id = session_manager.get_id()
    memory_manager.start_session(session_id)
    inject_memory = await memory_manager.build_context_block(session_id)
    context_manager.set_memory_block(inject_memory)


async def end_cur_session():
    try:
        await memory_manager.end_session()
        await context_manager.archive_session()
    except Exception as e:
        display(f"[Agent] Warning: 会话归档失败: {e}")


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
        "  /compact 触发上下文压缩",
        "  /memory 查看记忆系统状态\n"
    )

    try:
        await memory_manager.start()
    except Exception as e:
        display(f"[Agent] Warning: 记忆系统启动失败，本次运行不启用记忆: {e}")
    await start_new_session()

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
                display("\n[Agent] 会话结束，正在清理临时文件并归档会话历史")
                await asyncio.sleep(2)
                break
            elif cmd == '/reset':
                await end_cur_session()
                context_manager.reset_session()
                await start_new_session()
                display("\n[Agent] 会话已重置，可以开始新的对话。")
                continue
            elif cmd == '/compact':
                try:
                    await context_manager.manual_compact()
                except Exception as e:
                    display(e)
                continue
            elif cmd == '/memory':
                display(await memory_manager.describe() if memory_manager.enabled else "[Memory] 未启用")
                continue
            else:
                display(f"未知命令: {user_msg}，可用命令：/exit, /reset, /compact, /memory")
                continue

        try:
            response = await run_agent(user_msg, context_manager, agent_label="main")
            display(f"{response}\n")
        except Exception as e:
            display(f"发生错误: {e}")

    await end_cur_session()

    try:
        await memory_manager.close(flush=True)
    except Exception as e:
        display(f"[Agent] Warning: 记忆系统关闭异常: {e}")

    # ★会话结束护栏★
    await hook_manager.emit("on_session_end")
    ui.request_exit()


if __name__ == "__main__":
    asyncio.run(main())
