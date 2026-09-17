import inspect
import json
import subprocess
import functools
import asyncio
from typing import Callable, Annotated, get_type_hints, get_origin, get_args
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from typing import Literal

from hook_manager import HookReject, AgentHookManager
from mcp_service.mcp_client import MCPClientManager
from common_utils import type_to_json_schema

def read_file(path: str, offset: int = 0, limit: int = 0) -> str:
    """Read file with optional offsets or line limits"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        start = offset if offset else 0
        end = start + limit if limit else len(lines)
        numbered = [f"{i+1:4d} {line}" for i, line in enumerate(lines[start:end], start)]
        return ''.join(numbered)
    except Exception as e:
        return f"Error: {str(e)}"

def write_file(path: str, content: str) -> str:
    """Write content to file"""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error: {str(e)}"

def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace a unique string in file"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        if content.count(old_string) != 1:
            return f"Error: old_string must appear exactly once (found {content.count(old_string)})"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content.replace(old_string, new_string))
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error: {str(e)}"

def bash(command: str) -> str:
    """Run shell command"""
    result = ""
    try:
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate(timeout=30)
    except Exception as e:
        proc.kill()
        return f"Error: {str(e)}"
    if stdout:
        try:
            result += stdout.decode('utf-8')
        except UnicodeDecodeError:
            result += stdout.decode('gbk', errors='replace')
    if stderr:
        try:
            result += stderr.decode('utf-8')
        except UnicodeDecodeError:
            result += stderr.decode('gbk', errors='replace')
    return result


class TodoManager:
    """
    持有内存中的任务列表，负责校验更新，并把渲染结果返回给模型
    """

    # 嵌套 Pydantic 模型
    class TodoItem(BaseModel):
        content: str
        status: Literal["pending", "in_progress", "completed"]

    def __init__(self):
        self.items: list[TodoManager.TodoItem] = []

    # 这里可以直接使用 TodoItem，它会自动识别为嵌套类
    def update(self, todos: list[TodoItem]) -> str:
        if not isinstance(todos, list):
            raise ValueError("todos must be a list")
        if len(todos) > 20:
            raise ValueError("Max 20 todos allowed")

        validated = []
        in_progress_count = 0
        for index, todo in enumerate(todos):
            if not isinstance(todo, dict):
                raise ValueError(f"todos[{index}] must be an object")

            content = str(todo.get("content", "")).strip()
            status = str(todo.get("status", "pending")).lower()
            if not content:
                raise ValueError(f"todos[{index}] requires content")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"todos[{index}] has invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"content": content, "status": status})

        if in_progress_count > 1:
            raise ValueError("Only one todo can be in_progress at a time")

        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."

        lines = []
        for todo in self.items:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[todo["status"]]
            lines.append(f"{marker} {todo['content']}")

        done = sum(todo["status"] == "completed" for todo in self.items)
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()
def todo_write(todos: list[TodoManager.TodoItem]) -> str:
    """Create and manage a task list for your current coding session."""
    try:
        output = TODO.update(todos)
    except ValueError as e:
        return f"Error: {e}"
    print(f"\n\033[33m## Current Tasks\033[0m\n{output}")
    return output


class ToolManager:
    """管理工具的注册与执行"""
    def __init__(self, timeout: int = 30, max_concurrency: int = 3):
        """初始化时注册基本工具"""
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.tool_registry = {}
        self.mcp_tool_schemas = []
        asyncio.run(self.register_mcp_tools())
        self.register(read_file, write_file, edit_file, bash, todo_write)
        self.rounds_since_todo = 0  # todo reminder 计数器

    def register(self, *func: Callable):
        """将普通函数注册成工具函数"""
        for tool in func:
            name = tool.__name__
            self.tool_registry[name] = tool
            print(f"✅ 工具 '{name}' 已注册。")

    async def register_mcp_tools(self):
        """注册 MCP 工具"""
        mcp_manager = MCPClientManager("mcp_service/config.json")
        all_tools = await mcp_manager.async_list_tools()
        for service_name, tools in all_tools.items():
            for tool in tools:
                # 注册工具的 tool schema
                self.mcp_tool_schemas.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                })
                # 注册工具的可调用对象（适配器），partial 会固定指定的位置/关键字参数，并将后续调用时的 **kwargs 直接透传给底层函数
                adapter = functools.partial(mcp_manager.async_call_tool, service_name, tool["name"])
                adapter.__name__ = tool["name"]
                adapter.__doc__ = tool["description"]
                adapter._is_mcp = True
                self.register(adapter)

    async def exec_tool_call(self, tool_info: dict, hook_manager, agent_label: str = "") -> str:
        """封装通用的工具执行流程"""
        tool_name = tool_info.get("name")
        if tool_name not in self.tool_registry:
            return f"Error: Unknown tool '{tool_name}'"

        raw_args = tool_info.get("arguments", "")
        try:
            tool_args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as error:
            return f"Error: Invalid JSON arguments: {error}"

        # ★工具调用前护栏★
        checked = hook_manager.emit("on_tool_start", tool_args, tool_name)
        if isinstance(checked, HookReject):
            return f"Error: {str(checked)}"

        truncated_args = ', '.join([f"{k}={str(v).replace(chr(10), ' ')[:200]}" for k,v in tool_args.items()])
        print(f"[{agent_label}] [Tool] {tool_name}({truncated_args})")

        try:
            tool_impl = self.tool_registry.get(tool_name)
            timeout = 600 if tool_impl.__name__ == 'run_subagent' else self.timeout
            async with self.semaphore:
                if getattr(tool_impl, '_is_mcp', False):
                    response = await asyncio.wait_for(tool_impl(**tool_args), timeout=timeout)
                else:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(tool_impl, **tool_args),
                        timeout=timeout
                    )
        except TimeoutError:
            print(f"[{agent_label}] [Result] Error: Tool '{tool_name}' execution timed out")
            return f"Error: Tool '{tool_name}' execution timed out"
        except Exception as e:
            print(f"[{agent_label}] [Result] Error: Tool '{tool_name}' raised an exception: {type(e).__name__}: {e}")
            return f"Error: Tool '{tool_name}' raised an exception: {type(e).__name__}: {e}"

        # ★工具调用后护栏★
        response = hook_manager.emit("on_tool_end", response, tool_name)
        print(f"[{agent_label}] [Result] {response.replace(chr(10), ' ').replace(chr(13), ' ')[:200]}")

        # 记录连续有多少轮tool_call没有更新todo
        if tool_name == 'todo_write':
            self.rounds_since_todo = 0
        else:
            self.rounds_since_todo += 1

        return response

    def generate_openai_tool_schema(self, is_sub: bool = False):
        """自动解析工具函数的 docstring 和参数注解 Annotated，生成 OpenAI 标准的 function calling schema"""
        local_tool_schemas = []
        for func_name, func in self.tool_registry.items():
            if is_sub and getattr(func, "_subagent_exclude", False):
                continue
            if getattr(func, "_is_mcp", False):
                continue
            description = inspect.getdoc(func) or ""
            sig = inspect.signature(func)
            hints = get_type_hints(func, include_extras=True)
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                annotation = hints.get(param_name, param.annotation)
                param_description = ""
                real_type = annotation

                # 解析 Annotated
                if get_origin(annotation) is Annotated:
                    args = get_args(annotation)
                    real_type = args[0]  # 真实类型
                    for meta in args[1:]:
                        if isinstance(meta, FieldInfo):
                            param_description = meta.description or ""
                            break
                    # 如果还有其他元数据，可继续处理

                # 生成该参数的 JSON Schema 片段
                param_schema = type_to_json_schema(real_type)
                if param_description:
                    param_schema["description"] = param_description

                properties[param_name] = param_schema

                # 是否必须
                if param.default is param.empty:
                    required.append(param_name)

            local_tool_schemas.append({
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    }
                }
            })

        return local_tool_schemas + self.mcp_tool_schemas
