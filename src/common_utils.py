import re
import os
from typing import Optional, Literal, get_origin, get_args, List
from pydantic import BaseModel

def subagent_exclude(func):
    """
    装饰器：标记该工具函数在子Agent中不可见。
    子Agent的注册表构建时会自动过滤掉带有此标记的函数。
    """
    func._subagent_exclude = True
    return func


def get_workspace_tree(root_dir, max_depth=3, current_depth=0, prefix="",
                       excludes: Optional[list[str]] = None):
    """递归扫描工作区目录结构，生成系统提示中的目录树"""
    lines = []
    if current_depth > max_depth:
        return ""
    if current_depth == 0:
        lines.append(root_dir)

    path_excludes = [r"^\.", r"__pycache__"]
    if excludes:
        path_excludes += excludes
    patterns = [re.compile(p) for p in path_excludes]
    items = []
    for item in os.listdir(root_dir):
        if not any(pat.match(item) for pat in patterns):
            items.append(item)
    items.sort()

    for idx, item in enumerate(items):
        path = os.path.join(root_dir, item)
        is_last = (idx == len(items) - 1)
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + item)
        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            lines.append(get_workspace_tree(path, max_depth, current_depth + 1, prefix + extension, excludes))
    return "\n".join(lines)


def type_to_json_schema(tp):
    """将 Python 类型转换为 JSON Schema 片段（支持 Pydantic 模型展开）"""
    origin = get_origin(tp)

    # ----- 1. 处理 Pydantic BaseModel（非泛型，且是 BaseModel 子类） -----
    if origin is None and isinstance(tp, type) and issubclass(tp, BaseModel):
        # 获取模型字段信息（Pydantic v2 使用 model_fields）
        fields = tp.model_fields
        properties = {}
        required = []
        for field_name, field_info in fields.items():
            # field_info 是 pydantic.fields.FieldInfo 实例
            # 获取字段类型
            field_type = field_info.annotation
            # 获取描述：可从 field_info.description 或 metadata 中提取
            description = field_info.description or ""
            # 生成该字段的 schema
            field_schema = type_to_json_schema(field_type)
            if description:
                field_schema["description"] = description
            properties[field_name] = field_schema
            # 判断是否必填：如果没有默认值，且不是可选类型（Optional），则视为 required
            if field_info.is_required():
                required.append(field_name)
        return {"type": "object", "properties": properties, "required": required}

    # ----- 2. 原有的泛型处理（list, dict, Literal 等） -----
    if origin is list:
        args = get_args(tp)
        if args:
            item_schema = type_to_json_schema(args[0])
            return {"type": "array", "items": item_schema}
        else:
            return {"type": "array"}
    if origin is dict:
        return {"type": "object"}
    if origin is Literal:
        args = get_args(tp)
        return {"type": "string", "enum": list(args)}
    # 可以继续添加其他泛型如 Union, Optional 等（详见后文）

    # ----- 3. 基础类型映射 -----
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        # 可以加更多
    }
    if tp in type_map:
        return {"type": type_map[tp]}

    # 如果都未匹配，兜底为 string
    return {"type": "string"}


def get_messages_text(messages: List[dict]) -> str:
    messages_text = ""
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        msg_text = f"{role.upper()}: {content}"
        if tool_calls:
            tool_calls_list = []
            for tc in tool_calls:
                function = tc.get("function", {})
                if not function:
                    continue
                func_name = function.get("name", "unknown_tool")
                func_args = function.get("arguments", "")
                tool_calls_list.append(f"{func_name}({func_args})")
            if tool_calls_list:
                tool_calls_str = f" tool_calls: {', '.join(tool_calls_list)}"
                msg_text += tool_calls_str
        messages_text += msg_text + "\n"

    return messages_text


if __name__ == "__main__":
    workspace_tree = get_workspace_tree(os.getcwd(), excludes=["mcp_service", "skills", "chroma_db"])
    print(workspace_tree)
