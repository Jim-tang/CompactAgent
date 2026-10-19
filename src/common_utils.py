import re
import os
import stat
import platform
import hashlib
import shutil
import tempfile
from wcmatch import glob as wcglob
from typing import Optional, Literal, Dict, get_origin, get_args, List, Union
from pydantic import BaseModel
from datetime import datetime, timezone
from pathlib import Path
from string import Template


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TYPE_EXT_MAP: Dict[str, List[str]] = {
    'py': ['.py', '.pyi'],
    'js': ['.js', '.jsx', '.mjs', '.cjs', '.jsx'],
    'ts': ['.ts', '.tsx', '.mts', '.cts', '.tsx'],
    'rust': ['.rs'],
    'go': ['.go'],
    'java': ['.java'],
    'c': ['.c', '.h'],
    'cpp': ['.cpp', '.cc', '.cxx', '.hpp', '.hh', '.hxx', '.h'],
    'h': ['.h', '.hpp'],
    'html': ['.html', '.htm'],
    'css': ['.css'],
    'json': ['.json'],
    'md': ['.md', '.markdown'],
    'yaml': ['.yaml', '.yml'],
    'sh': ['.sh', '.bash', '.zsh'],
    'rb': ['.rb'],
    'php': ['.php'],
    'swift': ['.swift'],
    'kt': ['.kt', '.kts'],
    'scala': ['.scala'],
    'sql': ['.sql'],
    'xml': ['.xml'],
    'toml': ['.toml'],
    'ini': ['.ini'],
}

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

    # 处理 Pydantic BaseModel（非泛型，且是 BaseModel 子类）
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

    # 泛型处理（list, dict, Literal, Optional[T]等）
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
    if origin is Union:
        args = get_args(tp)
        not_none = [a for a in args if a is not type(None)]
        if len(not_none) == 1:
            return type_to_json_schema(not_none[0])

    # 基础类型映射
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

def collect_runtime() -> dict[str, str]:
    """收集跨平台运行时环境信息，所有值均为字符串，可直接用于模板替换。"""
    cwd = Path.cwd()
    return {
        "working_directory": str(cwd),
        "is_git_repo": str((cwd / ".git").exists()).lower(),
        "platform": platform.system().lower(),
        "os_version": platform.version(),
        "current_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "python_version": platform.python_version(),
    }

def render_prompt(template: str, variables: dict[str, str], max_depth: int = 3) -> str:
    """渲染提示词模板：先校验占位符，再递归替换，返回最终字符串。"""
    required = set(re.findall(r"\$\{(\w+)\}", template))
    missing = required - set(variables)
    if missing:
        raise ValueError(f"Missing required variables: {sorted(missing)}")

    result = template
    for _ in range(max_depth):
        prev = result
        result = Template(result).safe_substitute(variables)
        if result == prev:
            break
    return result

def load_gitignore(root: str) -> list[str]:
    path = os.path.join(root, ".gitignore")
    if not os.path.isfile(path):
        return []
    patterns: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "!")):
                continue
            line = line.lstrip("/").rstrip("/")
            if line:
                patterns.append(line)
                patterns.append(f"{line}/**")
    return patterns

def filter_by_gitignore(files: List[str], root: str, patterns: List[str]) -> List[str]:
    """Drop files matched by any gitignore pattern (paths relative to `root`)."""
    if not patterns:
        return files
    rels = [os.path.relpath(f, root).replace(os.sep, '/') for f in files]
    flags = wcglob.GLOBSTAR | wcglob.NEGATE | wcglob.DOTMATCH | wcglob.FORCEUNIX
    negs = [f"!{p}" for p in patterns]
    kept = set(wcglob.globfilter(rels, ["**"] + negs, flags=flags))
    return [f for f, r in zip(files, rels) if r in kept]

def safe_mtime(path: str) -> float:
    """获取文件修改时间"""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0

def resolve_root(path: Optional[str]) -> str:
    """解析搜索根目录"""
    if path:
        abs_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f'Path does not exist: {path}')
    else:
        abs_path = os.getcwd()
    return abs_path

def collect_files(path: str) -> List[str]:
    """递归搜集目标路径下的文件清单"""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Path does not exist: {path}')
    if os.path.isfile(path):
        return [path]
    files: List[str] = []
    vcs_to_exclude = frozenset(['.git', '.svn', '.hg', '.bzr', '.jj', '.sl'])
    win_reserved = ['CON', 'PRN', 'AUX', 'NUL']
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d not in vcs_to_exclude]
        for f in filenames:
            if f.split('.', 1)[0].upper() in win_reserved:
                continue
            files.append(os.path.join(dirpath, f))
    return files

def filter_by_type(files: List[str], target_type: Optional[str]) -> List[str]:
    """根据目标类型对应的文件后缀过滤文件清单"""
    if not target_type:
        return files
    exts = TYPE_EXT_MAP.get(target_type)
    if exts is None:
        raise ValueError(f'Unknown file type: {target_type}')
    return [f for f in files if any(f.endswith(e) for e in exts)]

def content_lines_for_file(
    abs_path: str,
    root_path: str,
    regex: re.Pattern,
    show_line_numbers: bool,
    context: Optional[int],
    max_columns: Optional[int] = 500
) -> List[str]:
    """单文件内的 content 行生成：
       匹配行  -> path:lineno:content    （无 -n 时 path:content）
       上下文行 -> path-lineno-content   （与 rg 的 `-` 分隔符一致）
    """
    with open(abs_path, encoding='utf-8', errors='replace') as fp:
        text = fp.read()

    if not text:
        return []

    lines = text.splitlines()
    match_flags: Dict[int, bool] = {
        idx: True for idx, line in enumerate(lines) if regex.search(line)
    }
    if not match_flags:
        return []

    if context is None or context <= 0:
        to_show: Dict[int, bool] = dict(match_flags)
    else:
        to_show = {}
        for li in match_flags:
            for j in range(max(0, li - context), min(len(lines), li + context + 1)):
                to_show[j] = to_show.get(j, False) or (j == li)

    out = []
    rel_path = os.path.relpath(abs_path, root_path)
    for idx in sorted(to_show):
        sep = ':' if to_show[idx] else '-'  # 匹配行 :，上下文行 -
        body = lines[idx][:max_columns]  # 截断超长行
        if show_line_numbers:
            out.append(f'{rel_path}{sep}{idx + 1}{sep}{body}')
        else:
            out.append(f'{rel_path}{sep}{body}')
    return out

def human_size(n: int) -> str:
    """将字节大小转换为已读的文件大小"""
    if n < 1024:
        return f"{n}B"
    for unit in ("K", "M", "G", "T", "P"):
        n /= 1024
        if n < 1024:
            return f"{n:.1f}{unit}" if n < 10 else f"{n:.0f}{unit}"
    return f"{n:.1f}E"

def mode_string(mode: int) -> str:
    """类似 `ls -l` 的 10 字符权限串，含 setuid/setgid/sticky。"""
    if stat.S_ISDIR(mode):      kind = "d"
    elif stat.S_ISLNK(mode):    kind = "l"
    elif stat.S_ISREG(mode):    kind = "-"
    elif stat.S_ISCHR(mode):    kind = "c"
    elif stat.S_ISBLK(mode):    kind = "b"
    elif stat.S_ISFIFO(mode):   kind = "p"
    elif stat.S_ISSOCK(mode):   kind = "s"
    else:                       kind = "?"

    chars = [kind]

    def bit(mask: int, yes: str) -> str:
        return yes if mode & mask else "-"

    chars += [bit(stat.S_IRUSR, "r"), bit(stat.S_IWUSR, "w")]
    if mode & stat.S_ISUID:
        chars.append("s" if mode & stat.S_IXUSR else "S")
    else:
        chars.append(bit(stat.S_IXUSR, "x"))

    chars += [bit(stat.S_IRGRP, "r"), bit(stat.S_IWGRP, "w")]
    if mode & stat.S_ISGID:
        chars.append("s" if mode & stat.S_IXGRP else "S")
    else:
        chars.append(bit(stat.S_IXGRP, "x"))

    chars += [bit(stat.S_IROTH, "r"), bit(stat.S_IWOTH, "w")]
    if mode & stat.S_ISVTX:
        chars.append("t" if mode & stat.S_IXOTH else "T")
    else:
        chars.append(bit(stat.S_IXOTH, "x"))

    return "".join(chars)

def atomic_write_with_backup(file_path: str, content: str, original: str) -> str:
    """通过临时文件 + 重命名实现原子写入，同时保留旧文件备份方便回退"""
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:16]
    file_name = os.path.basename(file_path)
    backup_dir = os.path.join(PROJECT_ROOT, ".temp")
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, f"{file_name}.{digest}.bak")

    # 在临时路径备份原始文件
    if not os.path.exists(backup_path):
        try:
            shutil.copy2(file_path, backup_path)
        except:
            pass  # 宽松处理，备份失败不影响写入

    # 写入临时文件
    dir_name = os.path.dirname(file_path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".edit_", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp_path, stat.S_IMODE(os.stat(file_path).st_mode))  # 复刻原文件权限
    os.replace(tmp_path, file_path)  # 原子替换

    return backup_path