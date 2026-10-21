import os
import re
import stat
import threading
from datetime import datetime
from wcmatch import glob as wcglob
from typing import Literal, Optional, Annotated
from pydantic import Field
from collections import namedtuple

import src.common_utils as util


class FileTracker:
    """追踪文件的已读快照（mtime_ns + size），实现先读后写保护"""
    ReadState = namedtuple("ReadState", ["mtime_ns", "size"])

    def __init__(self):
        self._states: dict[str, FileTracker.ReadState] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(file_path: str) -> str:
        return os.path.realpath(os.path.abspath(file_path))

    @staticmethod
    def _snapshot(file_path: str) -> Optional[tuple[int, int]]:
        try:
            st = os.stat(file_path)
        except OSError:
            return None
        return st.st_mtime_ns, st.st_size

    def mark_read(self, file_path: str) -> None:
        snap = self._snapshot(file_path)
        if snap is None:
            return
        with self._lock:
            self._states[self._key(file_path)] = FileTracker.ReadState(*snap)

    def check(self, file_path: str) -> Optional[str]:
        """返回错误信息；None 表示允许写入。"""
        snap = self._snapshot(file_path)
        if snap is None:
            return None
        with self._lock:
            state = self._states.get(self._key(file_path))

        if state is None:
            return f"Error: file has not been read yet: {file_path}. Call the file_read tool before modifying."
        if (state.mtime_ns, state.size) != snap:
            return f"Error: file has been modified since it was last read: {file_path}. Call the file_read tool again to obtain the latest state."

        return None

tracker = FileTracker()

def file_read(
    file_path: Annotated[str, Field(description="The absolute path to the file to read")],
    offset: Annotated[Optional[int], Field(description="The line number to start reading from. Only provide if the file is too large to read at once.", ge=1)] = 1,
    limit: Annotated[Optional[int], Field(description="The number of lines to read. Only provide if the file is too large to read at once.")] = 500,
)-> str:
    """Read a text-only file from the local filesystem and returns its contents with 1-based line number.

    Usage:
    - By default up to 500 lines are read, starting from the beginning of the file.
    - For large files, pass `offset` (the index of the first line to return, 1 means "from the top") and `limit` (maximum number of lines).
    - Any line longer than 1000 characters is truncated and marked with "… [line truncated]".
    - If more lines remain after the returned range, the output ends with a hint giving the `offset` to use for the next call.
    - You have the capability to call multiple tools in a single response. It is recommended to speculatively read multiple files as a batch that are potentially useful.
    """
    if not os.path.exists(file_path):
        return f"Error: File does not exist: {file_path}"
    if not os.path.isfile(file_path):
        return f"Error: Path is not a regular file: {file_path}"

    collected: list[tuple[int, str]] = []
    file_has_any_line = False
    has_more = False

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            # 流式读取，避免把大文件全读进内存
            for lineno, raw in enumerate(f, start=1):
                file_has_any_line = True
                if lineno < offset:
                    continue
                if len(collected) >= limit:
                    has_more = True
                    break
                collected.append((lineno, raw))
    except Exception as e:
        return f"Error: {str(e)}"

    if not collected:
        return f"No content: offset {offset} is past the end of the file ({file_path})."

    tracker.mark_read(file_path)

    if not file_has_any_line:
        return "Warning: the file exists but the contents are empty."

    out_lines = []
    # 添加行号并截断超长行
    for lineno, raw in collected:
        text = raw.rstrip("\n").rstrip("\r")
        if len(text) > 1000:
            text = text[:1000] + "… [line truncated]"
        out_lines.append(f"{lineno:04d}\t{text}")

    if has_more:
        next_offset = collected[-1][0] + 1
        out_lines.append(f"... (More lines available, call file_read again with offset={next_offset})")

    return "\n".join(out_lines)


def file_write(
    file_path: Annotated[str, Field(description="The absolute path to the file to write")],
    content: Annotated[str, Field(description="The content to write to the file")]
) -> str:
    """Writes a file to the local filesystem with the "utf-8" encoding.

    Usage:
    - This tool will overwrite the existing file if there is one at the provided path.
    - Prefer the file_edit tool for modifying existing files. Only use this tool to create new files or for complete rewrites.
    - NEVER create documentation files (*.md) or README files unless explicitly requested by the User.
    - Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked.
    - The `file_path` must be a distinct file path that includes a filename with an appropriate extension.
    """
    try:
        # 写入前检查文件是否被读取过
        if os.path.isfile(file_path):
            error = tracker.check(file_path)
            if error:
                return error

        action = "updated" if os.path.isfile(file_path) else "created"
        parent = os.path.dirname(file_path)
        os.makedirs(parent, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        tracker.mark_read(file_path)
        return f"Successfully {action} file: {file_path}"
    except Exception as e:
        return f"Error: {str(e)}"


def file_edit(
    file_path: Annotated[str, Field(description="The absolute path to the file to modify")],
    old_string: Annotated[str, Field(description="The text to replace")],
    new_string: Annotated[str, Field(description="The text to replace it with (must be different from old_string)")],
    replace_all: Annotated[Optional[bool], Field(description="Replace all occurrences of old_string (default false)")] = False,
) -> str:
    """
    Performs exact string replacements in files.

    Usage:
    - Use the smallest `old_string` that's clearly unique — usually 2-4 adjacent lines is sufficient. Avoid including 10+ lines of context when less uniquely identifies the target.
    - When editing text from file_read tool output, Everything after the line number prefix as "{lineno:04d}\t" is the actual file content to match. Never include any part of the line number prefix in the `old_string` or `new_string`.
    - ALWAYS prefer editing existing files. NEVER write new files unless explicitly required.
    - Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
    - The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
    - Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.
    - The `file_path` must be a distinct file path that includes a filename with an appropriate extension.
    """
    if not os.path.exists(file_path):
        return f"Error: File does not exist: {file_path}"
    if not os.path.isfile(file_path):
        return f"Error: Path is not a regular file: {file_path}"

    if old_string == new_string:
        return "Error: old_string and new_string must be different."
    if old_string == "":
        return "Error: old_string must not be empty. Use the file_write tool to create a new file."

    # 先读后写保护
    error = tracker.check(file_path)
    if error:
        return error

    abs_path = os.path.abspath(file_path)
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            original = f.read()
    except Exception as e:
        return f"Error reading file: {e}"

    count = original.count(old_string)
    if count == 0:
        return f"Error: old string not found in file: {file_path}"
    if count > 1 and not replace_all:
        return f"Error: old_string is not unique in the file, found {count} occurrences."
    updated = original.replace(old_string, new_string, count)

    try:
        backup_path = util.atomic_write_with_backup(abs_path, updated, original)
    except Exception as e:
        return f"Error writing file: {e}"

    tracker.mark_read(file_path)
    return f"Successfully edit file: {file_path} ({count} replacement(s)). Backup file: {backup_path}"


def file_glob(
    pattern: Annotated[str, Field(description="The glob pattern to match files against")],
    path: Annotated[Optional[str], Field(description="Optional directory path to search in. If not specified, the current working directory will be used.")] = None,
    respect_gitignore: Annotated[bool, Field(description="Whether to respect the root .gitignore file when searching")] = True,
) -> str:
    """Fast file pattern matching tool that returns matching file paths sorted by modification time.

    Usage notes:
    - Use this tool when you need to find files by name patterns like "**/*.js" or "src/**/*.ts"
    - It is always better to speculatively perform multiple searches as a batch that are potentially useful.
    - When you are doing an open-ended search that may require multiple rounds of globbing or grepping, use the run_subagent tool instead
    """
    search_root = util.resolve_root(path)
    if not os.path.isdir(search_root):
        raise NotADirectoryError(search_root)

    # 参照 ripgrep 的 basename 规则：模式不含 / 时匹配任意深度
    patterns = [pattern if "/" in pattern else f"**/{pattern}"]
    flags = wcglob.GLOBSTAR | wcglob.BRACE | wcglob.DOTMATCH

    if respect_gitignore:
        # NEGATE 让 !pattern 生效，一次调用完成 include + exclude
        flags |= wcglob.NEGATE
        ignore_patterns = util.load_gitignore(search_root)
        patterns.extend(f"!{p}" for p in ignore_patterns)

    matches = wcglob.glob(patterns, flags=flags, root_dir=search_root)
    full_paths = [os.path.join(search_root, m) for m in matches]
    full_paths.sort(key=util.safe_mtime)
    return "\n".join(full_paths)


def file_grep(
    pattern: Annotated[str, Field(description="The regular expression pattern to search for in file contents")],
    mode: Annotated[Literal["files_with_matches", "content"], Field(description="Output mode")],
    path: Annotated[Optional[str], Field(description="Optional path to search in. Can be a directory or a specific file. If not specified, the current working directory will be used.")] = None,
    context: Annotated[Optional[int], Field(description="Lines of context before and after each match. Only used when under 'content' mode.", ge=1, le=10)] = None,
    n: Annotated[bool, Field(description="Whether to show line numbers. Only used when under 'content' mode.")] = True,
    i: Annotated[bool, Field(description='Case-insensitive search')] = False,
    type: Annotated[Optional[str], Field(description='File type filter (e.g. py, ts, rust).')] = None,
) -> str:
    """Search file contents with a regular expression. When searching a directory, VCS directories (.git, .svn, etc.) are skipped.

    Usage notes:
    - `path` may be a directory or a specific file. When it is a file, only that file is searched and .gitignore rules do not apply to it.
    - Two output modes: `files_with_matches` return the paths of files containing at least one match, `content` shows matching lines optionally with surrounding context and line numbers.
    - In `content` mode, each line is formatted as <path>:<lineno>:<text> for matches and <path>-<lineno>-<text> for context.
    - Paths are relative to the search root (the given directory, or the parent directory of the given file) and sorted by modification time (newest first).
    """
    target = util.resolve_root(path)
    if os.path.isfile(target):
        # 显式指定文件：只搜该文件，且不应用 .gitignore 过滤（与 ripgrep 行为一致）
        search_root = os.path.dirname(target) or os.getcwd()
        candidates = [target]
    elif os.path.isdir(target):
        search_root = target
        all_files = util.collect_files(search_root)
        ignore_patterns = util.load_gitignore(search_root)
        candidates = util.filter_by_gitignore(all_files, search_root, ignore_patterns)
    else:
        raise NotADirectoryError(target)

    try:
        regex = re.compile(pattern, re.IGNORECASE if i else 0)
    except re.error as e:
        raise ValueError(f'Invalid regex: {e}') from e

    candidates = util.filter_by_type(candidates, type)

    if mode == "files_with_matches":
        matched = []
        for f in candidates:
            with open(f, encoding='utf-8', errors='replace') as fp:
                text = fp.read()
            if text and any(regex.search(line) for line in text.splitlines()):
                matched.append(f)
        matched.sort(key=util.safe_mtime)
        rel = [os.path.relpath(f, search_root) for f in matched]
        res = f"Found {len(rel)} files:\n{'\n'.join(rel)}" if len(rel) else "No files found"

    elif mode == "content":
        all_lines = []
        for f in candidates:
            all_lines.extend(
                util.content_lines_for_file(f, search_root, regex, n, context)
            )
        res = '\n'.join(all_lines) if len(all_lines) else "No matches found"

    else:
        res = f"Unsupported grep mode: {mode}"

    return res


def file_list(path: Annotated[Optional[str], Field(description="Optional directory path to list. If not specified, the current working directory will be used.")] = None) -> str:
    """Lists files and directories in a given path."""
    root = util.resolve_root(path)
    if not os.path.isdir(root):
        raise NotADirectoryError(root)

    rows = []
    with os.scandir(root) as it:
        for entry in it:
            name = entry.name
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError:
                rows.append((name, "other", "??????????", "-", "-", None))
                continue

            link_target = None
            if stat.S_ISLNK(st.st_mode):
                kind = "symlink"
                link_target = os.readlink(entry.path)
                size_str = util.human_size(st.st_size)
            elif stat.S_ISDIR(st.st_mode):
                kind = "directory"
                size_str = "-"  # 目录 st_size 无意义
            elif stat.S_ISREG(st.st_mode):
                kind = "file"
                size_str = util.human_size(st.st_size)
            else:
                kind = "other"  # fifo/socket/device
                size_str = "-"

            rows.append((
                name,
                kind,
                util.mode_string(st.st_mode),
                size_str,
                datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
                link_target,
            ))

    rows.sort(key=lambda r: (r[1] != "directory", r[0]))

    if not rows:
        return f'Directory "{path}" is empty (or all entries were ignored).'

    size_width = max(max(len(r[3]) for r in rows), 1)
    mode_width = max(max(len(r[2]) for r in rows), 10)
    mtime_width = max(max(len(r[4]) for r in rows), 16)

    lines = [f'Contents of "{path}" ({len(rows)} entries):']
    for name, kind, mode_str, size_str, mtime, target in rows:
        suffix = "/" if kind == "directory" else "@" if kind == "symlink" else ""
        arrow = f" -> {target}" if target else ""
        lines.append(
            f"{mode_str:<{mode_width}}  "
            f"{size_str:>{size_width}}  "
            f"{mtime:<{mtime_width}}  "
            f"{name}{suffix}{arrow}"
        )

    return "\n".join(lines)

file_toolset = (file_read, file_write, file_edit, file_glob, file_grep, file_list)