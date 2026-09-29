"""
terminal_ui.py — 基于 prompt_toolkit 的 CLI Agent 分屏终端 UI。

UI 在后台线程中运行，Agent 主流程可并发执行：
- 输出区：自动滚动、按消息类型着色、最多保留 MAX_LINES 行；
- 输入区：底部单行输入，Enter 提交，↑/↓ 遍历历史；
- 线程安全：通过 input_queue / display_queue 与主流程通信。
"""

from __future__ import annotations

import asyncio
import datetime
import os
import queue
import threading

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout, Window, HSplit, VSplit
from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.output.win32 import NoConsoleScreenBufferError
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

# 输出缓冲区最多保留的行数，超出时丢弃最旧的行
MAX_LINES = 1000
# 底部输入行的提示符
INPUT_PROMPT = "> "

# 消息类型着色方案（参考设计文档 3.1 节）
STYLE = Style.from_dict({
    "output-default": "ansidefault",        # Agent 响应：白色
    "output-tool-call": "ansicyan",         # 工具调用日志：青色
    "output-tool-result": "ansigreen",      # 工具执行结果：绿色
    "output-error": "ansired",              # 错误信息：红色
    "output-dim": "ansigray",               # 次要信息：灰色
    "output-approval": "ansiyellow",        # 系统提示/审批：黄色
    "output-user-input": "#6CBAFF bold",     # 输出区回显的用户输入：蓝色加粗
    "input-prompt": "#6CBAFF",               # 终端输入区提示符：蓝色
    "separator": "#444444",                  # 分隔线：深灰
})


def _detect_msg_type(text: str) -> str:
    """根据单行文本内容识别消息类型，用于 Lexer 着色"""
    t = text.lower()
    if "error" in t or "❌" in t:
        return "error"
    if "[tool]" in t:
        return "tool-call"
    if "[result]" in t:
        return "tool-result"
    if "[user]" in t:
        return "user-input"
    if "[approval]" in t:
        return "approval"
    return "default"


class _OutputLexer(Lexer):
    """输出区 Lexer：按行内容分类着色（不使用 ANSI 转义序列）"""

    def lex_document(self, document):
        """返回 prompt_toolkit 所需的（样式，行号）列表"""
        lines = document.lines

        def get_line(lineno: int):
            if lineno < 0 or lineno >= len(lines):
                return []
            line = lines[lineno]
            # 按规则整行着色
            return [(f"class:output-{_detect_msg_type(line)}", line)]

        return get_line


class SessionLogger:
    """会话日志记录器：将所有输出追加到带时间戳的日志文件（线程安全）"""

    def __init__(self, log_dir: str = "logs"):
        """创建日志目录与本次会话的日志文件 logs/session_YYYYMMDD_HHMMSS.log"""
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"session_{ts}.log")
        self._file = open(self.path, "w", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, text: str) -> None:
        """写入一行文本并立即 flush"""
        if not text:
            return
        with self._lock:
            self._file.write(text + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None


class TerminalUI:
    """分屏终端 UI：上半部分为输出区，下半部分为输入行"""

    def __init__(self, logger: SessionLogger | None = None):
        """初始化队列、缓冲区、事件循环引用等内部状态"""
        self._logger = logger
        self.input_queue: queue.Queue = queue.Queue()  # 用户输入队列，供主流程 ui.get_input() 消费
        self.display_queue: queue.Queue = queue.Queue()  # 待显示内容队列，由 _display_drain() 批量消费
        self.error: str | None = None

        self._lines: list[str] = []  # 输出区文本缓冲（list[str]，按行存储）
        self._exit = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._app: Application | None = None
        self._app_ready = threading.Event()
        self._auto_scroll = True
        self._output_area: TextArea | None = None
        self._pending_approval: asyncio.Future | None = None  # 审批模式下挂起的 Future

    def get_input(self, timeout: float | None = None) -> str | None:
        """从输入队列取一条用户输入（已 strip）；超时或无数据返回 None"""
        try:
            _input = self.input_queue.get(timeout=timeout)
            if _input:
                _input = _input.strip()
            return _input
        except queue.Empty:
            return None

    def put_display(self, text: str) -> None:
        """把一条待显示文本加入显示队列，并同步写入会话日志，队列满时静默丢弃"""
        if not text:
            return
        if self._logger is not None:
            self._logger.write(text)
        try:
            self.display_queue.put_nowait(text)
        except queue.Full:
            pass

    def request_exit(self) -> None:
        """请求退出 UI：设置退出标志并调度 _do_exit() 关闭 Application。"""
        self._exit.set()
        loop = self._loop
        if loop and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._do_exit)
            except RuntimeError:
                pass

    def wait_ready(self, timeout: float) -> bool:
        """阻塞等待 UI 就绪（Application 已接管终端）"""
        return self._app_ready.wait(timeout)

    def get_error(self) -> str | None:
        return self.error

    def is_alive(self) -> bool:
        return not self._exit.is_set()

    def _do_exit(self) -> None:
        """在 UI 事件循环线程内安全关闭 Application"""
        if self._app is not None:
            try:
                self._app.exit()
            except Exception:
                pass

    def _append_output(self, text: str) -> None:
        """把多行文本追加到 _lines 缓冲；超出 MAX_LINES 时丢弃最旧的行"""
        self._lines.extend(text.split("\n"))
        if len(self._lines) > MAX_LINES:
            self._lines = self._lines[-MAX_LINES:]

    @staticmethod
    def _line_start(text: str, row: int) -> int:
        """计算第 row 行首字符在 text 中的字符偏移"""
        pos = 0
        for _ in range(row):
            nl = text.find("\n", pos)
            if nl < 0:
                return len(text)
            pos = nl + 1
        return pos

    def _refresh_output(self) -> None:
        """把 _lines 同步到 TextArea，并按 _auto_scroll 决定光标位置"""
        area = self._output_area
        if area is None:
            return

        text = "\n".join(self._lines)
        prev_row = area.buffer.document.cursor_position_row  # 必须在改 text 前读

        area.text = text  # setter 会把 cursor 重置到 0

        if self._auto_scroll:
            # 光标 = 最后一行的行首（列 0）→ 垂直贴底，水平不动
            last_row = text.count("\n")
            area.buffer.cursor_position = self._line_start(text, last_row)
        else:
            # 用户手动滚动过：保持原行，但同样归到该行行首（列 0）
            row = max(0, min(prev_row, text.count("\n")))
            area.buffer.cursor_position = self._line_start(text, row)

        if self._app is not None:
            self._app.invalidate()

    def _scroll(self, delta: int) -> None:
        """按行数滚动输出区（鼠标滚轮触发），并关闭自动滚动"""
        self._auto_scroll = False
        area = self._output_area
        if area is None:
            return

        text = area.text
        if not text:
            return

        n_lines = text.count("\n") + 1
        cur_row = area.buffer.document.cursor_position_row
        if cur_row < 0 or cur_row >= n_lines:
            cur_row = n_lines - 1

        target = max(0, min(cur_row + delta, n_lines - 1))
        if target == cur_row:
            return

        area.buffer.cursor_position = self._line_start(text, target)
        if self._app is not None:
            self._app.invalidate()

    def _scroll_to_bottom(self) -> None:
        """重新开启自动滚动并把光标移到底部（最后一行的行首）"""
        self._auto_scroll = True
        area = self._output_area
        if area is None:
            return
        text = area.text
        last_row = text.count("\n")
        area.buffer.cursor_position = self._line_start(text, last_row)
        if self._app is not None:
            self._app.invalidate()

    def _build_layout(self) -> Layout:
        """构建分屏布局：输出 TextArea + 分隔线 + 底部输入行（VSplit）"""
        self._output_area = TextArea(
            text="",
            read_only=True,
            scrollbar=True,
            wrap_lines=False,
            focusable=False,
            focus_on_click=False,
            lexer=_OutputLexer(),
        )
        input_buffer = Buffer(name="input", multiline=False, enable_history_search=True)
        prompt = Window(
            content=FormattedTextControl(
                text=lambda: [("class:input-prompt", INPUT_PROMPT)],
                focusable=False,
                show_cursor=False,
            ),
            width=len(INPUT_PROMPT),
            dont_extend_width=True,
            dont_extend_height=True,
        )
        input_row = VSplit([
            prompt,
            Window(BufferControl(buffer=input_buffer), wrap_lines=False),
        ])
        separator = Window(height=1, char="─", style="class:separator")
        return Layout(HSplit([self._output_area, separator, input_row]))

    def _make_key_bindings(self) -> KeyBindings:
        """注册全局键绑定：Enter 提交、滚轮上下滚动"""
        kb = KeyBindings()

        @kb.add(Keys.Enter)
        def _submit(event):
            buf = event.app.current_buffer
            text = buf.text
            if not text.strip():
                return

            # 回显到输出区
            echo = f"[USER] {text}"
            if self._logger is not None:
                self._logger.write(echo)
            self._append_output(echo)
            self._refresh_output()

            fut = self._pending_approval
            if fut is not None and not fut.done():
                # 有待处理审批时，不进普通输入队列
                fut.set_result(text)
            else:
                self.input_queue.put(text)

            buf.reset()
            self._scroll_to_bottom()

        @kb.add(Keys.ScrollUp, eager=True)
        def _wheel_up(event):
            self._scroll(-3)

        @kb.add(Keys.ScrollDown, eager=True)
        def _wheel_down(event):
            self._scroll(+3)

        return kb

    async def _display_drain(self) -> None:
        """显示队列消费协程：按积压量分档批量取文本，并周期性刷新输出区。"""
        def _compute_batch(qsize: int) -> int:
            if qsize <= 50:
                return 1
            if qsize <= 200:
                return 2
            return 4
        try:
            while not self._exit.is_set():
                # 按积压量分档决定这一 tick 处理多少条
                batch = _compute_batch(self.display_queue.qsize())
                processed = 0
                for _ in range(batch):
                    try:
                        text = self.display_queue.get_nowait()
                    except queue.Empty:
                        break
                    self._append_output(text)
                    processed += 1
                if processed:
                    self._refresh_output()
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            pass

    async def _run_app(self) -> None:
        """创建并运行 prompt_toolkit Application；就绪后置位 _app_ready"""
        app = Application(
            layout=self._build_layout(),
            key_bindings=self._make_key_bindings(),
            style=STYLE,
            full_screen=True,
            refresh_interval=0.05,
            mouse_support=True,
        )
        self._app = app
        run_task = asyncio.ensure_future(app.run_async())
        await asyncio.sleep(0.1)  # 给 run_async 一点时间接管终端
        if not run_task.done():
            self._app_ready.set()
        await run_task
        if not self._exit.is_set() and self.error is None:
            self.error = "UI 提前退出（Application.run_async 返回但未收到退出请求）"

    async def _main(self) -> None:
        """UI 事件循环主协程：并发运行 _run_app 与 _display_drain，退出时收尾"""
        drain = asyncio.ensure_future(self._display_drain())
        try:
            await self._run_app()
        finally:
            drain.cancel()
            await asyncio.gather(drain, return_exceptions=True)

    async def request_approval(self, prompt: str) -> str:
        """在 UI 输出区显示 prompt，等待用户下一次回车输入并返回该文本"""
        if not self.is_alive() or self._loop is None or self._loop.is_closed():
            # UI 不可用：回退到内建 input()（headless / 测试场景）
            return await asyncio.to_thread(input, prompt + " ")

        # 在 UI 事件循环里挂上 future 并显示提示；返回一个 concurrent.futures.Future
        cfut = asyncio.run_coroutine_threadsafe(self._arm_approval(prompt), self._loop)
        # 桥接回当前 agent 的事件循环
        return await asyncio.wrap_future(cfut)

    async def _arm_approval(self, prompt: str) -> str:
        """在 UI 事件循环内挂起 _pending_approval，并把提示写入输出区"""
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending_approval = fut
        try:
            self.put_display(f"[Approval] {prompt}")
            return await fut
        finally:
            self._pending_approval = None

    def run(self) -> None:
        """在调用线程中运行 UI 事件循环（通常是一个后台线程）"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._main())
        except NoConsoleScreenBufferError:
            self.error = (
                "终端不支持 UI 模式（NoConsoleScreenBufferError）。\n"
                "请在 cmd.exe 或 Windows Terminal 中运行，或使用 winpty。\n"
            )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.error = f"UI 线程异常: {type(e).__name__}: {e}"
        finally:
            self._exit.set()
            if self._logger is not None:
                self._logger.close()
            try:
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            asyncio.set_event_loop(None)
            loop.close()


def launch_ui() -> TerminalUI:
    """在后台线程启动分屏终端 UI，并返回 TerminalUI 实例。

    同时把实例注册为模块级单例，供 get_ui() / display() 使用。
    - ui.get_input()       获取用户输入
    - ui.put_display(text) 输出内容
    - ui.request_exit()    请求退出
    """
    global _instance  # 把 TerminalUI 注册为模块单例
    logger = SessionLogger()
    ui = TerminalUI(logger=logger)

    with _instance_lock:
        _instance = ui

    threading.Thread(target=ui.run, daemon=True).start()
    ui.wait_ready(1.5)
    return ui


# ── 模块级单例 & 便捷 API
_instance: TerminalUI | None = None
_instance_lock = threading.Lock()

def get_ui() -> TerminalUI | None:
    """获取当前 UI 单例；未启动时为 None。"""
    with _instance_lock:
        return _instance

def display(*args, sep: str = "\n") -> None:
    """把文本贴到 UI 输出区（每条占一行），用法类似 print()。

    - UI 已启动：经 put_display 入队（同时写 session 日志）。
    - UI 未启动：回退到内置 print()，脚本 / headless 模式无需改调用点。
    """
    text = sep.join(str(a) for a in args)
    ui = _instance
    if ui is None or not ui.is_alive():
        print(text)
    else:
        ui.put_display(text)
