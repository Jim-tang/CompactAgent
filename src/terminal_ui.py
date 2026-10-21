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
import shutil
import threading

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout, Window, HSplit, VSplit
from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.output.win32 import NoConsoleScreenBufferError
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import TextArea

# 输出缓冲区最多保留的行数，超出时丢弃最旧的行
MAX_LINES = 1000
# 底部输入行的提示符
INPUT_PROMPT = "> "
# 流式结束标记：由 end_stream() 投递到显示队列，保证收尾按入队顺序执行
_STREAM_END = object()

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
    if "error:" in t:
        return "error"
    if "[tool]" in t:
        return "tool-call"
    if "[result]" in t:
        return "tool-result"
    if "[user]" in t:
        return "user-input"
    if "[approval]" in t:
        return "approval"
    if "[reasoning]" in t or "│ " in t:
        return "dim"
    return "default"


class _OutputLexer(Lexer):
    """输出区 Lexer：按行内容分类着色（不使用 ANSI 转义序列）"""

    def __init__(self, line_types: list[str]):
        # 与 TerminalUI._lines 一一对应。软折行产生的续行没有行首标识，
        # 但会沿用所属消息首行的类型，避免只有第一行有颜色。
        self._line_types = line_types

    def lex_document(self, document):
        """返回 prompt_toolkit 所需的（样式，行号）列表"""
        lines = document.lines

        def get_line(lineno: int):
            if lineno < 0 or lineno >= len(lines):
                return []
            line = lines[lineno]
            if lineno < len(self._line_types):
                msg_type = self._line_types[lineno]
            else:
                # 兜底：类型表未同步时仍按内容识别
                msg_type = _detect_msg_type(line)
            return [(f"class:output-{msg_type}", line)]

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
        self._line_types: list[str] = []  # 与 _lines 一一对应的消息类型（着色用）
        self._exit = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._app: Application | None = None
        self._app_ready = threading.Event()
        self._auto_scroll = True
        self._output_area: TextArea | None = None
        self._pending_approval: asyncio.Future | None = None  # 审批模式下挂起的 Future

        self._is_streaming = False  # 流式输出状态标志（写线程用）
        self._stream_line_idx: int | None = None  # 当前流式行在 _lines 中的下标（UI 线程用）
        self._stream_prefix: str | None = None  # 当前流式行的类型前缀（reasoning / content）
        self._stream_col: int = 0  # 当前流式行已占用的显示列数（UI 线程用）
        self._stream_log: list[str] = []  # 当前流式输出累积的文本（UI 线程用）
        self._stream_done = threading.Event()  # UI 线程处理完结束标记后置位
        self._stream_done.set()

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

    def put_stream(self, text: str, prefix: str) -> None:
        """流式输出文本到 UI 输出区

        首次调用时： is_first=True，创建新行并插入前缀
        后续调用时： is_first=False，把 chunk 内容追加到流式行末尾
        分片与结束标记都走同一条队列，由 UI 线程按入队顺序消费。
        """
        is_first = not self._is_streaming
        self._is_streaming = True
        if is_first:
            self._stream_done.clear()
        try:
            self.display_queue.put_nowait((is_first, text, prefix))
        except queue.Full:
            pass

    def end_stream(self):
        """结束流式输出：投递结束标记，由 UI 线程按序收尾并落盘日志。

        不能在这里直接重置 UI 线程的流式状态或清空日志缓冲，否则队列中尚未
        消费的分片会被当成普通文本处理，甚至丢失思考内容的尾部。
        """
        self._is_streaming = False
        try:
            self.display_queue.put_nowait(_STREAM_END)
        except queue.Full:
            # 队列已满（UI 线程未及时消费）：只能在当前线程直接收尾
            self._finish_stream()
            return
        # 等待 UI 线程真正消费完本轮所有分片并落盘日志后再返回：
        # 否则调用方（尤其是跑完即退出的测试脚本）可能提前结束进程，
        # 队列里剩下的思考内容永远来不及显示
        self._stream_done.wait(2.0)

    def _finish_stream(self) -> None:
        """在 UI 线程内完成一次流式输出：写日志并重置流式行状态。"""
        try:
            if self._stream_log:
                if self._logger is not None:
                    try:
                        self._logger.write("".join(self._stream_log))
                    except Exception:
                        pass
                self._stream_log.clear()
            self._stream_line_idx = None
            self._stream_prefix = None
            self._stream_col = 0
        finally:
            self._stream_done.set()

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

    def _add_line(self, text: str, msg_type: str) -> None:
        """追加一行文本，并记录它的着色类型（续行由调用方传入首行类型）"""
        self._lines.append(text)
        self._line_types.append(msg_type)

    def _append_stream(self, is_first: bool, text: str, prefix: str) -> None:
        """把流式分片追加到 _lines。

        判断是否需要新起一行只依赖分片自身携带的 is_first/prefix 以及
        UI 线程维护的流式行状态，不再读取写线程的 _is_streaming，避免
        end_stream() 提前重置标志后把队列中的分片误判为普通文本。
        文本按当前输出区宽度先行软折行，保证 _lines 里每一项都只占一屏行，
        避免出现超长“逻辑行”导致滚轮无法回退。
        """
        if not text:
            return

        # 统一换行符、展开制表符：get_cwidth("\t") 为 0，若按 0 列计算，
        # 终端把 tab 展开后实际宽度会超出预算，软折行就失效了
        text = text.replace("\r\n", "\n").replace("\r", "").replace("\t", "    ")

        new_line = (
            is_first
            or self._stream_line_idx is None
            or prefix != self._stream_prefix
        )
        body_prefix = "│ " if prefix == "[reasoning]" else ""
        line_type = _detect_msg_type(prefix) if prefix else "default"
        if new_line:
            if prefix == "[reasoning]":
                self._add_line(prefix, line_type)
            self._add_line(body_prefix, line_type)
            self._stream_line_idx = len(self._lines) - 1
            self._stream_prefix = prefix
            self._stream_col = len(body_prefix)
            # 日志里也换行，避免 reasoning 与 content 首尾粘连
            if self._stream_log and not self._stream_log[-1].endswith("\n"):
                self._stream_log.append("\n")

        width = self._content_width()
        idx = self._stream_line_idx
        for char in text:
            if char == "\n":
                self._add_line(body_prefix, line_type)
                idx = len(self._lines) - 1
                self._stream_col = len(body_prefix)
                continue
            char_width = get_cwidth(char)
            if char_width < 0:
                char_width = 0
            if self._stream_col + char_width > width:
                self._add_line(body_prefix, line_type)
                idx = len(self._lines) - 1
                self._stream_col = len(body_prefix)
            self._lines[idx] += char
            self._stream_col += char_width

        self._stream_line_idx = idx
        self._stream_log.append(text)

    def _content_width(self) -> int:
        """输出区可用于显示文本的列数。

        右侧滚动条占 1 列；BufferControl 还会给每行末尾补一个光标空格，
        因此再留 1 列，保证折行后每行在渲染时高度都是 1。
        """
        columns = 0
        if self._app is not None:
            try:
                columns = self._app.output.get_size().columns
            except Exception:
                columns = 0
        if columns <= 0:
            try:
                columns = shutil.get_terminal_size((80, 24)).columns
            except Exception:
                columns = 80
        return max(10, columns - 2)

    def _wrap_plain_text(self, text: str) -> list[tuple[str, str]]:
        """普通输出按显示宽度折行，返回 (行文本, 消息类型)。

        消息类型在折行前按原始行识别，再套用到该行的所有折行片段，
        这样续行不会因为没有行首标识而退回默认样式。
        """
        width = self._content_width()
        text = text.replace("\r\n", "\n").replace("\r", "").replace("\t", "    ")
        result: list[tuple[str, str]] = []
        for raw in text.split("\n"):
            msg_type = _detect_msg_type(raw)
            if not raw:
                result.append(("", msg_type))
                continue
            line = ""
            col = 0
            for char in raw:
                char_width = get_cwidth(char)
                if char_width < 0:
                    char_width = 0
                if col + char_width > width and line:
                    result.append((line, msg_type))
                    line = ""
                    col = 0
                line += char
                col += char_width
            result.append((line, msg_type))
        return result

    def _trim_lines(self) -> None:
        """超出 MAX_LINES 时丢弃最旧的行，并同步修正流式行下标"""
        overflow = len(self._lines) - MAX_LINES
        if overflow <= 0:
            return
        del self._lines[:overflow]
        del self._line_types[:overflow]
        if self._stream_line_idx is not None:
            self._stream_line_idx -= overflow
            if self._stream_line_idx < 0:
                self._stream_line_idx = None
                self._stream_prefix = None

    def _append_output(self, item: str | tuple[bool, str, str] | object) -> None:
        """把文本、流式分片或流结束标记追加到 _lines 缓冲"""
        if item is _STREAM_END:
            self._finish_stream()
            return

        if isinstance(item, tuple):
            self._append_stream(*item)
        else:
            # 非流式文本会打断当前流式行，下一分片应另起一行
            self._stream_line_idx = None
            self._stream_prefix = None
            for line, msg_type in self._wrap_plain_text(item):
                self._add_line(line, msg_type)

        self._trim_lines()

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
            # 光标置于文本末尾：文本已按宽度软折行，这里让窗口始终滚到
            # 最后一行，保证流式内容和新输出都保持可见
            area.buffer.cursor_position = len(text)
            area.window.vertical_scroll_2 = 0
        else:
            # 用户手动滚动过：保持原行，但同样归到该行行首（列 0）
            row = max(0, min(prev_row, text.count("\n")))
            area.buffer.cursor_position = self._line_start(text, row)

        if self._app is not None:
            self._app.invalidate()

    def _handle_mouse_wheel(self, mouse_event):
        """接管输出区的鼠标滚轮，返回 None 表示事件已处理。

        prompt_toolkit 默认的 Window._scroll_up 在 wrap_lines=True 时按逻辑行
        滚动，且滚出的位置会在渲染阶段被重新计算抵消（长行上几乎滚不动），
        因此统一交给自己的 _scroll 处理。
        """
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self._scroll(-3)
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self._scroll(+3)
            return None
        return NotImplemented

    def _scroll(self, delta: int) -> None:
        """按逻辑行滚动输出区（鼠标滚轮触发）"""
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

        if delta > 0 and target >= n_lines - 1:
            # 向下滚到末尾：重新开启自动跟随，方便继续看新输出
            self._auto_scroll = True
            area.buffer.cursor_position = len(text)
        else:
            self._auto_scroll = False
            if target == cur_row:
                return
            area.buffer.cursor_position = self._line_start(text, target)
            # 换行模式下渲染会用 cursor 推导行内偏移，归零才能保证
            # 每次滚动都从目标逻辑行的行首开始显示
            area.window.vertical_scroll_2 = 0

        if self._app is not None:
            self._app.invalidate()

    def _scroll_to_bottom(self) -> None:
        """重新开启自动滚动并把光标移到文本末尾（贴底）"""
        self._auto_scroll = True
        area = self._output_area
        if area is None:
            return
        area.buffer.cursor_position = len(area.text)
        area.window.vertical_scroll_2 = 0
        if self._app is not None:
            self._app.invalidate()

    def _build_layout(self) -> Layout:
        """构建分屏布局：输出 TextArea + 分隔线 + 底部输入行（VSplit）"""
        self._output_area = TextArea(
            text="",
            read_only=True,
            scrollbar=True,
            # 思考内容通常远超一屏宽度；关闭换行只会把后半段裁到屏幕右侧，
            # 用户又无法横向滚动，看起来就是“思考内容显示不全”
            wrap_lines=True,
            focusable=False,
            focus_on_click=False,
            lexer=_OutputLexer(self._line_types),
        )
        # 覆盖 TextArea 内部 Window 的默认滚轮处理，否则 wrap_lines=True
        # 时长行上的回滚会被 prompt_toolkit 自己的滚动重算抵消
        self._output_area.window._mouse_handler = self._handle_mouse_wheel

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
        input_window = Window(BufferControl(buffer=input_buffer), wrap_lines=False)
        separator = Window(height=1, char="─", style="class:separator")
        # 光标停在输入行/分隔线/提示符上时，滚轮也应滚动输出区
        for window in (prompt, input_window, separator):
            window._mouse_handler = self._handle_mouse_wheel
        input_row = VSplit([
            prompt,
            input_window,
        ])
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
            # 每轮把积压尽量清空（单轮上限 256），消费速率必须明显高于
            # token 到达速率，否则流结束后队列里仍堆着大量思考分片来不及绘制
            return max(1, min(qsize, 256))
        try:
            while not self._exit.is_set():
                batch = _compute_batch(self.display_queue.qsize())
                processed = 0
                for _ in range(batch):
                    try:
                        text = self.display_queue.get_nowait()
                    except queue.Empty:
                        break
                    processed += 1
                    try:
                        self._append_output(text)
                    except Exception:
                        # 单条输出异常不应终止整个消费协程，否则后续流式内容
                        # 将永远不再刷新，表现为输出“卡住/显示不全”
                        continue
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

def display_stream(text: str, prefix: str = "") -> None:
    """流式输出文本到 UI 输出区。"""
    ui = _instance
    if ui is None or not ui.is_alive():
        print(text, end="", flush=True)
    else:
        ui.put_stream(text, prefix)

def end_stream() -> None:
    """结束流式输出"""
    ui = _instance
    if ui is None or not ui.is_alive():
        # 无 UI 时 display_stream 用 print(end="") 直出，这里补一个换行，
        # 避免后续 display() 的内容粘在最后一段思考文字后面
        print()
        return
    else:
        ui.end_stream()
