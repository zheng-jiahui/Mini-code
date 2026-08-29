"""
终端渲染层：把主循环产生的事件流打印成人能看的东西。

刻意做得极简（不引入 rich/textual），只依赖标准库；
颜色在 Windows Terminal / macOS / Linux 下均可用，非 TTY 环境自动关闭。
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

__all__ = ["Console"]

_USE_COLOR = (
    sys.stdout.isatty()
    and os.environ.get("NO_COLOR") is None
    and os.environ.get("TERM") != "dumb"
)

if os.name == "nt" and _USE_COLOR:  # 让 Windows 控制台识别 ANSI 转义序列
    try:
        os.system("")
    except Exception:  # pragma: no cover
        pass


class _Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GREY = "\033[90m"


class Console:
    """极简终端 UI。

    对外只有三类方法：
        展示类：banner / step / thinking / tool_call / tool_result / final / warn / error / info
        交互类：confirm / ask
        结构类：rule / spinner（上下文管理器）
    """

    def __init__(self, verbose: bool = True, color: Optional[bool] = None):
        self.verbose = verbose
        self.color = _USE_COLOR if color is None else color
        self._start = time.time()

    # ---------------- 基础设施 ----------------
    def _c(self, text: str, code: str) -> str:
        return f"{code}{text}{_Style.RESET}" if self.color else text

    def echo(self, text: str = "", *, err: bool = False) -> None:
        stream = sys.stderr if err else sys.stdout
        print(text, file=stream, flush=True)

    def rule(self, title: str = "") -> None:
        width = 72
        if title:
            left = (width - len(title) - 2) // 2
            self.echo(self._c("─" * max(left, 1) + f" {title} " + "─" * max(width - left - len(title) - 2, 1), _Style.DIM))
        else:
            self.echo(self._c("─" * width, _Style.DIM))

    def elapsed(self) -> str:
        return f"{time.time() - self._start:.1f}s"

    # ---------------- 展示 ----------------
    def banner(self, summary: str, version: str = "") -> None:
        self.rule()
        self.echo(self._c(f"  MiniCode {version}".strip(), _Style.BOLD + _Style.CYAN))
        self.echo(self._c(f"  {summary}", _Style.GREY))
        self.echo(self._c("  输入 /help 查看命令，Ctrl-C 中断当前任务", _Style.GREY))
        self.rule()

    def step(self, n: int, max_steps: int, note: str = "") -> None:
        if not self.verbose:
            return
        suffix = f" · {note}" if note else ""
        self.echo(self._c(f"\n[{n}/{max_steps}]{suffix}", _Style.BOLD + _Style.BLUE))

    def thinking(self, text: str) -> None:
        if not self.verbose or not text.strip():
            return
        first = True
        for line in text.strip().splitlines():
            prefix = "  ︎思考 " if first else "       "
            self.echo(self._c(prefix, _Style.DIM) + self._c(line, _Style.GREY))
            first = False

    def tool_call(self, name: str, args: Dict[str, Any]) -> None:
        if not self.verbose:
            return
        brief = ", ".join(f"{k}={_short(str(v))}" for k, v in list(args.items())[:4])
        self.echo(self._c("  ▶ 调用 ", _Style.MAGENTA) + self._c(name, _Style.BOLD) + self._c(f"({brief})", _Style.GREY))

    def tool_result(self, name: str, ok: bool, text: str, meta: str = "") -> None:
        if not self.verbose:
            return
        mark = self._c("✔", _Style.GREEN) if ok else self._c("✘", _Style.RED)
        head = f"  {mark} {name}"
        if meta:
            head += self._c(f" · {meta}", _Style.GREY)
        self.echo(head)
        body = (text or "").rstrip()
        if body:
            for line in body.splitlines()[:40]:
                self.echo(self._c("      " + line, _Style.GREY if ok else _Style.YELLOW))
            if len(body.splitlines()) > 40:
                self.echo(self._c("      ...（回执已折叠，模型仍能看到完整内容）", _Style.GREY))

    def final(self, text: str) -> None:
        self.rule("结果")
        self.echo(text.strip())
        self.rule()

    def info(self, text: str) -> None:
        self.echo(self._c(f"  ℹ {text}", _Style.CYAN))

    def warn(self, text: str) -> None:
        self.echo(self._c(f"  ⚠ {text}", _Style.YELLOW), err=True)

    def error(self, text: str) -> None:
        self.echo(self._c(f"  ✘ {text}", _Style.RED), err=True)

    def stats(self, data: Dict[str, Any]) -> None:
        parts = [f"{k}={v}" for k, v in data.items()]
        self.echo(self._c("  " + " · ".join(parts), _Style.GREY))

    # ---------------- 流式输出 ----------------
    def stream_begin(self, label: str = "") -> None:
        """准备一段流式输出。与 spinner 二选一——spinner 靠 `\\r` 覆盖同一行，
        会和流式正文互相擦除，所以同一轮只能用其中一个。"""
        self._streaming_label = label
        self._streaming_any = False
        self._streaming_started = False

    def stream(self, text: str) -> None:
        """写入一个增量分片（不换行，立即 flush——不然就不叫"边生成边看"了）。

        标签延迟到第一片**非空白**内容时才打印：模型在"只吐工具调用"的轮次里
        往往先发几个纯空白分片，一上来就打标签会让界面刷出一串空行。
        """
        if not text:
            return
        if not self._streaming_started:
            if not text.strip():
                return
            if self._streaming_label:
                sys.stdout.write(self._c(f"  {self._streaming_label} ▸ ", _Style.GREY))
            self._streaming_started = True
        self._streaming_any = True
        sys.stdout.write(text)
        sys.stdout.flush()

    def stream_end(self) -> None:
        """结束流式输出：补一个换行，避免后续回执和正文挤在同一行。

        若这一轮一个字都没流出来（模型只给了工具调用），补一行提示——
        否则界面上会毫无反馈，反而不如原来的 spinner。
        """
        if getattr(self, "_streaming_any", False):
            sys.stdout.write("\n")
            sys.stdout.flush()
        elif getattr(self, "_streaming_label", ""):
            self.echo(self._c(f"  … {self._streaming_label}", _Style.GREY))
        self._streaming_any = False
        self._streaming_started = False

    # ---------------- 交互 ----------------
    def confirm(self, question: str, default: bool = False) -> bool:
        """向用户确认。非交互环境下返回 default。"""
        hint = "[Y/n]" if default else "[y/N]"
        try:
            self.echo(self._c(f"  ⚠ {question} {hint} ", _Style.YELLOW), err=False)
            ans = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            self.echo("")
            return default
        if not ans:
            return default
        return ans in ("y", "yes", "是", "1")

    def ask(self, question: str) -> str:
        try:
            return input(self._c(f"  ? {question} ", _Style.CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    # ---------------- 上下文管理器 ----------------
    @contextmanager
    def spinner(self, text: str = "思考中") -> Iterator[Callable[[str], None]]:
        """极简 spinner：非交互环境下退化为一行提示。

        Yields:
            可用于更新提示文案的回调。
        """
        if not self.verbose or not sys.stdout.isatty():
            self.echo(self._c(f"  … {text}", _Style.GREY))
            yield lambda _t: None
            return

        import itertools
        import threading

        stop = threading.Event()
        state = {"text": text}

        def _run() -> None:
            for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
                if stop.is_set():
                    break
                sys.stdout.write(f"\r  {self._c(ch, _Style.CYAN)} {state['text']}")
                sys.stdout.flush()
                if stop.wait(0.08):
                    break
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        try:
            yield lambda new_text: state.__setitem__("text", new_text)
        finally:
            stop.set()
            t.join(timeout=0.5)


def _short(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
