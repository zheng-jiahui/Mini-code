"""
命令执行工具：run_command。

要点：
    · 单进程、同步、可超时：超时后杀掉整个进程树并返回已产生的部分输出
      （很多测试卡住时，前几行输出恰恰是定位问题的关键）；
    · stdout 与 stderr 合并返回，但保留退出码 —— 模型需要 exit code 判断成败；
    · 工作目录默认 workspace，可用 cwd 指定子目录；
    · 危险命令在执行前由 CommandGuard 判定（deny / confirm / allow）；
    · 统一设置 PYTHONIOENCODING 等环境变量，避免 Windows 下中文输出乱码。
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Dict

from ..errors import SecurityError, ToolError
from .base import ToolContext, ToolResult, tool_spec

__all__ = ["run_command", "register"]


@tool_spec(
    name="run_command",
    description=(
        "在工作区内执行一条 shell 命令（同步，带超时）。"
        "用于运行测试、安装依赖、执行脚本、查看版本等。"
        "命令应当是非交互式的：例如用 `pytest -q` 而不是等待输入的 `pytest`。"
        "stdout 与 stderr 会合并返回，并附带退出码。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的命令，如 `pytest -q`、`python main.py`"},
            "cwd": {"type": "string", "description": "执行目录，相对工作区，默认工作区根目录", "default": "."},
            "timeout": {"type": "integer", "description": "超时秒数，默认取配置中的 command_timeout"},
        },
        "required": ["command"],
    },
    dangerous=True,
    category="执行",
)
def run_command(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    command = (args.get("command") or "").strip()
    if not command:
        raise ToolError("命令为空", tool="run_command")

    # 1) 安全审查
    decision, reason = ctx.session.get("command_guard").check(command) if ctx.session.get("command_guard") else ("ok", "")
    policy = getattr(ctx.config, "command_policy", "confirm")
    if decision == "deny" or (decision == "confirm" and policy == "deny"):
        raise SecurityError(f"命令被拒绝执行：{reason}", tool="run_command",
                            hint="该命令命中危险命令黑名单，请改用更安全的方式达成目标。")
    if decision == "confirm" and policy == "confirm":
        if not ctx.confirm(f"即将执行危险命令：\n      {command}\n      原因：{reason}\n      是否继续？"):
            raise SecurityError("用户拒绝了这条命令", tool="run_command",
                                hint="请换一种更安全的方式；若必须执行，请让用户手动运行。")

    # 2) 解析 cwd
    cwd_arg = args.get("cwd") or "."
    cwd = ctx.resolve(cwd_arg, must_exist=True)
    if not cwd.is_dir():
        raise ToolError(f"cwd 不是目录：{cwd_arg}", tool="run_command")
    timeout = int(args.get("timeout") or getattr(ctx.config, "command_timeout", 120))

    # 3) 构造环境
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env["NO_COLOR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    started_note = f"$ {command}    (cwd={ctx.guard.relpath(cwd)})"
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise ToolError(f"无法启动进程：{exc}", tool="run_command") from exc

    timed_out = False
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        try:
            out, _ = proc.communicate(timeout=5)
        except Exception:  # noqa: BLE001
            out = ""
    except KeyboardInterrupt:
        _kill_tree(proc)
        raise

    output = out or ""
    exit_code = proc.returncode if proc.returncode is not None else -1

    ctx.record_change("command", command[:120])

    parts = [started_note]
    if output.strip():
        parts.append(output.rstrip())
    else:
        parts.append("(无输出)")
    if timed_out:
        parts.append(f"[超时] 命令执行超过 {timeout}s 已被终止（以上为终止前的部分输出）。")
    parts.append(f"[exit code: {exit_code}]")

    ok = (exit_code == 0) and not timed_out
    result = ToolResult.success("\n".join(parts), meta={"exit_code": exit_code, "timed_out": timed_out, "cwd": str(cwd)})
    result.ok = ok
    if not ok:
        result.error = "\n".join(parts)
        result.output = ""
    return result


def _kill_tree(proc: subprocess.Popen) -> None:
    """尽可能杀掉进程及其子进程（shell=True 时子进程是 shell 的孩子）。"""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
            )
        else:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001 —— 尽力而为
        try:
            proc.kill()
        except Exception:
            pass


def register(registry) -> None:
    registry.register(run_command)
