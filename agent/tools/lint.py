"""
检查工具：lint —— 跑一遍「代码能不能通过检查」，把问题结构化地喂回给模型。

为什么要有它：
    商用 code agent 普遍会在「写完代码」之后、finish 之前，先跑一遍 linter / 类型检查 /
    语法检查，把报错逐条喂回模型去改——而不是等用户跑测试才发现一堆低级错误。
    `run_command` 也能跑，但它给的是原始 stdout；lint 额外做了：
      1) 零配置可用——不传 command 时，对 Python 文件用内置 compile() 做语法体检
         （纯标准库、零依赖、不联网，eval/演示环境也能跑）；
      2) 结构化解析：把 `文件:行: 信息` 这类行抽成清单，模型一眼看到底错在哪几行；
      3) 只读护栏：禁止 --fix / --write / -i 等会改写文件的选项，天然只"看"不"改"。

与 run_command 的边界：
    lint 只做「检查并回报」，不负责任意执行；要做删除/安装/运行测试等，仍用 run_command。
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ..errors import ToolError
from .base import ToolContext, ToolResult, tool_spec
from .search import _iter_files

__all__ = ["lint", "register"]

# 解析常见 linter 输出行：file:line:col: msg 或 file:line: msg
_ISSUE_RE = re.compile(
    r"^(?P<path>[^\s:][^:\n]*?):(?P<line>\d+)(?::(?P<col>\d+))?:\s*(?P<msg>.*)$"
)
# 会改写文件的选项——lint 只读，禁止
_WRITE_FLAGS = {"--fix", "--write", "-i", "--in-place", "--upgrade", "--output", "-o", "--autofix"}

_MAX_ISSUES = 200


def _parse_issues(text: str) -> List[str]:
    issues: List[str] = []
    for line in text.splitlines():
        m = _ISSUE_RE.match(line.strip())
        if m:
            issues.append(f"{m.group('path')}:{m.group('line')}"
                          + (f":{m.group('col')}" if m.group('col') else "")
                          + f": {m.group('msg').strip()}")
        if len(issues) >= _MAX_ISSUES:
            break
    return issues


def _auto_py_check(root: Path, ctx: ToolContext) -> List[str]:
    """零配置：对所有 .py 文件用内置 compile() 做语法体检，返回结构化问题清单。"""
    issues: List[str] = []
    files = [root] if root.is_file() else list(_iter_files(root, "*.py"))
    for f in files:
        try:
            raw = f.read_bytes()
        except OSError:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            issues.append(f"{ctx.guard.relpath(f)}:0: 文件不是合法 UTF-8 文本")
            continue
        try:
            compile(text, str(f), "exec")
        except SyntaxError as exc:
            line = exc.lineno or 0
            issues.append(f"{ctx.guard.relpath(f)}:{line}: {exc.msg}"
                          + (f" (near: {exc.text.strip() if exc.text else ''})" if exc.text else ""))
        if len(issues) >= _MAX_ISSUES:
            break
    return issues


@tool_spec(
    name="lint",
    description=(
        "跑一遍代码检查，把问题结构化地回报，便于写完代码后、finish 前先自查。"
        "不传 command 时：对 Python 文件用内置语法检查（compile）零配置体检；"
        "传 command 时：运行你给的检查命令（如 `ruff check .`、`mypy src/`），解析其输出。"
        "本工具只读，禁止 --fix 等改写选项；要跑测试/执行/安装请用 run_command。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "要检查的文件或目录，相对工作区，默认 '.'；不传 command 时仅对 .py 做语法检查",
                "default": ".",
            },
            "command": {
                "type": "string",
                "description": "可选：完整检查命令，如 `ruff check .`。不传则对 Python 文件做内置语法检查",
                "default": "",
            },
        },
        "required": [],
    },
    category="检查",
    when_not_to_use=(
        "单个明显的笔误直接 read_file 看那几行更快，lint 适合「一次性看清一个文件/目录里所有问题」。"
        "本工具只检查、不改写；想自动修就用 run_command 跑带 --fix 的检查器（后果自负）。"
        "跑单元测试请用 run_command，不要塞给 lint。"
    ),
)
def lint(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    command = (args.get("command") or "").strip()
    target = (args.get("target") or ".").strip() or "."
    root = ctx.resolve(target, must_exist=True)

    # ---- 模式一：用户显式给出检查命令 ----
    if command:
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            raise ToolError(f"命令解析失败：{exc}", tool="lint") from exc
        forbidden = [t for t in tokens if t in _WRITE_FLAGS or t.startswith("--output=")]
        if forbidden:
            raise ToolError(
                f"lint 是只读检查，禁止改写选项：{', '.join(forbidden)}",
                tool="lint",
                hint="本工具不负责自动修复；需要 --fix 请用 run_command 并自行承担后果。",
            )
        timeout = float(getattr(ctx.config, "command_timeout", 120))
        cwd = Path(str(ctx.workspace)).expanduser().resolve()
        try:
            proc = subprocess.run(tokens, cwd=str(cwd), capture_output=True, text=True,
                                  timeout=timeout, check=False)
        except FileNotFoundError:
            return ToolResult.failure(
                f"未找到命令可执行文件：{tokens[0]}",
                hint="确认检查器已安装；或省略 command 用内置 Python 语法检查。",
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(f"lint 执行超时（>{timeout:.0f}s）", hint="缩小 target 范围后重试。")
        out = (proc.stdout or "") + (proc.stderr or "")
        issues = _parse_issues(out)
        if not issues:
            if proc.returncode == 0:
                return ToolResult.success(
                    f"✓ {command} 通过，未发现检查问题。",
                    meta={"issues": 0, "command": command, "returncode": proc.returncode},
                )
            return ToolResult.success(
                f"{command} 退出码 {proc.returncode}，输出未解析出 file:line 格式的问题：\n"
                + out[:4000],
                meta={"issues": 0, "command": command, "returncode": proc.returncode},
            )
        head = "\n".join(issues[:_MAX_ISSUES])
        more = "" if len(issues) <= _MAX_ISSUES else f"\n    …（另有 {len(issues) - _MAX_ISSUES} 条）"
        return ToolResult.success(
            f"{command} 发现 {len(issues)} 个问题：\n{head}{more}",
            meta={"issues": len(issues), "command": command, "returncode": proc.returncode},
        )

    # ---- 模式二：零配置 Python 语法体检 ----
    issues = _auto_py_check(root, ctx)
    if not issues:
        return ToolResult.success(
            f"✓ {ctx.guard.relpath(root)} 下的 Python 文件语法检查全部通过。",
            meta={"issues": 0, "mode": "py-syntax"},
        )
    head = "\n".join(issues[:_MAX_ISSUES])
    more = "" if len(issues) <= _MAX_ISSUES else f"\n    …（另有 {len(issues) - _MAX_ISSUES} 条）"
    return ToolResult.success(
        f"Python 语法检查发现 {len(issues)} 个问题：\n{head}{more}",
        meta={"issues": len(issues), "mode": "py-syntax"},
    )


def register(registry) -> None:
    registry.register(lint)
