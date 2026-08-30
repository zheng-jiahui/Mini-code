"""
Git 工具：git —— 让 agent 能看清仓库状态、比对改动、生成提交信息参考。

安全边界（与考核要求一致）：
    本工具**只放行只读类 git 子命令 + 安全的 add（暂存）**，
    明确拒绝 push / pull / fetch / reset / checkout / clean / commit / merge /
    rebase / revert 等会改写历史或触碰远端、或破坏用户仓库的操作。
    任何试图改写历史或推送的行为都被拦下，并提示用户自行在终端执行。

为什么要做这个工具：
    商用编程智能体普遍能「看 diff / 看 log / 看 status」。
    但本项目刻意不把 `run_command` 当万能后门——git 这类高危操作单独立一个
    白名单工具，既能让模型用，又从机制上堵死了"误执行 git reset --hard"的风险。
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ..errors import ToolError
from .base import ToolContext, ToolResult, tool_spec

__all__ = ["git", "register"]

# 放行：只读 + 安全的暂存（add）。其余一律拒绝。
_ALLOWED_SUBCOMMANDS = {
    "status", "diff", "log", "show", "branch", "tag", "remote",
    "rev-parse", "blame", "ls-files", "shortlog", "stash", "add",
}
# 拒绝：会改写历史 / 触碰远端 / 破坏工作区的子命令（作为白名单之外的兜底拦截）
_DENY_SUBCOMMANDS = {
    "push", "pull", "fetch", "clone", "reset", "checkout", "clean", "commit",
    "merge", "rebase", "revert", "rm", "mv", "am", "apply", "cherry-pick",
    "submodule", "init", "config", "worktree", "gc",
}


# ----------------------------------------------------------------------------
# 辅助（必须放在 @tool_spec 之前）
# ----------------------------------------------------------------------------
def _guard(args_str: str) -> List[str]:
    """解析并校验 git 参数，返回 tokens；不合法则抛 ToolError。"""
    if not args_str.strip():
        raise ToolError("请提供 git 子命令，如 status / diff / log", tool="git")
    try:
        tokens = shlex.split(args_str)
    except ValueError as exc:
        raise ToolError(f"参数解析失败：{exc}", tool="git") from exc
    if not tokens:
        raise ToolError("请提供 git 子命令", tool="git")

    sub = tokens[0]
    # 兜底：白名单之外 + 黑名单内的都拒绝
    if sub in _DENY_SUBCOMMANDS:
        raise ToolError(
            f"git {sub} 被禁止（会改写历史 / 触碰远端 / 破坏仓库）",
            tool="git",
            hint="本工具只做只读查看与暂存；提交与推送请由你在终端自行执行。",
        )
    if sub not in _ALLOWED_SUBCOMMANDS:
        raise ToolError(
            f"不支持的 git 子命令 `{sub}`",
            tool="git",
            hint="可用：" + ", ".join(sorted(_ALLOWED_SUBCOMMANDS)),
        )
    # stash 只允许 list，避免误执行 stash drop/pop
    if sub == "stash" and (len(tokens) < 2 or tokens[1] != "list"):
        raise ToolError("git stash 只允许 `list`", tool="git")
    return tokens


# ----------------------------------------------------------------------------
# git
# ----------------------------------------------------------------------------
@tool_spec(
    name="git",
    description=(
        "在沙箱内执行**安全的** git 命令：查看仓库状态 / 比对改动 / 看提交历史 / 暂存文件。"
        "例如 git status、git diff、git log -n、git show、git branch、git add <文件>。"
        "push / pull / reset / checkout / commit 等会改写历史或触碰远端的操作被禁止。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "args": {
                "type": "string",
                "description": "git 后的子命令与参数，如 \"status --short\"、\"log -5\"、\"diff\"、\"add src/main.py\"",
            },
        },
        "required": ["args"],
    },
    category="版本控制",
    when_not_to_use=(
        "只是改几个文件、想看自己写了啥，用 read_file / build_diff 更直接，不用 git。"
        "本工具用于「看清整个仓库的状态 / 与上次提交的差异 / 生成提交信息参考」。"
        "绝不要用它 push、reset --hard 或改写历史——这些被禁止，且可能破坏用户仓库。"
    ),
)
def git(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    tokens = _guard(args.get("args") or "")
    cwd = Path(str(ctx.workspace)).expanduser().resolve()
    timeout = float(getattr(ctx.config, "command_timeout", 120))

    try:
        proc = subprocess.run(
            ["git", *tokens],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return ToolResult.failure(
            "未找到 git 可执行文件（命令未安装或不在 PATH 中）",
            hint="请先安装 Git，或改用 read_file / build_diff 查看改动。",
        )
    except subprocess.TimeoutExpired:
        return ToolResult.failure(
            f"git {' '.join(tokens)} 执行超时（>{timeout:.0f}s）",
            hint="命令可能卡住，请缩小范围后重试。",
        )

    out = (proc.stdout or "") + (proc.stderr or "")
    if not out.strip():
        out = f"（git {' '.join(tokens)} 无输出；退出码 {proc.returncode}）"
    max_chars = int(getattr(ctx.config, "max_tool_output_chars", 12_000))
    if len(out) > max_chars:
        from ..security import truncate_output
        out = truncate_output(out, max_chars, note="git 输出过长")

    if proc.returncode != 0:
        return ToolResult.failure(
            f"git {' '.join(tokens)} 返回非零退出码 {proc.returncode}",
            hint="查看上方输出定位原因；若是『not a git repository』，说明当前目录还不是仓库。",
            meta={"returncode": proc.returncode},
        )
    return ToolResult.success(out, meta={"command": "git " + " ".join(tokens)})


def register(registry) -> None:
    registry.register(git)
