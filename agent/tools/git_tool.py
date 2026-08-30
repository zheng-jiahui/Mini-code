"""
Git 工具：git —— 让 agent 能看清仓库状态、比对改动、生成提交信息参考，并做「受控提交」。

安全边界（与考核要求一致）：
    本工具放行**只读类 git 子命令 + 安全的 add（暂存）+ 受控 commit（提交）**。
    其中 commit 做了严格限制：必须有提交信息、禁止改写历史的选项
    （--amend / --no-verify / --allow-empty / --date / -a / --all）、且禁止空暂存区提交，
    从机制上避免模型「误提交 / 改写历史 / 全量暂存」。
    明确拒绝 push / pull / fetch / clone / reset / checkout / clean / merge /
    rebase / revert 等会改写历史或触碰远端、或破坏用户仓库的操作。

为什么要做这个工具：
    商用编程智能体普遍能「看 diff / 看 log / 看 status / 做一次常规提交」。
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

# 放行：只读 + 安全的暂存（add）+ 受控提交（commit，见 _guard 的额外约束）。其余一律拒绝。
_ALLOWED_SUBCOMMANDS = {
    "status", "diff", "log", "show", "branch", "tag", "remote",
    "rev-parse", "blame", "ls-files", "shortlog", "stash", "add", "commit",
}
# 拒绝：会改写历史 / 触碰远端 / 破坏工作区的子命令（作为白名单之外的兜底拦截）
_DENY_SUBCOMMANDS = {
    "push", "pull", "fetch", "clone", "reset", "checkout", "clean",
    "merge", "rebase", "revert", "rm", "mv", "am", "apply", "cherry-pick",
    "submodule", "init", "config", "worktree", "gc",
}

# 受控 commit 禁止的选项（改写历史 / 跳过校验 / 自动全量暂存）
_COMMIT_FORBIDDEN = {"--amend", "--no-verify", "--allow-empty", "--date", "-a", "--all"}


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
    # commit 仅允许常规提交，禁止改写历史 / 跳过校验 / 自动全量暂存的选项
    if sub == "commit":
        for t in tokens[1:]:
            base = t.split("=", 1)[0]
            if base in _COMMIT_FORBIDDEN:
                raise ToolError(
                    f"git commit 禁止使用 {t}（会改写历史 / 跳过校验 / 自动全量暂存）",
                    tool="git",
                    hint="本工具的 commit 仅支持常规 `commit -m '<msg>'`，"
                         "不允许 --amend / --no-verify / --allow-empty / --date / -a / --all。",
                )
    return tokens


def _prepare_commit(tokens: List[str], message_param: Any) -> List[str]:
    """从 commit 参数中提取提交信息并拼回 tokens；缺失信息则抛 ToolError。

    优先用结构化 `message` 参数；若未提供，则从 args 内的 `-m` / `--message` 解析。
    最终只保留一个 `-m <msg>`，避免重复。
    """
    msg: Any = None
    cleaned = ["commit"]
    i = 1
    while i < len(tokens):
        t = tokens[i]
        if t in ("-m", "--message"):
            if i + 1 < len(tokens):
                msg = tokens[i + 1]
                i += 2
                continue
            raise ToolError("git commit 的 -m 缺少提交信息", tool="git")
        if t.startswith("--message="):
            msg = t.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(t)
        i += 1
    if message_param and str(message_param).strip():
        msg = str(message_param).strip()
    if not msg or not str(msg).strip():
        raise ToolError(
            "git commit 必须有提交信息",
            tool="git",
            hint="请通过 message 参数，或在 args 中写 `commit -m '你的提交说明'`。",
        )
    cleaned.extend(["-m", str(msg).strip()])
    return cleaned


# ----------------------------------------------------------------------------
# git
# ----------------------------------------------------------------------------
@tool_spec(
    name="git",
    description=(
        "在沙箱内执行**安全的** git 命令：查看仓库状态 / 比对改动 / 看提交历史 / 暂存文件 / 受控提交。"
        "例如 git status、git diff、git log -n、git show、git branch、git add <文件>。"
        "受控 commit 仅支持常规 `git commit -m '<msg>'`（需先 git add 暂存、禁止改写历史的选项）。"
        "push / pull / reset / checkout / merge / rebase 等会改写历史或触碰远端的操作被禁止。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "args": {
                "type": "string",
                "description": "git 后的子命令与参数，如 \"status --short\"、\"log -5\"、\"diff\"、\"add src/main.py\"、\"commit\"",
            },
            "message": {
                "type": "string",
                "description": "仅 commit 使用：提交信息。也可在 args 内用 `-m '说明'`。二者至少其一非空；本工具会拒绝空提交信息。",
            },
        },
        "required": ["args"],
    },
    category="版本控制",
    when_not_to_use=(
        "只是改几个文件、想看自己写了啥，用 read_file / build_diff 更直接，不用 git。"
        "本工具用于「看清整个仓库的状态 / 与上次提交的差异 / 生成提交信息参考 / 做一次常规提交」。"
        "绝不要用它 push、reset --hard 或改写历史——这些被禁止，且可能破坏用户仓库。"
    ),
)
def git(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    tokens = _guard(args.get("args") or "")
    cwd = Path(str(ctx.workspace)).expanduser().resolve()
    timeout = float(getattr(ctx.config, "command_timeout", 120))

    # 受控提交：先拼好带提交信息的 tokens，再做「空暂存区」拦截
    if tokens[0] == "commit":
        tokens = _prepare_commit(tokens, args.get("message"))
        try:
            staged = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(
                "检查暂存区超时，未执行提交",
                hint="请稍后重试，或改用终端执行 git commit。",
            )
        if not staged.stdout.strip():
            return ToolResult.failure(
                "没有已暂存（staged）的改动，无法提交",
                hint="请先用 `git add <文件>` 暂存要提交的文件，再执行 commit。",
            )

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
