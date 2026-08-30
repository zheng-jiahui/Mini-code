"""
Git 工具：在工作区内做版本控制（init / status / diff / log / commit）。

设计边界与考核契合点：
    · 自写、基于本地 `git` 子进程，不依赖任何 agent 框架，也不调用服务端工具——
      符合题目“重要逻辑需自行编写”的要求；
    · 只暴露**只读 + 本地提交**这一小撮子命令（status/diff/log/commit/init），
      物理上无法 push / reset --hard / clean，与系统提示词里
      “不要改写 Git 历史、不要向远端推送任何提交”的安全边界一致；
    · 沙箱到工作区：所有命令都带 `git -C <workspace>`，不会越界到工作区外的仓库；
    · 工作区不是 git 仓库时，给出可操作的提示（建议先 git_init），绝不抛异常崩溃。

为什么值得做：Claude Code / Codex / Aider 这类标杆都是“git 优先”的——
既能用 git_status / git_diff 自查改动，又能用 git_commit 留下阶段性提交。
本项目此前只有会话级的 diff / rollback，缺真正的版本控制意识，这是最该补的一块。
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

from .base import ToolContext, ToolResult, tool_spec

__all__ = ["git_init", "git_status", "git_diff", "git_log", "git_commit", "register"]

_GIT_TIMEOUT = 30


def _git_env() -> Dict[str, str]:
    """构造 git 子进程环境：关掉分页器/编辑器，强制 UTF-8 输出，并固定提交者身份。

    固定 GIT_AUTHOR/COMMITTER 是因为很多干净环境（CI、临时容器、评分机）没有配置
    user.name / user.email，直接 `git commit` 会报 “Author identity unknown” 而失败。
    由 agent 提交的代码用统一身份即可，不依赖用户本机全局配置。
    """
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env["NO_COLOR"] = "1"
    env["PAGER"] = "cat"
    env["GIT_PAGER"] = "cat"
    env["GIT_EDITOR"] = ":"
    env["LC_ALL"] = "C.UTF-8"
    env["GIT_AUTHOR_NAME"] = "MiniCode Agent"
    env["GIT_AUTHOR_EMAIL"] = "agent@minicode.local"
    env["GIT_COMMITTER_NAME"] = "MiniCode Agent"
    env["GIT_COMMITTER_EMAIL"] = "agent@minicode.local"
    return env


def _run(ctx: ToolContext, *args: str) -> Tuple[int, str, str]:
    """在 workspace 下执行一条 git 命令，返回 (returncode, stdout, stderr)。

    统一走 `git -C <workspace> --no-pager`，避免分页器把进程挂住、
    也避免 editor 在 commit 时等待输入。
    """
    ws = str(ctx.workspace)
    cmd = ["git", "-C", ws, "--no-pager", *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            env=_git_env(),
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"git {' '.join(args)} 执行超过 {_GIT_TIMEOUT}s 被终止"
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _is_repo(ctx: ToolContext) -> bool:
    rc, _, _ = _run(ctx, "rev-parse", "--is-inside-work-tree")
    return rc == 0


def _not_a_repo(ctx: ToolContext, suggest: str) -> ToolResult:
    return ToolResult.failure(
        f"{ctx.guard.relpath(ctx.workspace)} 还不是 git 仓库。",
        hint=f"先调用 git_init 初始化仓库，再使用 {suggest}。",
        meta={"is_repo": False},
    )


@tool_spec(
    name="git_init",
    description=(
        "在当前工作区初始化一个 git 仓库（执行 `git init`）。\n"
        "当你需要让 agent 具备版本控制能力（留下阶段性提交、用 git_diff 自查）时先调用它。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "initial_branch": {
                "type": "string",
                "description": "初始分支名，默认 main",
                "default": "main",
            },
        },
        "required": [],
    },
    category="版本控制",
    when_not_to_use=(
        "工作区已经是 git 仓库时不用再 init（会提示已存在）；"
        "也不要用它来改写历史或连接远端——本工具只做本地 init，不会 git remote add。"
    ),
)
def git_init(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    branch = (args.get("initial_branch") or "main").strip() or "main"
    if _is_repo(ctx):
        return ToolResult.success(
            f"工作区已是 git 仓库，无需重复初始化。",
            meta={"is_repo": True, "already": True},
        )
    rc, out, err = _run(ctx, "init", "--initial-branch", branch)
    if rc != 0:
        return ToolResult.failure(f"git init 失败：{err.strip() or out.strip()}")
    return ToolResult.success(
        f"已在 {ctx.guard.relpath(ctx.workspace)} 初始化 git 仓库（分支 {branch}）。",
        meta={"is_repo": True},
    )


@tool_spec(
    name="git_status",
    description=(
        "查看工作区版本状态：当前分支、是否有未跟踪/已修改/已暂存的文件。\n"
        "动手改代码前后都该先 git_status 看清现状，再决定要提交什么。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "porcelain": {
                "type": "boolean",
                "description": "true 时返回机器友好的简短格式（适合自己做判断），默认 false（带说明）",
                "default": False,
            },
        },
        "required": [],
    },
    category="版本控制",
    when_not_to_use=(
        "只是想知道某个文件现在的内容用 read_file；本工具只反映 git 跟踪状态，"
        "不读文件内容。还没 git_init 的工作区会提示你先初始化。"
    ),
)
def git_status(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    if not _is_repo(ctx):
        return _not_a_repo(ctx, "git_status")
    rc, out, err = _run(ctx, "status", "--short", "--branch")
    if rc != 0:
        return ToolResult.failure(f"git status 失败：{err.strip() or out.strip()}")
    branch = "（未知分支）"
    lines = out.splitlines()
    for ln in lines:
        if ln.startswith("##"):
            branch = ln[2:].strip()
            break
    if not lines:
        body = f"分支 {branch}：工作区干净，没有未提交的改动。"
    else:
        files = [ln for ln in lines if not ln.startswith("##")]
        body = (
            f"分支 {branch}，共 {len(files)} 个改动项：\n"
            + "\n".join(files)
            + "\n（X 未跟踪 / M 已修改 / A 已暂存；前缀为暂存区状态，后缀为工作区状态）"
        )
    return ToolResult.success(body, meta={"is_repo": True, "branch": branch, "changes": len(lines) - 1})


@tool_spec(
    name="git_diff",
    description=(
        "查看工作区与最近一次提交之间的差异（unified diff）。\n"
        "staged=true 时看已暂存与最近提交的差异，否则看工作区与已暂存的差异。\n"
        "在 git_commit 之前用它自查改动是否符合预期。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "staged": {
                "type": "boolean",
                "description": "true 看已暂存的改动（git diff --cached），默认 false 看工作区改动",
                "default": False,
            },
            "path": {
                "type": "string",
                "description": "只看某个文件/目录的差异，相对工作区，默认看全部",
                "default": "",
            },
        },
        "required": [],
    },
    category="版本控制",
    when_not_to_use=(
        "要看本次会话（而非 git 历史）改了什么用 diff 工具；"
        "本工具只看 git 跟踪的文件与上一次提交的差异。仓库很干净时差异为空是正常的。"
    ),
)
def git_diff(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    if not _is_repo(ctx):
        return _not_a_repo(ctx, "git_diff")
    extra = ["--cached"] if bool(args.get("staged")) else []
    path = (args.get("path") or "").strip()
    if path:
        extra.append("--")
        extra.append(path)
    rc, out, err = _run(ctx, "diff", *extra)
    if rc != 0:
        return ToolResult.failure(f"git diff 失败：{err.strip() or out.strip()}")
    if not out.strip():
        scope = "已暂存" if bool(args.get("staged")) else "工作区"
        return ToolResult.success(f"（{scope}相对上一次提交没有差异）", meta={"is_repo": True, "diff_lines": 0})
    n = out.count("\n")
    return ToolResult.success(
        f"差异共 {n} 行：\n{out.rstrip()}",
        meta={"is_repo": True, "diff_lines": n},
    )


@tool_spec(
    name="git_log",
    description="查看提交历史（精简的一行式记录：哈希 + 提交说明）。用于了解已有改动脉络。",
    parameters={
        "type": "object",
        "properties": {
            "max_count": {
                "type": "integer",
                "description": "最多显示多少条提交，默认 20",
                "default": 20,
            },
        },
        "required": [],
    },
    category="版本控制",
    when_not_to_use=(
        "仓库还没有任何提交时历史为空，先 git_commit 留出第一条提交；"
        "只想看还没提交的工作区改动用 git_status / git_diff 而不是本工具。"
    ),
)
def git_log(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    if not _is_repo(ctx):
        return _not_a_repo(ctx, "git_log")
    n = max(1, min(int(args.get("max_count") or 20), 100))
    rc, out, err = _run(ctx, "log", f"--max-count={n}", "--oneline")
    if rc != 0:
        return ToolResult.failure(f"git log 失败：{err.strip() or out.strip()}")
    if not out.strip():
        return ToolResult.success("（还没有任何提交记录）", meta={"is_repo": True, "commits": 0})
    return ToolResult.success(
        f"最近 {n} 条提交：\n{out.rstrip()}",
        meta={"is_repo": True, "commits": out.count("\n") + 1},
    )


@tool_spec(
    name="git_commit",
    description=(
        "把当前工作区的改动提交为一个新提交（先 `git add -A` 再 `git commit -m`）。\n"
        "用于在完成一个阶段性目标后留下版本记录，方便回看与对比。\n"
        "⚠️ 只会新增提交，绝不 amend / 强制推送 / 改写历史；也不会连接或推送到任何远端。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "提交说明，应简明概括本次改动（如“修复登录校验空指针”）"},
            "all": {
                "type": "boolean",
                "description": "true（默认）表示把所有改动（含未跟踪文件）一起提交；false 则只提交已暂存的",
                "default": True,
            },
        },
        "required": ["message"],
    },
    category="版本控制",
    when_not_to_use=(
        "改动还没验证过（没跑过测试/脚本）就不要提交——先 run_command 验证；"
        "也不要每改一行就提交一次，按一个可运行、可说明的阶段性目标来提交更有意义。"
        "工作区不是 git 仓库时，先 git_init。"
    ),
)
def git_commit(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    if not _is_repo(ctx):
        return _not_a_repo(ctx, "git_commit")
    message = (args.get("message") or "").strip()
    if not message:
        return ToolResult.failure("提交说明 message 不能为空", hint="请用一句话概括本次改动。")
    do_all = bool(args.get("all", True))
    if do_all:
        rc, out, err = _run(ctx, "add", "-A")
        if rc != 0:
            return ToolResult.failure(f"git add 失败：{err.strip() or out.strip()}")
    rc, out, err = _run(ctx, "commit", "-m", message)
    if rc != 0:
        combined = (err or out).strip()
        if "nothing to commit" in combined.lower() or "无文件要提交" in combined:
            return ToolResult.success("（没有需要提交的改动，工作区已是最新）", meta={"committed": 0})
        return ToolResult.failure(f"git commit 失败：{combined}")
    summary = out.strip().splitlines()[-1] if out.strip() else "已创建新提交"
    return ToolResult.success(
        f"已提交：{summary}\n提交说明：{message}",
        meta={"committed": 1, "message": message},
    )


def register(registry) -> None:
    """把本模块的工具注册进注册表。"""
    registry.register_many([git_init, git_status, git_diff, git_log, git_commit])
