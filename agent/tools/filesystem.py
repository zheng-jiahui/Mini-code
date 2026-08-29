"""
文件系统工具：read_file / write_file / list_dir。

设计要点：
    · 全部路径先过 PathGuard（沙箱），再落盘；
    · read_file 带行号输出 —— 模型后续若要"改第 40 行"，行号是唯一可靠锚点；
    · write_file 是**整体覆盖**语义，因此覆盖前自动备份到 .agent_backups/<时间戳>/，
      用户可用 /undo 回滚；同时把"行数变化"回执给模型，便于它判断写入是否符合预期；
    · 二进制文件不硬读，直接告知，避免把一堆乱码塞进上下文。
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..errors import SecurityError, ToolError
from ..security import truncate_output
from .base import ToolContext, ToolResult, ToolSpec, tool_spec

__all__ = ["read_file", "write_file", "list_dir", "register"]

_IGNORE_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea", ".vscode",
    ".agent_backups", ".agent_sessions", "dist", "build", ".next",
}

_TEXT_SUFFIX_HINT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp", ".hpp",
    ".go", ".rs", ".rb", ".php", ".sh", ".bash", ".sql", ".yaml", ".yml",
    ".json", ".toml", ".ini", ".cfg", ".md", ".txt", ".csv", ".html", ".css",
}


# ----------------------------------------------------------------------------
# read_file
# ----------------------------------------------------------------------------
@tool_spec(
    name="read_file",
    description=(
        "读取文本文件内容，带行号返回。"
        "修改任何已存在的文件之前，必须先用本工具读取原文，禁止凭猜测重写。"
        "大文件可用 offset/limit 分段读取。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径，相对工作区，如 src/main.py"},
            "offset": {"type": "integer", "description": "从第几行开始读（1 起，默认 1）", "default": 1},
            "limit": {"type": "integer", "description": "最多读多少行，默认读全部（超长自动截断）"},
        },
        "required": ["path"],
    },
    category="文件",
)
def read_file(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = ctx.resolve(args["path"], must_exist=True)
    if path.is_dir():
        raise ToolError(f"{args['path']} 是目录而不是文件", tool="read_file", hint="读目录请用 list_dir。")

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ToolError(f"读取失败：{exc}", tool="read_file") from exc

    if _looks_binary(raw):
        return ToolResult.failure(
            f"{_rel(ctx, path)} 疑似二进制文件（{len(raw)} 字节），已拒绝读取",
            hint="若确实需要，请让用户手动提供文本版本。",
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")

    lines = text.splitlines()
    offset = max(1, int(args.get("offset") or 1))
    limit = int(args.get("limit") or 0) or len(lines)
    chunk = lines[offset - 1: offset - 1 + limit]

    max_chars = int(getattr(ctx.config, "max_file_read_chars", 40_000))
    numbered = "\n".join(f"{i + offset:>5}| {line}" for i, line in enumerate(chunk))
    truncated = False
    if len(numbered) > max_chars:
        numbered = truncate_output(numbered, max_chars, note="文件内容过长")
        truncated = True

    header = f"文件：{_rel(ctx, path)}（共 {len(lines)} 行，本次返回第 {offset}-{offset + len(chunk) - 1} 行）"
    return ToolResult.success(
        f"{header}\n{numbered}",
        meta={
            "path": str(path),
            "total_lines": len(lines),
            "truncated": truncated,
        },
    )


# ----------------------------------------------------------------------------
# write_file
# ----------------------------------------------------------------------------
@tool_spec(
    name="write_file",
    description=(
        "写入文件（整体覆盖语义）。必须给出文件的完整内容，而不是 diff 或省略号。"
        "父目录不存在时自动创建。覆盖已有文件前会自动备份。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "目标文件路径，相对工作区，如 src/app.py"},
            "content": {"type": "string", "description": "文件完整内容"},
            "append": {"type": "boolean", "description": "true 表示追加而非覆盖，默认 false", "default": False},
        },
        "required": ["path", "content"],
    },
    dangerous=True,
    category="文件",
)
def write_file(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    target = ctx.resolve(args["path"])
    content = args.get("content") or ""
    append = bool(args.get("append"))

    if target.exists() and target.is_dir():
        raise ToolError(f"{args['path']} 是目录，不能作为文件写入", tool="write_file")

    existed = target.exists()
    old_lines = 0
    backup_path = ""

    if existed:
        try:
            old_text = target.read_text(encoding="utf-8", errors="replace")
            old_lines = len(old_text.splitlines())
        except OSError:
            old_lines = 0
        if not append and getattr(ctx.config, "backup_on_write", True):
            backup_path = _backup(ctx, target)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if append:
            with target.open("a", encoding="utf-8", newline="") as f:
                f.write(content)
        else:
            target.write_text(content, encoding="utf-8", newline="")
    except OSError as exc:
        raise ToolError(f"写入失败：{exc}", tool="write_file", hint="检查路径是否越界、文件是否被占用或只读。") from exc

    new_lines = len(content.splitlines())
    ctx.record_change("write", _rel(ctx, target))

    detail = f"{'追加' if append else '写入'} {_rel(ctx, target)}：{old_lines} → {new_lines} 行，{len(content)} 字符"
    if backup_path:
        detail += f"\n原文件已备份至 {backup_path}"
    return ToolResult.success(detail, meta={"path": str(target), "lines": new_lines, "backup": backup_path})


# ----------------------------------------------------------------------------
# list_dir
# ----------------------------------------------------------------------------
@tool_spec(
    name="list_dir",
    description="列出目录结构（树状，自动忽略 .git / node_modules / __pycache__ 等噪声目录）。",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "目录路径，相对工作区，默认 '.'", "default": "."},
            "depth": {"type": "integer", "description": "递归层数，1 表示只看一层，默认 2", "default": 2},
            "max_entries": {"type": "integer", "description": "最多列出多少条目，默认 200", "default": 200},
        },
        "required": [],
    },
    category="文件",
)
def list_dir(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    root = ctx.resolve(args.get("path") or ".", must_exist=True)
    if not root.is_dir():
        raise ToolError(f"{args.get('path')} 不是目录", tool="list_dir")

    depth = max(1, min(int(args.get("depth") or 2), 5))
    max_entries = max(20, int(args.get("max_entries") or 200))

    lines: List[str] = []
    count = 0

    def walk(cur: Path, level: int, prefix: str) -> None:
        nonlocal count
        if level > depth or count >= max_entries:
            return
        try:
            entries = sorted(cur.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except (OSError, PermissionError):
            return
        dirs = [e for e in entries if e.is_dir() and e.name not in _IGNORE_DIRS]
        files = [e for e in entries if e.is_file()]
        for e in dirs + files:
            if count >= max_entries:
                return
            count += 1
            mark = "/" if e.is_dir() else ""
            size = f"  ({_human_size(e.stat().st_size)})" if e.is_file() else ""
            lines.append(f"{prefix}{e.name}{mark}{size}")
            if e.is_dir():
                walk(e, level + 1, prefix + "  ")

    lines.append(f"{_rel(ctx, root)}/")
    walk(root, 1, "  ")
    if count >= max_entries:
        lines.append(f"...（条目数达到上限 {max_entries}，可用 depth 或具体子目录缩小范围）")
    return ToolResult.success("\n".join(lines), meta={"entries": count})


# ----------------------------------------------------------------------------
# 辅助
# ----------------------------------------------------------------------------
def _rel(ctx: ToolContext, path: Path) -> str:
    return ctx.guard.relpath(path)


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}GB"


def _looks_binary(raw: bytes) -> bool:
    if b"\x00" in raw[:4096]:
        return True
    try:
        raw[:8192].decode("utf-8")
        return False
    except UnicodeDecodeError:
        return True


def _backup_root(ctx: ToolContext) -> Path:
    """备份根目录：与 workplace 同级（项目根/.agent_backups）。

    config.workspace_root 是"生成代码的家"（…/workplace），它的父目录即项目根。
    """
    home = getattr(ctx.config, "workspace_root", None) or ctx.config.workspace
    base = Path(str(home)).expanduser().resolve().parent
    return base / (getattr(ctx.config, "backup_dir", None) or ".agent_backups")


def _backup(ctx: ToolContext, target: Path) -> str:
    """把即将被覆盖的文件备份到 .agent_backups/.overwrites/<任务名>/<时间戳>/。

    放在 .overwrites 这个隐藏目录下，是为了不干扰顶层的任务快照
    （.agent_backups/<任务名>_<时间戳>_<第N次>/），也避免影响"第几次"的计数。
    """
    try:
        rel = Path(ctx.guard.relpath(target))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        task_name = Path(str(ctx.config.workspace)).expanduser().resolve().name
        dest_dir = _backup_root(ctx) / ".overwrites" / task_name / stamp
        dest = dest_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, dest)
        return ctx.guard.relpath(dest)
    except OSError:
        return ""


def register(registry) -> None:
    """把本模块的工具注册进注册表。"""
    registry.register_many([read_file, write_file, list_dir])
