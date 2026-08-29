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

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..errors import SecurityError, ToolError
from ..security import truncate_output
from .base import DIFF_CAPTURE_CAP, ToolContext, ToolResult, ToolSpec, tool_spec

__all__ = ["read_file", "write_file", "edit_block", "list_dir", "register"]

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
    when_not_to_use=(
        "不要为了「看看有什么」就把整个大文件读进来——先 list_dir 看清结构，"
        "或用 grep_search 直接定位；只读需要改的那一段（offset/limit）。"
        "二进制文件（图片/压缩包）读出来是乱码，别读。"
    ),
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


def _count_lines(path: Path) -> int:
    """流式统计行数：不把整个文件读进内存，用于大文件覆盖写时的回执统计。

    注意：内部辅助函数必须放在 @tool_spec 装饰器**之外**——装饰器紧贴它下面的第一个
    def，误插到中间会让装饰器挂到辅助函数上，工具就退化成裸函数了。
    """
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


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
    when_not_to_use=(
        "改一个已存在文件里的少量内容时，不要用 write_file 整体重写——"
        "长文件会被输出上限截断（V0 的 HTTP 400 就是这么来的），用 edit_block。"
        "也不要用它做小步追加日志，append=true 更合适。"
    ),
)
def write_file(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    target = ctx.resolve(args["path"])
    content = args.get("content") or ""
    append = bool(args.get("append"))

    if target.exists() and target.is_dir():
        raise ToolError(f"{args['path']} 是目录，不能作为文件写入", tool="write_file")

    existed = target.exists()
    old_lines = 0
    old_text = ""
    old_captured = False          # 原文件内容是否真的读到了（过大时放弃，避免撑爆内存）
    backup_path = ""

    if existed:
        try:
            # 只在体量可控时才把原文读进内存——覆盖写一个几十 MB 的日志文件时，
            # 为了生成一份注定超限被丢弃的 diff 而把全文驻留内存不划算。
            # 行数另行流式统计，保证回执里的「N → M 行」始终准确。
            if target.stat().st_size <= DIFF_CAPTURE_CAP * 4:   # UTF-8 最多 4 字节/字符
                old_text = target.read_text(encoding="utf-8", errors="replace")
                old_captured = True
            old_lines = _count_lines(target)
        except OSError:
            old_lines = 0
            old_text = ""
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
    # 原文没读到（文件过大）时必须显式标 captured=False：
    # 否则 before="" 会被当成"文件原本是空的"，diff 显示成全量新增，反而误导。
    content_captured = (not existed) or old_captured
    before_text = old_text if existed else None
    after_text = ((old_text + content) if append else content) if content_captured else content
    ctx.record_change("write", _rel(ctx, target), before=before_text, after=after_text,
                      path=_rel(ctx, target), captured=content_captured)

    detail = f"{'追加' if append else '写入'} {_rel(ctx, target)}：{old_lines} → {new_lines} 行，{len(content)} 字符"
    if backup_path:
        detail += f"\n原文件已备份至 {backup_path}"
    return ToolResult.success(detail, meta={"path": str(target), "lines": new_lines, "backup": backup_path})


# ----------------------------------------------------------------------------
# edit_block —— 精确替换（改大文件里的少量内容时必须用它）
# ----------------------------------------------------------------------------
@tool_spec(
    name="edit_block",
    description=(
        "精确替换文件中的一段文本：用 old_text 定位，替换成 new_text，文件其余部分原样保留。\n"
        "old_text 在文件中必须**唯一**；若不唯一，会返回每一处的行号，"
        "你补更多上下文后重试即可（确要一次改多处时传 expected_replacements）。\n"
        "修改已存在文件里的少量内容时**必须用本工具**——用 write_file 整体重写"
        "会因输出过长被截断，甚至丢参数导致调用失败。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径，相对工作区"},
            "old_text": {"type": "string", "description": "要被替换的原文片段（需唯一，不要抄行号）"},
            "new_text": {"type": "string", "description": "替换后的新文本（空串表示删除该片段）"},
            "expected_replacements": {
                "type": "integer",
                "description": "期望替换几处，默认 1；确要一次改多处时传对应数量",
                "default": 1,
            },
        },
        "required": ["path", "old_text", "new_text"],
    },
    category="文件",
    when_not_to_use=(
        "整篇重写（改动超过文件一半、或结构性重构）时用 write_file 更省事。"
        "还没 read_file 确认原文就别动手——old_text 靠猜必错。"
        "连续两次匹配失败就停手重新 read_file，不要靠加转义反复试。"
    ),
)
def edit_block(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    target = ctx.resolve(args["path"], must_exist=True)
    if target.is_dir():
        raise ToolError(f"{args['path']} 是目录，不能编辑", tool="edit_block")

    old_text = args.get("old_text")
    new_text = args.get("new_text")
    if old_text is None or new_text is None:
        raise ToolError("old_text 与 new_text 都是必填参数", tool="edit_block")
    if not str(old_text).strip():
        raise ToolError("old_text 不能为空，否则无法定位", tool="edit_block")

    try:
        original = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ToolError(f"读取失败：{exc}", tool="edit_block") from exc

    # 模型常把 read_file 的行号一起抄进来，先尝试剥离后再匹配
    needle, stripped = _normalize_needle(str(old_text), original)
    matches = _find_all(original, needle)
    expected = max(1, int(args.get("expected_replacements") or 1))

    if not matches:
        raise ToolError(
            f"在 {_rel(ctx, target)} 中找不到 old_text",
            tool="edit_block",
            hint=_not_found_hint(original, needle),
        )

    if len(matches) != expected:
        raise ToolError(
            f"old_text 在 {_rel(ctx, target)} 中匹配到 {len(matches)} 处，"
            f"但 expected_replacements={expected}。不唯一时拒绝替换，以免改错位置。",
            tool="edit_block",
            hint="请补更多上下文让 old_text 唯一，或传入正确的 expected_replacements：\n"
                 + _render_matches(original, matches, needle),
        )

    backup_path = ""
    if getattr(ctx.config, "backup_on_write", True):
        backup_path = _backup(ctx, target)

    new_content = original.replace(needle, str(new_text), expected)
    try:
        target.write_text(new_content, encoding="utf-8", newline="")
    except OSError as exc:
        raise ToolError(f"写入失败：{exc}", tool="edit_block",
                        hint="文件可能只读或被其它程序占用。") from exc

    ctx.record_change("edit", _rel(ctx, target), before=original, after=new_content, path=_rel(ctx, target))

    line_no = original.count("\n", 0, matches[0]) + 1
    delta = len(new_content.splitlines()) - len(original.splitlines())
    detail = (f"已替换 {_rel(ctx, target)} 第 {line_no} 行起的 "
              f"{len(needle.splitlines())} 行（共 {len(matches)} 处），"
              f"行数变化 {'+' if delta >= 0 else ''}{delta}")
    if stripped:
        detail += "\n提示：old_text 里带了行号前缀，已自动剥离后匹配；下次请不要抄行号。"
    if backup_path:
        detail += f"\n原文件已备份至 {backup_path}"
    return ToolResult.success(
        detail, meta={"path": str(target), "line": line_no, "backup": backup_path}
    )


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
    when_not_to_use=(
        "已知文件名要找内容时用 find_files/grep_search，别列一遍目录再猜；"
        "depth 别开太大（>3 层在 node_modules 类目录里会淹没上下文）。"
    ),
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
# ---- edit_block 的匹配辅助 ----
# read_file 的输出形如 "   12| def f():"，模型常把行号一起抄进 old_text
_LINE_NO_PREFIX = re.compile(r"^\s*\d+\|\s")


def _normalize_needle(old_text: str, original: str):
    """返回 (实际用于匹配的文本, 是否剥离过行号)。

    直接匹配失败时，尝试剥离行号前缀再匹配——能救回大量
    "内容明明一样却找不到" 的情况，同时回执里会提醒模型别再抄行号。
    """
    if old_text in original:
        return old_text, False
    lines = old_text.splitlines()
    stripped = [_LINE_NO_PREFIX.sub("", ln) for ln in lines]
    if stripped != lines:
        candidate = "\n".join(stripped)
        if old_text.endswith("\n"):
            candidate += "\n"
        if candidate in original:
            return candidate, True
    return old_text, False


def _find_all(text: str, needle: str) -> List[int]:
    """needle 在 text 中所有匹配的起始字符偏移。"""
    out: List[int] = []
    start = 0
    while True:
        i = text.find(needle, start)
        if i < 0:
            break
        out.append(i)
        start = i + 1
    return out


def _render_matches(original: str, matches: List[int], needle: str) -> str:
    """列出每处匹配的行号，帮模型决定补多少上下文。"""
    out = []
    for off in matches[:5]:
        line_no = original.count("\n", 0, off) + 1
        head = (needle.splitlines() or [""])[0].strip()[:60]
        out.append(f"  第 {line_no} 行：{head}")
    if len(matches) > 5:
        out.append(f"  ...（共 {len(matches)} 处，仅列出前 5 处）")
    return "\n".join(out)


def _not_found_hint(original: str, needle: str) -> str:
    """找不到时给出可操作的排查方向，并附上文件中最相近的几行。"""
    msg = ["检查：① 是否误抄了 read_file 的行号；② 缩进与空白是否一致；③ 文件是否已被改动过。"]
    first = (needle.splitlines() or [""])[0].strip()
    if first:
        hits = []
        for i, line in enumerate(original.splitlines(), 1):
            if first in line:
                hits.append(f"  第 {i} 行：{line.strip()[:80]}")
                if len(hits) >= 3:
                    break
        if hits:
            msg.append("文件中存在相似内容：\n" + "\n".join(hits))
    return "\n".join(msg)


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
    registry.register_many([read_file, write_file, edit_block, list_dir])
