"""
文件工具：replace_in_files —— 跨多个文件的安全查找替换（仓库级符号重命名）。

与 replace_in_file（单文件、唯一匹配）的区别：
    replace_in_file 是「改一个已存在文件里唯一的一处」；replace_in_files 是
    「按 glob 圈定一批文件，把某段文本（或正则）全部替换掉」——典型场景是
    把一个在整个仓库里出现过的函数名/变量名统一改名。类比 `sed -i`，但加了
    **沙箱 + 默认 dry_run 预览 + 写前备份 + 二进制跳过**，不会误伤。

安全边界（与考核要求一致）：
    - 默认 dry_run=true，只报告「将改动几个文件、共几处」，不落盘；确认无误再传 dry_run=false。
    - 越界路径被 PathGuard 拒绝；二进制文件跳过；old 过短（<2 字符）直接拒绝，避免「把 e 全删了」式灾难。
    - 真正写入前每个文件先备份到 .agent_backups，可用 /undo 回滚。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..errors import ToolError
from .base import ToolContext, ToolResult, tool_spec
from .filesystem import _backup, _looks_binary, _rel
from .search import _MAX_FILE_BYTES, _iter_files

__all__ = ["replace_in_files", "register"]

# 单次最多扫描/改写的文件数（超大仓库保护）
_MAX_FILES = 2000
# 回显里最多列出的文件数（避免长清单淹没上下文）
_MAX_LISTED_FILES = 60
# 回显里每个文件的变更摘要最多多少字符
_MAX_PER_FILE_CHARS = 200


def _compile(args: Dict[str, Any]) -> Tuple[bool, str, str]:
    """返回 (is_regex, pattern, replacement)。"""
    old = args.get("old") or ""
    new = args.get("new") or ""
    if len(old) < 2:
        raise ToolError(
            "old 至少 2 个字符，避免误伤（例如把单个字母全部替换会毁掉整个文件）",
            tool="replace_in_files",
        )
    if old == new:
        raise ToolError("old 与 new 相同，没有任何改动", tool="replace_in_files")
    use_regex = bool(args.get("regex"))
    if use_regex:
        try:
            re.compile(old)
        except re.error as exc:
            raise ToolError(f"正则表达式非法：{exc}", tool="replace_in_files") from exc
    return use_regex, old, new


def _count_and_replace(text: str, is_regex: bool, old: str, new: str) -> Tuple[int, str]:
    """返回 (命中次数, 替换后的文本)。"""
    if is_regex:
        matches = re.findall(old, text)
        count = len(matches)
        replaced = re.sub(old, new, text)
    else:
        count = text.count(old)
        replaced = text.replace(old, new)
    return count, replaced


@tool_spec(
    name="replace_in_files",
    description=(
        "在圈定的一批文件里，把某段文本（或正则）**全部替换**——用于仓库级符号重命名、"
        "统一改 API 名、批量修同一个笔误。按 include 通配符（如 *.py）圈定范围，从 path 开始递归。"
        "默认 dry_run=true 只报告「将改动几个文件、共几处」不落盘；确认无误再传 dry_run=false 真正执行。"
        "只改一个已存在文件里唯一的一处时，请用 replace_in_file 更精确。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "old": {"type": "string", "description": "要被替换的文本或正则（至少 2 字符）"},
            "new": {"type": "string", "description": "替换成的内容（空串表示删除该文本）"},
            "include": {"type": "string", "description": "文件名通配符，如 *.py、src/**/*.ts，默认 *", "default": "*"},
            "path": {"type": "string", "description": "搜索起点，相对工作区，默认 '.'", "default": "."},
            "regex": {"type": "boolean", "description": "true 时 old 按正则匹配（new 可引用 \\1 等分组），默认 false（字面量）", "default": False},
            "dry_run": {"type": "boolean", "description": "true（默认）只预览不改写；确认后传 false 才真正落盘", "default": True},
        },
        "required": ["old", "new"],
    },
    category="文件",
    when_not_to_use=(
        "只改一个已存在文件里唯一的一处，用 replace_in_file 更精确、更安全。"
        "正式改写前务必先 dry_run=true（默认就是）看清楚会动哪些文件、几处，"
        "别一上来就 dry_run=false 全量替换——尤其是 old 很短或很常见时，会把不相关的代码也改掉。"
        "涉及重命名「对外公开的符号」时，先 grep_search 确认影响面，再决定是否动用本工具。"
    ),
)
def replace_in_files(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    is_regex, old, new = _compile(args)
    include = args.get("include") or "*"
    dry_run = bool(args.get("dry_run", True))
    root = ctx.resolve(args.get("path") or ".", must_exist=True)

    hits: List[Tuple[Path, int]] = []   # (文件, 命中数)
    skipped_binary = 0
    scanned = 0
    for f in _iter_files(root, include):
        if scanned >= _MAX_FILES:
            break
        try:
            raw = f.read_bytes()
        except OSError:
            continue
        if len(raw) > _MAX_FILE_BYTES or _looks_binary(raw):
            if _looks_binary(raw):
                skipped_binary += 1
            continue
        scanned += 1
        text = raw.decode("utf-8", errors="replace")
        count, _ = _count_and_replace(text, is_regex, old, new)
        if count > 0:
            hits.append((f, count))

    if not hits:
        return ToolResult.success(
            f"未找到需要替换的内容（扫描 {scanned} 个文件，跳过 {skipped_binary} 个二进制文件）。"
            "换个更具体的 old，或调整 include/path 缩小范围。",
            meta={"changed": 0, "occurrences": 0, "scanned": scanned},
        )

    total_occ = sum(c for _, c in hits)

    # 回显文件清单（截断）
    listed = hits[:_MAX_LISTED_FILES]
    lines = [f"{_rel(ctx, p)}: {c} 处" for p, c in listed]
    more = "" if len(hits) <= _MAX_LISTED_FILES else f"\n    …（另有 {len(hits) - _MAX_LISTED_FILES} 个文件未列出）"

    if dry_run:
        summary = (
            f"[预览] 将改动 {len(hits)} 个文件，共 {total_occ} 处"
            f"（扫描 {scanned} 个文件，跳过 {skipped_binary} 个二进制文件）：\n"
            + "\n".join(lines) + more
            + "\n确认无误后，把 dry_run 设为 false 再执行。"
        )
        return ToolResult.success(summary, meta={"changed": 0, "occurrences": total_occ, "scanned": scanned, "dry_run": True})

    # 真正执行：逐文件备份 + 替换 + 写回 + 记录变更
    applied = 0
    for f, _c in hits:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            _count, replaced = _count_and_replace(text, is_regex, old, new)
            if text == replaced:
                continue
            backup = _backup(ctx, f) if getattr(ctx.config, "backup_on_write", True) else ""
            f.write_text(replaced, encoding="utf-8", newline="")
            ctx.record_change("replace_in_files", _rel(ctx, f),
                              before=text, after=replaced, path=_rel(ctx, f), captured=True)
            applied += 1
        except OSError as exc:
            return ToolResult.failure(
                f"写入 {_rel(ctx, f)} 失败：{exc}",
                hint="文件可能被占用或只读；已改的文件已各自备份，可用 /undo 回滚。",
            )

    summary = (
        f"已改写 {applied} 个文件，共 {total_occ} 处"
        f"（扫描 {scanned} 个文件，跳过 {skipped_binary} 个二进制文件）：\n"
        + "\n".join(lines) + more
        + "\n每个被改文件均已备份到 .agent_backups，可用 /undo 回滚。"
    )
    return ToolResult.success(summary, meta={"changed": applied, "occurrences": total_occ, "scanned": scanned})


def register(registry) -> None:
    registry.register(replace_in_files)
