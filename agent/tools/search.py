"""
检索工具：grep_search / find_files。

纯 Python 实现（不依赖 ripgrep），好处是跨平台零依赖、行为可预期。
对模型来说，检索能力决定了"能不能先看清代码再动手"——
比直接 read_file 逐文件扫描省下大量上下文。
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from ..errors import ToolError
from .base import ToolContext, ToolResult, tool_spec

__all__ = ["grep_search", "find_files", "register"]

_SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea", ".agent_backups",
    "dist", "build", ".next", "coverage",
}
_MAX_FILE_BYTES = 2 * 1024 * 1024  # 跳过大于 2MB 的文件


@tool_spec(
    name="grep_search",
    description=(
        "在工作区内按正则搜索文件内容，返回 文件:行号:内容。"
        "用于定位函数定义、报错来源、配置项等，比逐个 read_file 高效得多。"
        "设置 context > 0 可在每条匹配上下各多显示几行代码，先看明白再动手。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "正则表达式，如 `def test_|class .*Test`"},
            "path": {"type": "string", "description": "搜索起点，相对工作区，默认 '.'", "default": "."},
            "include": {"type": "string", "description": "文件名过滤通配符，如 `*.py`，默认不限", "default": "*"},
            "max_matches": {"type": "integer", "description": "最多返回多少条匹配 / 多少个文件，默认 50", "default": 50},
            "case_sensitive": {"type": "boolean", "description": "是否区分大小写，默认 false", "default": False},
            "context": {"type": "integer", "description": "匹配行前后各多显示几行上下文（默认 0，即只显示匹配行）", "default": 0},
        },
        "required": ["pattern"],
    },
    category="检索",
    when_not_to_use=(
        "已知确切文件名就直接 read_file；要按名字找文件用 find_files。"
        "别用 .* 这类宽泛模式——命中几百条只会淹没上下文，先用 include 收窄范围。"
        "context 别开太大，几行足够看清上下文，开太大同样会淹没关键信息。"
    ),
)
def grep_search(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    pattern = args.get("pattern") or ""
    if not pattern:
        raise ToolError("搜索模式为空", tool="grep_search")
    try:
        regex = re.compile(pattern, 0 if args.get("case_sensitive") else re.IGNORECASE)
    except re.error as exc:
        raise ToolError(f"正则表达式非法：{exc}", tool="grep_search",
                        hint="注意 Python re 语法；特殊字符需要转义。") from exc

    root = ctx.resolve(args.get("path") or ".", must_exist=True)
    include = args.get("include") or "*"
    max_matches = max(1, int(args.get("max_matches") or 50))
    context = max(0, int(args.get("context") or 0))

    files: List[Path] = [root] if root.is_file() else _iter_files(root, include)
    scanned = 0

    # ---- 仅匹配行（context=0）：保持「文件:行号:内容」紧凑格式 ----
    if context == 0:
        hits: List[str] = []
        for f in files:
            if len(hits) >= max_matches:
                break
            try:
                if f.stat().st_size > _MAX_FILE_BYTES:
                    continue
                text = f.read_text(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                continue
            scanned += 1
            rel = ctx.guard.relpath(f)
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hits.append(f"{rel}:{lineno}: {line.strip()[:300]}")
                    if len(hits) >= max_matches:
                        break
        if not hits:
            return ToolResult.success(
                f"未找到匹配 /{pattern}/（扫描 {scanned} 个文件）",
                meta={"matches": 0, "scanned": scanned},
            )
        tail = f"\n...（已达上限 {max_matches}，请缩小范围或改用更精确的模式）" if len(hits) >= max_matches else ""
        return ToolResult.success(
            f"匹配 /{pattern}/：{len(hits)} 处（扫描 {scanned} 个文件）\n" + "\n".join(hits) + tail,
            meta={"matches": len(hits), "scanned": scanned},
        )

    # ---- 带上下文：每个文件输出一个带行号的小代码块，> 标出命中行 ----
    blocks: List[str] = []
    total_matches = 0
    for f in files:
        if len(blocks) >= max_matches:
            break
        try:
            if f.stat().st_size > _MAX_FILE_BYTES:
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        scanned += 1
        lines = text.splitlines()
        match_idx = [i for i, ln in enumerate(lines) if regex.search(ln)]
        if not match_idx:
            continue
        idx_set = set(match_idx)
        total_matches += len(match_idx)
        # 合并相邻区间，避免重叠上下文重复显示
        ranges: List[List[int]] = []
        for i in match_idx:
            lo = max(0, i - context)
            hi = min(len(lines) - 1, i + context)
            if ranges and lo <= ranges[-1][1] + 1:
                ranges[-1][1] = hi
            else:
                ranges.append([lo, hi])
        rel = ctx.guard.relpath(f)
        buf = [f"── {rel} ──"]
        for lo, hi in ranges:
            for j in range(lo, hi + 1):
                mark = ">" if j in idx_set else " "
                buf.append(f"{j + 1:>5}{mark} {lines[j].rstrip()[:300]}")
            buf.append("    …")
        blocks.append("\n".join(buf))

    if not blocks:
        return ToolResult.success(
            f"未找到匹配 /{pattern}/（扫描 {scanned} 个文件）",
            meta={"matches": 0, "scanned": scanned},
        )
    tail = f"\n\n...（已达上限 {max_matches} 个文件，请缩小范围）" if len(blocks) >= max_matches else ""
    return ToolResult.success(
        f"匹配 /{pattern}/：{len(blocks)} 个文件含命中（共 {total_matches} 处，扫描 {scanned} 个文件）\n"
        + "\n\n".join(blocks) + tail,
        meta={"files_with_match": len(blocks), "matches": total_matches, "scanned": scanned},
    )


@tool_spec(
    name="find_files",
    description="按文件名通配符查找文件，返回匹配的路径列表。用于快速弄清项目结构。",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "文件名通配符，如 `*.py`、`test_*.py`、`**/*.md`", "default": "*"},
            "path": {"type": "string", "description": "搜索起点，相对工作区，默认 '.'", "default": "."},
            "max_results": {"type": "integer", "description": "最多返回多少条，默认 100", "default": 100},
        },
        "required": [],
    },
    category="检索",
    when_not_to_use=(
        "要找的是**内容**（函数/报错/配置项）时用 grep_search，本工具只按文件名匹配。"
        "别用 `*`（全量）去摸清项目结构，那正是 list_dir 的活。"
    ),
)
def find_files(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    root = ctx.resolve(args.get("path") or ".", must_exist=True)
    pattern = args.get("pattern") or "*"
    max_results = max(1, int(args.get("max_results") or 100))

    results: List[str] = []
    for f in _iter_files(root, pattern):
        results.append(ctx.guard.relpath(f))
        if len(results) >= max_results:
            break

    if not results:
        return ToolResult.success(f"未找到匹配 {pattern} 的文件", meta={"count": 0})
    tail = f"\n...（已达上限 {max_results}）" if len(results) >= max_results else ""
    return ToolResult.success(f"找到 {len(results)} 个文件：\n" + "\n".join(results) + tail, meta={"count": len(results)})


def _iter_files(root: Path, include: str):
    """遍历目录下的文件，跳过噪声目录。

    匹配时同时尝试「文件名」与「相对路径」两种口径，
    这样 `*.py`、`test_*.py` 与 `src/**/*.py` 都能按直觉命中。
    """
    if root.is_file():
        yield root
        return
    pattern = include or "*"
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in sorted(filenames):
            p = Path(dirpath) / name
            rel = p.relative_to(root).as_posix()
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, pattern):
                yield p


def register(registry) -> None:
    registry.register_many([grep_search, find_files])
