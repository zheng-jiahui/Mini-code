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
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "正则表达式，如 `def test_|class .*Test`"},
            "path": {"type": "string", "description": "搜索起点，相对工作区，默认 '.'", "default": "."},
            "include": {"type": "string", "description": "文件名过滤通配符，如 `*.py`，默认不限", "default": "*"},
            "max_matches": {"type": "integer", "description": "最多返回多少条匹配，默认 50", "default": 50},
            "case_sensitive": {"type": "boolean", "description": "是否区分大小写，默认 false", "default": False},
        },
        "required": ["pattern"],
    },
    category="检索",
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

    files: List[Path] = [root] if root.is_file() else _iter_files(root, include)
    hits: List[str] = []
    scanned = 0
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
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                hits.append(f"{ctx.guard.relpath(f)}:{lineno}: {line.strip()[:300]}")
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
