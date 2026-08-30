"""
检索工具：recall —— 按「相关度」在工作区里找回最相关的文件与片段。

与 grep_search 的区别：
    grep_search 是「精确正则匹配某几行」；recall 是「给定一段自然语言/想法，
    找出**哪些文件整体最相关**」，并对每个命中文件给出最相关的那几行片段。
    类比商业 code agent（Cursor / Claude Code）的 "find relevant code"——
    模型先有个模糊意图（"处理登录失败重试的地方"），不必知道确切函数名也能定位。

实现：纯 Python、零依赖、只读。
    - 遍历工作区文本文件（复用 search 的噪声目录跳过与体积上限）；
    - 查询做轻量分词（英文词 + 中文二元文法），与文件做覆盖度打分排序；
    - 对每个命中文件截取「命中词最密集」的几行，带行号回显。
本工具只读、无副作用，可在一轮内与其它只读工具并行执行。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..errors import ToolError
from .base import ToolContext, ToolResult, tool_spec
from .search import _MAX_FILE_BYTES, _iter_files

__all__ = ["recall", "register"]

_ASCII_WORD = re.compile(r"[a-zA-Z0-9_]+")
_CJK = re.compile(r"[一-鿿]+")

# 单次最多扫描的文件数（超大仓库保护，避免一次调用扫爆）
_MAX_SCAN_FILES = 4000
# 每个命中文件回显的片段半窗（前后各几行）
_SNIPPET_HALF = 2
# 单文件片段最多回显多少字符（避免巨文件淹没上下文）
_MAX_SNIPPET_CHARS = 1200


def _tokenize(text: str) -> List[str]:
    """英文按词（≥2 字符）、中文按二元文法；返回查询/文件共用的 token 列表。"""
    tokens: List[str] = []
    low = text.lower()
    for m in _ASCII_WORD.finditer(low):
        w = m.group(0)
        if len(w) >= 2:
            tokens.append(w)
    for m in _CJK.finditer(text):
        run = m.group(0)
        if len(run) == 1:
            tokens.append(run)
        else:
            for i in range(len(run) - 1):
                tokens.append(run[i:i + 2])
    return tokens


def _score(text: str, qtokens: List[str]) -> Tuple[float, int]:
    """返回 (覆盖度 0..1, 命中总次数上限夹取)。覆盖度 = 命中的不同查询词占比。"""
    if not qtokens:
        return 0.0, 0
    low = text.lower()
    distinct = set(qtokens)
    present = 0
    total = 0
    for t in distinct:
        c = low.count(t)
        if c > 0:
            present += 1
            total += min(c, 100)
    coverage = present / len(distinct)
    return coverage, total


def _best_window(lines: List[str], qtokens: List[str]) -> Tuple[int, int]:
    """找命中词最密集的行，返回 [起始, 结束] 行号（0 基，含端点）。"""
    best = 0
    best_hits = -1
    for i, ln in enumerate(lines):
        low = ln.lower()
        hits = sum(1 for t in qtokens if t in low)
        if hits > best_hits:
            best_hits = hits
            best = i
    if best_hits <= 0:
        return 0, min(len(lines) - 1, 0)
    lo = max(0, best - _SNIPPET_HALF)
    hi = min(len(lines) - 1, best + _SNIPPET_HALF)
    return lo, hi


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(8192)
    except OSError:
        return True
    return b"\x00" in chunk


@tool_spec(
    name="recall",
    description=(
        "按「相关度」在工作区里找回**最相关的文件与片段**：给定一段想法或自然语言"
        "（如「处理登录失败重试的逻辑」「导出 CSV 的地方」），它会找出整体最相关的若干文件，"
        "并给出每个文件里最相关的几行。适合「知道要找什么、但不知道确切函数名/文件名」时先定位。"
        "若要精确正则匹配某几行，用 grep_search；若要按文件名找文件，用 find_files。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "查询意图，自然语言或关键词均可，如「登录失败重试」「导出 CSV」「缓存失效」",
            },
            "top_k": {
                "type": "integer",
                "description": "返回最相关的文件数，默认 5",
                "default": 5,
            },
            "path": {
                "type": "string",
                "description": "搜索起点，相对工作区，默认 '.'",
                "default": ".",
            },
        },
        "required": ["query"],
    },
    category="检索",
    when_not_to_use=(
        "已知确切函数名/字符串，直接用 grep_search（精确正则匹配、更快更准）。"
        "已知确切文件名，用 find_files。"
        "recall 是「模糊意图 → 哪些文件相关」的排序检索，不保证逐字命中；"
        "若工作区极大，recall 要通读全部文本文件，比 grep_search 慢，先用 path 收窄范围更高效。"
    ),
)
def recall(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = (args.get("query") or "").strip()
    if not query:
        raise ToolError("查询为空，请提供要检索的意图或关键词", tool="recall")
    top_k = max(1, int(args.get("top_k") or 5))
    root = ctx.resolve(args.get("path") or ".", must_exist=True)

    qtokens = _tokenize(query)
    if not qtokens:
        return ToolResult.failure(
            "查询无法提取有效检索词（至少包含一个英文词或中文字）",
            hint="换个更具体的说法，例如「登录失败重试」而非「重试」。",
        )

    ranked: List[Tuple[float, int, Path, str]] = []
    scanned = 0
    matched = 0
    for f in _iter_files(root, "*"):
        if scanned >= _MAX_SCAN_FILES:
            break
        try:
            if f.stat().st_size > _MAX_FILE_BYTES or _is_binary(f):
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        scanned += 1
        coverage, total = _score(text, qtokens)
        if coverage <= 0:
            continue
        matched += 1
        ranked.append((coverage, total, f, text))

    if not ranked:
        return ToolResult.success(
            f"未找到与「{query}」相关的文件（扫描 {scanned} 个文件）。"
            "换个更具体的说法，或先用 path 缩小范围。",
            meta={"matches": 0, "scanned": scanned},
        )

    # 主排序：覆盖度降序；覆盖度相同则命中总次数降序
    ranked.sort(key=lambda r: (r[0], r[1]), reverse=True)
    ranked = ranked[:top_k]

    blocks: List[str] = []
    for cov, _total, f, text in ranked:
        rel = ctx.guard.relpath(f)
        lines = text.splitlines()
        lo, hi = _best_window(lines, qtokens)
        buf = [f"── {rel}  （相关度 {cov:.2f}）──"]
        for j in range(lo, hi + 1):
            snippet = lines[j].rstrip()[:300]
            if len(snippet) >= 300:
                snippet += "…"
            buf.append(f"{j + 1:>5}  {snippet}")
        block = "\n".join(buf)
        if len(block) > _MAX_SNIPPET_CHARS:
            block = block[:_MAX_SNIPPET_CHARS] + "\n    …（片段过长已截断）"
        blocks.append(block)

    header = f"与「{query}」最相关的 {len(ranked)} 个文件（扫描 {scanned} 个，命中 {matched} 个）："
    return ToolResult.success(
        header + "\n" + "\n\n".join(blocks),
        meta={"matches": len(ranked), "scanned": scanned, "query": query},
    )


def register(registry) -> None:
    registry.register(recall)
