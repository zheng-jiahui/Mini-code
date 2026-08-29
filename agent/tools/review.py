"""
可审阅性：把本次会话的改动渲染成 unified diff，供模型自查、也供用户通过 /diff 查看。

设计边界：
    · 改动的 before/after 内容由 ToolContext.record_change(kind, detail, before=, after=) 采集，
      本模块只负责"渲染"，不碰文件系统、不调模型。
    · 大文件（采集时已跳过 before/after）只列变更摘要，不生成 diff，避免撑爆上下文。
    · 既是给用户看的 /diff，也是给模型调用的 diff 工具——同一份渲染，两种入口。
"""

from __future__ import annotations

import difflib
from typing import Any, Dict, List, Sequence

from .base import ToolContext, ToolResult, tool_spec

__all__ = ["build_diff", "diff", "register"]


def build_diff(changes: Sequence[Dict[str, Any]]) -> str:
    """把本次会话的 changes 渲染成 unified diff 文本。

    每个有 before/after 的变更生成一段 unified diff；新建文件（before=None）视为全量新增；
    无前后文（超大数据未采集）的只列摘要。
    """
    if not changes:
        return "（本次会话未修改任何文件）"

    parts: List[str] = []
    for c in changes:
        kind = c.get("kind", "change")
        detail = c.get("detail", "")
        name = c.get("path") or detail
        before = c.get("before")
        after = c.get("after")

        if before is None and after is None:
            parts.append(f"[{kind}] {detail}（改动过大，未生成 diff）")
            continue
        if after is None:
            parts.append(f"[{kind}] {detail}（无 after 内容，未生成 diff）")
            continue

        b_lines = (before or "").splitlines()
        a_lines = after.splitlines()
        diff = list(difflib.unified_diff(
            b_lines, a_lines,
            fromfile=f"a/{name}", tofile=f"b/{name}", lineterm="",
        ))
        if diff:
            parts.append("\n".join(diff))
        else:
            parts.append(f"[{kind}] {name}（内容无变化）")

    return "\n\n".join(parts) if parts else "（本次会话未修改任何文件）"


@tool_spec(
    name="diff",
    description=(
        "查看本次会话改了哪些文件、具体改了什么（unified diff 格式）。\n"
        "在调用 finish 之前用它自查改动是否符合预期，避免把意外修改一并提交。"
    ),
    parameters={"type": "object", "properties": {}},
    category="控制",
)
def diff(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    text = build_diff(ctx.session.get("changes", []))
    return ToolResult.success(text, meta={"changes": len(ctx.session.get("changes", []))})


def register(registry) -> None:
    registry.register(diff)
