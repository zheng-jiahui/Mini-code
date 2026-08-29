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

from .base import DIFF_CAPTURE_CAP, ToolContext, ToolResult, tool_spec

__all__ = ["build_diff", "diff", "register"]


def build_diff(changes: Sequence[Dict[str, Any]]) -> str:
    """把本次会话的 changes 渲染成 unified diff 文本。

    渲染规则：
      · 只有**文件内容类**改动（带 `captured=True`）才生成 diff —— run_command、
        rollback 这类操作不是文件改动，混进来只会产生误导性的"未生成 diff"噪声。
      · 新建文件（`before` 为 None 但已采集）显示为全量新增，这是最该被审阅的一类改动。
      · 内容过大没采集的，只在末尾列个"另有 N 个未生成 diff"，不占正文。
    """
    if not changes:
        return "（本次会话未修改任何文件）"

    parts: List[str] = []
    skipped: List[str] = []
    for c in changes:
        if not c.get("captured"):
            # 只把"文件改动但太大没采集"的列出；run_command 等无 path 的操作不提
            if c.get("path"):
                skipped.append(c.get("path") or c.get("detail", "?"))
            continue

        kind = c.get("kind", "change")
        name = c.get("path") or c.get("detail", "?")
        before = c.get("before")
        after = c.get("after")

        if after is None:
            parts.append(f"[{kind}] {name}（无改动后内容，未生成 diff）")
            continue

        diff = list(difflib.unified_diff(
            (before or "").splitlines(), after.splitlines(),
            fromfile=f"a/{name}", tofile=f"b/{name}", lineterm="",
        ))
        if diff:
            parts.append("\n".join(diff))
        else:
            parts.append(f"[{kind}] {name}（内容无变化）")

    if not parts and not skipped:
        return "（本次会话没有产生文件内容的改动）"

    out: List[str] = [f"本次会话共 {len(parts)} 个文件变更："]
    out.extend(parts)
    if skipped:
        out.append(f"另有 {len(skipped)} 个变更内容过大（>{DIFF_CAPTURE_CAP} 字符），"
                   f"未生成 diff：{'、'.join(skipped)}")
    return "\n\n".join(out)


@tool_spec(
    name="diff",
    description=(
        "查看本次会话改了哪些文件、具体改了什么（unified diff 格式）。\n"
        "在调用 finish 之前用它自查改动是否符合预期，避免把意外修改一并提交。"
    ),
    parameters={"type": "object", "properties": {}},
    category="控制",
    when_not_to_use=(
        "要看**某个文件现在的内容**用 read_file，本工具只反映本次会话的改动，"
        "不读磁盘、也不显示未被本会话碰过的文件。别在每轮结束都调一次，"
        "在 finish 前调一次自查即可。"
    ),
)
def diff(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    text = build_diff(ctx.session.get("changes", []))
    return ToolResult.success(text, meta={"changes": len(ctx.session.get("changes", []))})


def register(registry) -> None:
    registry.register(diff)
