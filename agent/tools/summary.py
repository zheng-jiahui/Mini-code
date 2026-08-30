"""
汇报工具：summary —— 把「本次会话改了哪些文件、怎么改的」整理成一份可粘贴的概览。

与 /diff、/stats 的分工：
    - /diff：逐个文件的 unified diff（看具体改了哪几行）；
    - /stats：质量指标（成功率、返工、耗时分布）；
    - summary：一份**叙述式概览**——哪些文件被新建/修改/删除、各加减了多少行、本次任务是什么，
      适合直接贴进 PR 描述、提交说明或交给用户做交接。只读、无副作用。
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import ToolContext, ToolResult, tool_spec

__all__ = ["summary", "register"]

_KIND_LABEL = {
    "write": "写入", "edit": "编辑", "replace": "替换", "replace_in_files": "跨文件替换",
    "delete": "删除", "move": "移动", "copy": "复制", "run_command": "执行命令",
}


def _line_delta(change: Dict[str, Any]) -> str:
    before = change.get("before")
    after = change.get("after")
    captured = change.get("captured", False)
    if not captured:
        return "（未采集原文，无行数）"
    b_lines = (before.count("\n") + 1) if before else 0
    a_lines = (after.count("\n") + 1) if after else 0
    if before is None and after is not None:
        return f"+{a_lines}（新建）"
    if after is None and before is not None:
        return f"-{b_lines}（删除）"
    delta = a_lines - b_lines
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta}（{b_lines}→{a_lines} 行）"


@tool_spec(
    name="summary",
    description=(
        "整理本次会话的改动概览：哪些文件被新建/修改/删除、各加减多少行、本次任务是什么。"
        "适合直接贴进 PR 描述或提交说明、或交给用户做交接。只读、无副作用。"
        "要看具体改了哪几行用 /diff；要看质量指标（成功率/耗时）用 /stats。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "verbose": {
                "type": "boolean",
                "description": "true 时逐文件列出（含行数变化），false 时只给汇总计数，默认 true",
                "default": True,
            },
        },
        "required": [],
    },
    category="汇报",
    when_not_to_use=(
        "只想看某个文件具体改了哪几行，用 /diff 更直接；想看成功率/返工/耗时等指标用 /stats。"
        "summary 是「这次会话整体改了什么」的叙述式概览，不是逐行 diff，也不是质量指标。"
    ),
)
def summary(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    verbose = bool(args.get("verbose", True))
    changes: List[Dict[str, Any]] = ctx.session.get("changes", []) or []
    task = ctx.session.get("task") or ctx.session.get("task_name") or "（未命名任务）"

    if not changes:
        return ToolResult.success(
            f"本次会话（任务：{task}）还没有任何改动记录。"
            "先动手改文件、跑命令，或调用 finish 收尾后这里才有内容。",
            meta={"changed_files": 0, "changes": 0},
        )

    # 按文件聚合（同文件多次改动合并展示，但保留每次动作）
    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for ch in changes:
        rel = ch.get("path") or ch.get("detail") or "（未命名）"
        by_file.setdefault(rel, []).append(ch)

    file_lines: List[str] = []
    for rel, chs in by_file.items():
        kinds = "/".join(_KIND_LABEL.get(c.get("kind", ""), c.get("kind", "")) for c in chs)
        if verbose:
            delta = _line_delta(chs[-1])  # 取该文件最后一次改动后的净变化
            file_lines.append(f"- {rel}  [{kinds}]  {delta}")
        else:
            file_lines.append(f"- {rel}  [{kinds}]")

    # 动作类型计数
    kind_counts: Dict[str, int] = {}
    for ch in changes:
        k = ch.get("kind", "other")
        kind_counts[k] = kind_counts.get(k, 0) + 1

    kind_summary = "，".join(f"{_KIND_LABEL.get(k, k)} {n}" for k, n in sorted(kind_counts.items()))
    header = f"本次会话改动概览（任务：{task}）\n共改动 {len(by_file)} 个文件，{len(changes)} 次动作：{kind_summary}"
    body = "\n".join(file_lines) if file_lines else "（无文件改动）"
    return ToolResult.success(
        f"{header}\n\n{body}",
        meta={"changed_files": len(by_file), "changes": len(changes), "kinds": kind_counts},
    )


def register(registry) -> None:
    registry.register(summary)
