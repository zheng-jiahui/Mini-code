"""
任务清单工具：todo —— 让 agent 像人一样维护一份"待办清单"。

为什么需要它：
    商业编程智能体（Claude Code / Codex）普遍内置任务清单，原因是
    "列清楚要做什么"能显著减少 agent 在长任务里东改西改、漏掉步骤、
    或过早宣布完成。它和 plan 的区别在于：plan 只是"一次性记下来"，
    todo 是**可追踪状态**的清单——每做完一条就打勾，模型与用户随时能看清进度。

状态存放在 ctx.session["todo"]（随会话存活，压缩/续跑都不丢），不落盘文件，
因此与文件类工具严格隔离；它只改变 agent 的"内部认知"，不改变工作区。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from ..errors import ToolError
from .base import ToolContext, ToolResult, tool_spec

__all__ = ["todo", "register"]

_SESSION_KEY = "todo"
_STATUS_PENDING = "pending"
_STATUS_DONE = "done"


# ----------------------------------------------------------------------------
# 辅助（必须放在 @tool_spec 之前）
# ----------------------------------------------------------------------------
def _get_todos(ctx: ToolContext) -> List[Dict[str, Any]]:
    """取当前清单；首次访问时初始化为空列表。"""
    return ctx.session.setdefault(_SESSION_KEY, [])


def _render(todos: List[Dict[str, Any]]) -> str:
    """把清单渲染成带勾选框的文本。"""
    if not todos:
        return "（任务清单为空。用 action=add 添加第一条待办。）"
    lines = []
    for i, item in enumerate(todos, start=1):
        mark = "x" if item.get("status") == _STATUS_DONE else " "
        lines.append(f"{i}. [{mark}] {item.get('text', '')}")
    done = sum(1 for t in todos if t.get("status") == _STATUS_DONE)
    lines.append(f"—— 共 {len(todos)} 条，已完成 {done} 条 ——")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# todo
# ----------------------------------------------------------------------------
@tool_spec(
    name="todo",
    description=(
        "维护一份可追踪状态的任务清单（待办 / 已完成）。\n"
        "action=add 追加一条；action=list 查看全部；action=complete 标记某条完成；"
        "action=remove 删除某条；action=clear 清空。\n"
        "适合 3 步以上的多步骤任务：先 add 出计划，每完成一条就 complete 它，"
        "这样你和用户都能随时看清进度，避免漏做或过早收尾。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "操作类型",
                "enum": ["add", "list", "complete", "remove", "clear"],
            },
            "content": {"type": "string", "description": "action=add 时的任务描述（其它操作可省略）"},
            "id": {
                "type": "integer",
                "description": "action=complete / remove 时的条目序号（list 里显示的 1 起编号）",
            },
        },
        "required": ["action"],
    },
    category="控制",
    when_not_to_use=(
        "三五步就能做完的小任务不要维护清单——直接用 plan 或直接动手，列清单本身也要花一轮。"
        "清单不是许愿：列完要真正去调用工具完成每一条，并记得 complete，别只列不做。"
    ),
)
def todo(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    action = (args.get("action") or "").strip().lower()
    todos = _get_todos(ctx)

    if action == "list":
        return ToolResult.success(_render(todos), meta={"count": len(todos)})

    if action == "clear":
        removed = len(todos)
        todos.clear()
        return ToolResult.success(f"已清空任务清单（原 {removed} 条）。", meta={"removed": removed})

    if action == "add":
        content = (args.get("content") or "").strip()
        if not content:
            raise ToolError("action=add 时必须提供 content（任务描述）", tool="todo")
        todos.append({"text": content, "status": _STATUS_PENDING, "created": time.time()})
        return ToolResult.success(
            f"已添加第 {len(todos)} 条：{content}\n" + _render(todos),
            meta={"count": len(todos)},
        )

    if action in ("complete", "remove"):
        raw_id = args.get("id")
        if raw_id is None:
            raise ToolError(f"action={action} 时必须提供 id（list 里的序号）", tool="todo")
        try:
            idx = int(raw_id) - 1
        except (TypeError, ValueError):
            raise ToolError(f"id 必须是数字，收到 {raw_id!r}", tool="todo")
        if idx < 0 or idx >= len(todos):
            raise ToolError(
                f"id={raw_id} 超出范围（当前共 {len(todos)} 条）",
                tool="todo",
                hint="先调用 action=list 查看有效序号。",
            )
        if action == "complete":
            todos[idx]["status"] = _STATUS_DONE
            return ToolResult.success(
                f"已标记完成：{todos[idx]['text']}\n" + _render(todos),
                meta={"count": len(todos)},
            )
        removed = todos.pop(idx)
        return ToolResult.success(
            f"已删除：{removed.get('text', '')}\n" + _render(todos),
            meta={"count": len(todos)},
        )

    raise ToolError(
        f"未知 action={action!r}", tool="todo",
        hint="可选：add / list / complete / remove / clear。",
    )


def register(registry) -> None:
    registry.register(todo)
