"""
元工具：finish / ask_user —— 它们不产生副作用，但决定了循环的"退出语义"。

为什么需要显式的 finish？
    只靠"模型不再调用工具"来判断结束并不可靠：模型可能因为上下文变长而中途停手，
    也可能在失败后沉默。显式工具让"结束"成为一次可被观测、可被记录的决策。
"""

from __future__ import annotations

from typing import Any, Dict

from .base import ToolContext, ToolResult, tool_spec

__all__ = ["finish", "ask_user", "plan", "todo", "register", "FINISH_SENTINEL"]

FINISH_SENTINEL = "__finish__"


@tool_spec(
    name="finish",
    description=(
        "结束任务并给出总结。**当你已完成任务、或确认无法继续时必须调用它**。\n"
        "summary 需包含：① 修改了哪些文件；② 如何验证的（跑了什么命令、结果如何）；\n"
        "③ 遗留问题或需要用户注意的地方。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "给用户的最终总结"},
        },
        "required": ["summary"],
    },
    # 注意：finish 必须"可见"——它要出现在工具清单与 function schema 里，模型才调得到；
    # 只是主循环会对它做特殊处理（直接终止，不把回执喂回模型）。
    hidden=False,
    category="控制",
    when_not_to_use=(
        "改动了文件却一次都没验证过时不要收尾——先 run_command 跑一遍；"
        "也不要在报错还没定位清楚时用它糊弄过去，那不是完成，是逃避。"
    ),
)
def finish(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    summary = (args.get("summary") or "").strip() or "（模型未给出总结）"
    ctx.session["finished"] = True
    ctx.session["summary"] = summary
    return ToolResult.success(FINISH_SENTINEL, meta={"finish": True})


@tool_spec(
    name="ask_user",
    description=(
        "当需求存在歧义、缺少关键信息（如选哪个框架、用哪个端口、是否需要联网）时，"
        "向用户提问并等待回答。不要把问题憋在心里瞎猜。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "要问用户的问题，尽量给出选项"},
        },
        "required": ["question"],
    },
    category="控制",
    when_not_to_use=(
        "能通过 read_file/grep_search 自己查清的（现有代码怎么写的、用的什么框架）"
        "就别问，自己看。也不要一次问一串问题——先问最卡住的那一个；"
        "已在计划里确认过的决定不要重复问。"
    ),
)
def ask_user(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    question = (args.get("question") or "").strip()
    if not question:
        return ToolResult.failure("问题不能为空", hint="请明确你要问什么。")
    if ctx.console is None:  # 非交互环境（如测试）
        return ToolResult.success("（非交互环境，用户无法回答，请自行做出合理假设并继续）")
    answer = ctx.console.ask(question)
    return ToolResult.success(f"用户回答：{answer or '（未回答，请自行合理假设并继续）'}")


@tool_spec(
    name="plan",
    description=(
        "对于较复杂的任务，先把分步计划写在这里（例如：1. 读 X 弄清现状；2. 用 edit_block 改 Y；"
        "3. 跑测试验证）。记录后请按计划在后续步骤执行，便于用户审阅、也让你自己对齐目标。\n"
        "简单任务（改几行、写个小脚本）不必调用本工具。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "分步计划，每条是一句可执行的操作描述",
            },
        },
        "required": ["steps"],
    },
    category="控制",
    when_not_to_use=(
        "三五步就能做完的小任务不要先列计划，直接做（列计划本身也要花一轮）。"
        "计划不是许愿——步骤要能对应到具体工具调用；也不要列完就当执行完了。"
    ),
)
def plan(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    steps = args.get("steps") or []
    if not isinstance(steps, list) or not steps:
        return ToolResult.failure("计划不能为空", hint="请列出 2-5 条分步计划。")
    steps = [str(s).strip() for s in steps if str(s).strip()]
    if not steps:
        return ToolResult.failure("计划不能为空", hint="请列出具体的步骤。")
    text = "已记录计划，将按此执行：\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    ctx.session["plan"] = steps
    return ToolResult.success(text, meta={"steps": len(steps)})


_STATUS_ORDER = {"pending": 0, "in_progress": 1, "completed": 2}


def _render_todos(todos: list) -> str:
    if not todos:
        return "（任务清单为空，调用 todo(action=\"add\", items=[...]) 添加）"
    lines = []
    for t in todos:
        mark = {"pending": "⬜", "in_progress": "🔵", "completed": "✅"}.get(t["status"], "⬜")
        lines.append(f"{t['id']}. {mark} [{t['status']}] {t['text']}")
    return "当前任务清单：\n" + "\n".join(lines)


@tool_spec(
    name="todo",
    description=(
        "维护一份跨轮持久的任务清单（带进度状态），适合多步骤、需要中途对齐目标的任务。\n"
        "action 取值：add（追加若干待办）、update（把某项标记为 in_progress/completed）、"
        "list（查看）、clear（清空）。\n"
        "比一次性的 plan 更实用：你能随时把第 N 项标记为进行中/已完成，让模型和自我都看清进度。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "操作：add | update | list | clear，默认 list",
                "default": "list",
            },
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "action=add 时要追加的待办文本列表",
            },
            "id": {
                "type": "integer",
                "description": "action=update 时要更新的任务编号",
            },
            "status": {
                "type": "string",
                "description": "action=update 时设成的新状态：in_progress | completed（或 pending 重新打开）",
            },
        },
        "required": [],
    },
    category="控制",
    when_not_to_use=(
        "三五步就能做完的小任务不必建清单，直接做更省事；"
        "清单不是许愿——加进去的项要能对应到具体工具调用，并随进度真正去 update 状态，"
        "别列完就再也不看。只问一个问题用 ask_user，别往清单里塞。"
    ),
)
def todo(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    action = (args.get("action") or "list").strip().lower()
    todos: list = ctx.session.setdefault("todos", [])

    if action == "add":
        items = args.get("items") or []
        if not isinstance(items, list) or not items:
            return ToolResult.failure("action=add 时必须提供 items 列表", hint="如 todo(action=\"add\", items=[\"读现状\", \"改 X\"])")
        added = 0
        for it in items:
            s = str(it).strip()
            if not s:
                continue
            todos.append({"id": len(todos) + 1, "text": s, "status": "pending"})
            added += 1
        if added == 0:
            return ToolResult.failure("items 为空或全部为空白，未添加任何待办")
        return ToolResult.success(
            f"已添加 {added} 项，清单共 {len(todos)} 项。\n" + _render_todos(todos),
            meta={"count": len(todos)},
        )

    if action == "update":
        tid = args.get("id")
        status = (args.get("status") or "").strip().lower()
        if not isinstance(tid, int) or tid <= 0:
            return ToolResult.failure("action=update 时必须提供合法的 id（正整数）")
        if status not in _STATUS_ORDER:
            return ToolResult.failure(
                f"status 必须是 in_progress / completed / pending 之一，收到 {status!r}",
                hint="例如把第 2 项标记为进行中：todo(action=\"update\", id=2, status=\"in_progress\")",
            )
        target = next((t for t in todos if t["id"] == tid), None)
        if target is None:
            return ToolResult.failure(f"找不到编号为 {tid} 的任务", hint="先用 todo(action=\"list\") 查看有效编号。")
        target["status"] = status
        return ToolResult.success(
            f"已将第 {tid} 项标记为 {status}。\n" + _render_todos(todos),
            meta={"updated": tid, "status": status},
        )

    if action == "clear":
        before = len(todos)
        todos.clear()
        return ToolResult.success(f"已清空任务清单（原 {before} 项）。", meta={"cleared": before})

    # 默认 list
    return ToolResult.success(_render_todos(todos), meta={"count": len(todos)})


def register(registry) -> None:
    registry.register_many([finish, ask_user, plan, todo])
