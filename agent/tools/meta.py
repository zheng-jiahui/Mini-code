"""
元工具：finish / ask_user —— 它们不产生副作用，但决定了循环的"退出语义"。

为什么需要显式的 finish？
    只靠"模型不再调用工具"来判断结束并不可靠：模型可能因为上下文变长而中途停手，
    也可能在失败后沉默。显式工具让"结束"成为一次可被观测、可被记录的决策。
"""

from __future__ import annotations

from typing import Any, Dict

from .base import ToolContext, ToolResult, tool_spec

__all__ = ["finish", "ask_user", "plan", "register", "FINISH_SENTINEL"]

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


def register(registry) -> None:
    registry.register_many([finish, ask_user, plan])
