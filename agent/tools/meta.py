"""
元工具：finish / ask_user —— 它们不产生副作用，但决定了循环的"退出语义"。

为什么需要显式的 finish？
    只靠"模型不再调用工具"来判断结束并不可靠：模型可能因为上下文变长而中途停手，
    也可能在失败后沉默。显式工具让"结束"成为一次可被观测、可被记录的决策。
"""

from __future__ import annotations

from typing import Any, Dict

from .base import ToolContext, ToolResult, tool_spec

__all__ = ["finish", "ask_user", "register", "FINISH_SENTINEL"]

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
)
def ask_user(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    question = (args.get("question") or "").strip()
    if not question:
        return ToolResult.failure("问题不能为空", hint="请明确你要问什么。")
    if ctx.console is None:  # 非交互环境（如测试）
        return ToolResult.success("（非交互环境，用户无法回答，请自行做出合理假设并继续）")
    answer = ctx.console.ask(question)
    return ToolResult.success(f"用户回答：{answer or '（未回答，请自行合理假设并继续）'}")


def register(registry) -> None:
    registry.register_many([finish, ask_user])
