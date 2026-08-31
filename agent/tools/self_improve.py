"""
self_improve 工具：把「自我改进钩子」暴露给模型。

两种触发路径：
1. 自动（主循环在任务结束时调用，受 config.auto_improve 控制）：从本次会话信号沉淀经验。
2. 手动（本工具）：agent 可主动 `digest` 把当前信号落盘、`list` 查看已沉淀经验、
   `forget` 清空自动经验（保留 memory 工具写的其它内容）。
"""

from __future__ import annotations

from typing import Any, Dict

from ..self_improve import (
    SessionSignal,
    derive_lessons,
    forget_lessons,
    read_lessons,
    record_lessons,
)
from .base import ToolContext, ToolResult, tool_spec

__all__ = ["self_improve", "register"]


# ----------------------------------------------------------------------------
# 辅助（必须放在 @tool_spec 之前）
# ----------------------------------------------------------------------------
def _build_signal_from_ctx(ctx: ToolContext) -> SessionSignal:
    """优先用主循环暂存的完整信号；否则从 ctx 上可拿到的计数器重建。"""
    stashed = ctx.session.get("_last_signal")
    if isinstance(stashed, dict):
        return SessionSignal(
            repair_rounds=int(stashed.get("repair_rounds", 0) or 0),
            tool_errors=int(stashed.get("tool_errors", 0) or 0),
            permission_denied=int(stashed.get("permission_denied", 0) or 0),
            empty_responses=int(stashed.get("empty_responses", 0) or 0),
            finish_blocked=int(stashed.get("finish_blocked", 0) or 0),
            compactions=int(stashed.get("compactions", 0) or 0),
            aborted=bool(stashed.get("aborted", False)),
            llm_error=bool(stashed.get("llm_error", False)),
        )
    # 退路：只有局部计数器可用（repair/compaction 需主循环聚合，这里拿不到就留 0）
    return SessionSignal(
        tool_errors=int(ctx.session.get("tool_errors", 0) or 0),
        permission_denied=int(ctx.session.get("permission_denied", 0) or 0),
        empty_responses=int(ctx.session.get("empty_responses", 0) or 0),
        finish_blocked=int(ctx.session.get("finish_blocked", 0) or 0),
    )


# ----------------------------------------------------------------------------
# self_improve
# ----------------------------------------------------------------------------
@tool_spec(
    name="self_improve",
    description=(
        "自我改进：把本次任务的失败/修复经验沉淀到项目记忆 `.minicode/memory.md`，"
        "下次启动自动注入 system 提示词，形成「失败→记忆→下次更聪明」的闭环。\n"
        "action=digest 立刻把当前会话信号整理成经验并落盘；action=list 查看已沉淀的自动经验；"
        "action=forget 清空自动经验（保留 memory 工具写的其它内容）。\n"
        "任务正常结束时主循环已自动 digest，无需每次手动调用；本工具用于中途主动沉淀或复盘。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "操作类型",
                "enum": ["digest", "list", "forget"],
            },
        },
        "required": ["action"],
    },
    category="记忆",
    when_not_to_use=(
        "无需每次都调用：任务结束主循环已自动 digest。只有想中途主动沉淀一条经验、"
        "或复盘/清空时才用。不要把整段日志灌进去——它应只沉淀可泛化的结论。"
    ),
)
def self_improve(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    action = (args.get("action") or "").strip().lower()
    root = getattr(ctx.config, "workspace_root", None) or ctx.workspace

    if action == "list":
        lessons = read_lessons(root)
        if not lessons:
            return ToolResult.success("（暂无自动沉淀的经验。任务失败后主循环会自动 digest。）")
        body = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lessons))
        return ToolResult.success(f"已沉淀的自动经验（{len(lessons)} 条）：\n{body}")

    if action == "forget":
        removed = forget_lessons(root)
        return ToolResult.success(
            f"已清空 {removed} 条自动经验（memory 工具写的其它项目记忆保留）。"
            if removed else "（没有可清空的自动经验。）",
            meta={"removed": removed},
        )

    if action == "digest":
        signal = _build_signal_from_ctx(ctx)
        lessons = derive_lessons(signal)
        if not lessons:
            return ToolResult.success(
                "本次会话没有触发需要沉淀的信号（无失败/修复/中断/压缩/权限拒绝），无需写入。",
                meta={"added": 0},
            )
        added = record_lessons(root, lessons)
        if added == 0:
            return ToolResult.success("经验已存在，无需重复写入。", meta={"added": 0})
        body = "\n".join(f"- {t}" for t in lessons[:added])
        return ToolResult.success(
            f"已沉淀 {added} 条经验到项目记忆（.minicode/memory.md，下次启动注入提示词）：\n{body}",
            meta={"added": added},
        )

    return ToolResult.failure(
        f"未知 action={action!r}（可选 digest / list / forget）",
        hint="用 action=digest 落盘、action=list 查看、action=forget 清空。",
    )


def register(registry) -> None:
    registry.register(self_improve)
