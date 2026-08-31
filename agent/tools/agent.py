"""
编排工具：delegate —— 派发一个受控子智能体去自主完成子任务，再把结论拿回来。

为什么要有它（贴近 Claude Code 的 Task / Codex 的子代理）：
    大任务拆成可独立验证的小块后，父任务的上下文压力会显著下降，子智能体也能并行探索、
    互不干扰。商用编程智能体普遍具备"派一个小弟去调研/写个独立模块"的能力。

安全边界（与考核要求一致，**全程零框架、零 SDK**，只是复用本项目自带的 AgentLoop）：
    1. 子智能体复用父任务的同一工作区、同一模型端点、同一套自写循环；
    2. 子智能体**不能再次派发**（注册表里剔除 delegate）也**不能向用户反问**（剔除 ask_user），
       只能自己解决或返回阶段性结论——从机制上杜绝无限递归；
    3. 子智能体步数预算被单独夹断（child_max_steps），不会失控空转；
    4. 子智能体关闭检查点/会话落盘写入（auto_checkpoint=False、session_log=None），
       不污染父任务的检查点与日志；其写操作同样受父任务的 permission_mode 约束；
    5. 子智能体失败**不会拖垮父任务**——异常被兜住，父任务可据此自行重试或降级。
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from .base import ToolContext, ToolRegistry, ToolResult, tool_spec

__all__ = ["delegate", "register"]

# 子智能体默认可用的工具（只读 + 安全执行类）。无论调用方传什么，
# delegate 与 ask_user 永远不在其列（防递归、无用户可问）。
_DEFAULT_CHILD_TOOLS = (
    "read_file", "list_dir", "grep_search", "find_files", "recall",
    "read_many_files", "run_command", "web_fetch", "diff", "summary",
)

# 子智能体**永远**需要 finish：它是子任务把结论回传给父任务的唯一出口，
# 没有它子智能体无法终止（会一直空转直到步数耗尽）。finish 之外、delegate/ask_user
# 之外的写/控制类工具默认不给，需要的话由调用方显式通过 tools 参数授权。
_ALWAYS_CHILD_TOOLS = frozenset({"finish"})

# 子智能体步数预算上限（无论父任务 max_steps 多大，子任务都不能无限制空转）
_CHILD_MAX_STEPS = 25

# 永远从子智能体注册表里剔除的工具（控制类，避免子任务失控）
_FORBIDDEN_IN_CHILD = frozenset({"delegate", "ask_user"})


@tool_spec(
    name="delegate",
    description=(
        "派发一个**受控的子智能体**去自主完成一段自包含的子任务，再把它的结论拿回来。\n"
        "适用于：探索型调研（\"读遍 src/ 找出所有用到 X 的地方并总结\"）、可独立验证的小任务"
        "（\"给 util.py 写单元测试并跑通\"）。子智能体与父任务共享同一份工作区，但拥有独立上下文、"
        "独立步数预算，且**不能再次派发子任务、也不能向用户反问**，只能自己解决或返回阶段性结论。\n"
        "返回子智能体最终给出的总结。用它把大任务拆成可并行/可独立验证的小块，"
        "能显著降低父任务的上下文压力。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "派给子智能体的、自包含的自然语言子任务描述（说清要做什么、产出是什么）",
            },
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "允许子智能体使用的工具名白名单（可选）。不填则默认只给只读 + 安全执行类工具"
                    "（read_file/list_dir/grep_search/find_files/recall/read_many_files/"
                    "run_command/web_fetch/diff/summary）。无论填什么，delegate 与 ask_user "
                    "永远不在子智能体可用之列（防递归、无用户可问）。"
                ),
            },
        },
        "required": ["task"],
    },
    category="编排",
    when_not_to_use=(
        "子任务太小、自己顺手就做了，不必开个子智能体（反而更慢、更费 token）。\n"
        "需要立刻拿到结果并据此继续写代码，且子任务与当前上下文强耦合时，直接自己调用工具更稳。\n"
        "涉及必须和用户确认的选择，不要用 delegate——子智能体无法向用户反问，会把问题又抛回给你。"
    ),
)
def delegate(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    loop = getattr(ctx, "loop", None)
    if loop is None:
        return ToolResult.failure(
            "delegate 需要在 AgentLoop 内运行（ctx.loop 缺失）",
            hint="这是内部装配问题，请确认 loop 在构造时把自身挂到 ctx.loop。",
        )

    task = (args.get("task") or "").strip()
    if len(task) < 4:
        return ToolResult.failure(
            "task 过短，请给子智能体一段自包含的子任务描述",
            hint="至少 4 个字符，说清要做什么、产出是什么。",
        )

    # ---- 1) 受控子工具集：永远保留 finish（出口），永远剔除 delegate（防递归）与 ask_user ----
    parent_registry: ToolRegistry = loop.registry
    wanted = set(args.get("tools") or [])
    if wanted:
        base = wanted | _ALWAYS_CHILD_TOOLS
    else:
        base = set(_DEFAULT_CHILD_TOOLS) | _ALWAYS_CHILD_TOOLS
    child_names = [n for n in parent_registry.names()
                   if n in base and n not in _FORBIDDEN_IN_CHILD]

    child_registry = ToolRegistry()
    for n in child_names:
        spec = parent_registry.get(n)
        if spec is not None:
            child_registry.register(spec)

    # ---- 2) 子智能体配置：共享工作区，但收紧步数预算、关掉检查点/会话落盘写入 ----
    child_cfg = copy.deepcopy(loop.config)
    child_cfg.max_steps = min(int(loop.config.max_steps), _CHILD_MAX_STEPS)
    child_cfg.auto_checkpoint = False       # 不写 .minicode/checkpoints，避免污染父任务检查点
    child_cfg.session_log = None            # 不写会话 JSONL
    child_cfg.plan_hint = False
    # 子智能体无交互环境：permission_mode 退化为 auto（与无头 run_command 一致），
    # 其写操作仍受沙箱与危险命令拦截约束。

    # ---- 3) 构建并运行子智能体（复用父任务的模型端点与历史机制）----
    # 延迟导入避免循环依赖：agent.tools 在 agent.loop 之前被导入。
    from ..loop import AgentLoop

    child = AgentLoop(child_cfg, loop.profile, loop.backend, child_registry, console=None)

    try:
        res = child.run(task)
    except Exception as exc:  # noqa: BLE001 —— 子智能体失败不应拖垮父任务
        return ToolResult.failure(
            f"子智能体执行失败：{type(exc).__name__}: {exc}",
            hint="父任务可据此自行重试或降级处理。",
            meta={"child_failed": True, "child_tools": child_names},
        )

    summary = res.answer or child.ctx.session.get("summary") or "（子智能体未返回总结）"
    meta = {
        "child_steps": res.steps,
        "child_tool_calls": res.tool_calls,
        "child_finish_reason": res.finish_reason,
        "child_tools": child_names,
    }
    header = (
        f"[子智能体结论 · 用了 {res.steps} 步 / {res.tool_calls} 次工具，"
        f"子任务状态：{res.finish_reason}]\n"
    )
    return ToolResult.success(header + summary, meta=meta)


def register(registry: ToolRegistry) -> None:
    registry.register(delegate)
