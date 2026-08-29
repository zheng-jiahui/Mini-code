"""
工具包：集中注册，供主循环一次性装配。

新增工具的步骤：
    1. 在 filesystem.py / shell.py / search.py / meta.py 或新模块里用 @tool_spec 定义；
    2. 在该模块的 register(registry) 中加入；
    3. 在 build_default_registry() 中调用该 register。
无需改动主循环——这就是注册表模式的好处。
"""

from __future__ import annotations

from typing import Optional

from ..config import AgentConfig
from ..security import CommandGuard, PathGuard
from .base import ToolContext, ToolRegistry

__all__ = ["build_default_registry", "build_tool_context"]


def build_default_registry(include: Optional[list] = None) -> ToolRegistry:
    """装配默认工具集。

    Args:
        include: 可选白名单，仅注册指定类别，如 ["文件", "执行"]。
    """
    from . import filesystem, meta, repair, search, shell

    registry = ToolRegistry()
    modules = [filesystem, search, shell, meta, repair]
    for mod in modules:
        mod.register(registry)

    if include:
        wanted = set(include)
        for name in list(registry.names()):
            spec = registry.get(name)
            if spec and spec.category not in wanted:
                registry._tools.pop(name, None)  # 仅在装配期使用内部字典，逻辑简单可控
    return registry


def build_tool_context(config: AgentConfig, console=None, session: Optional[dict] = None) -> ToolContext:
    """构造工具执行上下文（路径沙箱 + 命令审查 + 变更记录）。"""
    guard = PathGuard(
        workspace=config.resolved_workspace(),
        enabled=config.restrict_to_workspace,
    )
    ctx = ToolContext(
        workspace=config.resolved_workspace(),
        guard=guard,
        config=config,
        console=console,
        session=session or {},
    )
    ctx.session["command_guard"] = CommandGuard(config.dangerous_commands, config.command_policy)
    return ctx
