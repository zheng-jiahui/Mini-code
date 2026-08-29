"""
自修复闭环的「回滚」工具：把当前任务目录恢复到上一次能跑通的快照。

为什么值得做：每次任务结束都会把完整代码归档到 .agent_backups/{任务名}_{时间戳}_第N次/，
既然保留了每一次的完整快照，「回到上一个已知可用状态」就是免费的——只要把最新那份拷回来。
这是 V2「连续修不好就回滚」的物理支撑。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

from .base import ToolContext, ToolResult, tool_spec

__all__ = ["rollback", "register"]


@tool_spec(
    name="rollback",
    description=(
        "把当前任务目录恢复到上一次能跑通的快照（.agent_backups 里最新的 `{任务名}_*` 归档）。\n"
        "当连续多次修复都失败、代码已被改乱时，用它回到已知可用的状态，再从那里重新分析。\n"
        "⚠️ 会覆盖当前任务目录里同名文件。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "可选：指定要恢复到的快照名（不含路径）；不填则用最新的一份",
                "default": "",
            },
        },
    },
    category="控制",
)
def rollback(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    task_name = ctx.session.get("task_name") or ctx.workspace.name
    backup_dir = getattr(ctx.config, "backup_dir", None) or ".agent_backups"
    # ctx.workspace 是 workplace/{任务名}/，所以备份根 = 其两级父目录下的 backup_dir
    backup_root = ctx.workspace.parent.parent / backup_dir
    if not backup_root.exists():
        return ToolResult.failure(
            "还没有任何备份快照，无法回滚",
            hint="先完成过至少一次任务（且改动过文件）才会有 .agent_backups 归档。",
        )

    snaps = sorted(
        (p for p in backup_root.iterdir() if p.is_dir() and p.name.startswith(f"{task_name}_")),
        key=lambda p: p.name, reverse=True,
    )
    if not snaps:
        return ToolResult.failure(
            f"没有任务「{task_name}」的快照可回滚",
            hint="可能该任务尚未生成过备份，或任务名不匹配。",
        )

    target = snaps[0]
    if args.get("target"):
        cand = backup_root / str(args["target"])
        if cand in snaps:
            target = cand
        else:
            return ToolResult.failure(f"找不到指定的快照：{args['target']}")

    dest = ctx.workspace
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in target.iterdir():
        if item.name.startswith("."):   # 跳过 .agent_sessions 等内部目录
            continue
        tgt = dest / item.name
        if item.is_dir():
            shutil.rmtree(tgt, ignore_errors=True)
            shutil.copytree(item, tgt)
        else:
            shutil.copy2(item, tgt)
        count += 1

    ctx.record_change("rollback", target.name)
    return ToolResult.success(
        f"已回滚到快照 `{target.name}`，恢复了 {count} 个项目到 `{ctx.guard.relpath(dest)}`。",
        meta={"rollback_to": target.name, "restored": count},
    )


def register(registry) -> None:
    registry.register(rollback)
