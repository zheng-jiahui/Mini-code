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
            "files": {
                "type": "array",
                "description": "可选：只恢复这些文件（相对任务目录，如 [\"calc.py\"]）；不填则恢复整个快照",
                "items": {"type": "string"},
                "default": [],
            },
        },
    },
    category="控制",
    when_not_to_use=(
        "只是一次运行失败、且你已经看清报错在哪一行时，直接 edit_block 定向修，"
        "不要回滚——回滚会丢掉之后的全部改动。它是「修不动了」的退路，不是常规手段；"
        "只在修复预算快耗尽、代码已被改乱时用。"
    ),
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

    only_files = [str(f) for f in (args.get("files") or [])]
    dest = ctx.workspace
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    if only_files:
        # 单文件级回退：只从快照里挑指定的文件恢复
        for rel in only_files:
            src = (target / rel).resolve()
            # 防止越界读取快照目录之外的文件
            if target != src and target not in src.parents:
                return ToolResult.failure(f"非法的文件路径：{rel}", hint="请使用相对任务目录的文件名。")
            if not src.exists():
                return ToolResult.failure(f"快照中不存在文件：{rel}")
            tgt = dest / rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.rmtree(tgt, ignore_errors=True)
                shutil.copytree(src, tgt)
            else:
                shutil.copy2(src, tgt)
            count += 1
        summary = f"已从快照 `{target.name}` 恢复 {count} 个指定文件"
    else:
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
        summary = f"已回滚到快照 `{target.name}`，恢复了 {count} 个项目到 `{ctx.guard.relpath(dest)}`"

    ctx.record_change("rollback", target.name)
    return ToolResult.success(
        summary + "。",
        meta={"rollback_to": target.name, "restored": count, "files": only_files},
    )


def register(registry) -> None:
    registry.register(rollback)
