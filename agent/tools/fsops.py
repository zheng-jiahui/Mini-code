"""
文件系统增删工具：move_file / copy_file / delete。

为什么补这一组：
    已有 read / write / edit_block / apply_patch，但缺「移动 / 复制 / 删除」——
    商业 code agent 都能安全地增删文件。设计上贯彻两条既有原则：
    · 全部路径先过 PathGuard 沙箱，越界一律拒绝；
    · delete 在真正删除前先备份到 .agent_backups（删文件可恢复，不靠 rm 不可逆转）。
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ..errors import ToolError
from .base import ToolContext, ToolResult, tool_spec
from .filesystem import _backup, _backup_root, _rel

__all__ = ["move_file", "copy_file", "delete", "register"]

_MAX_DIR_BACKUP_BYTES = 5_000_000  # 超过此体量的目录删除前不备份（避免塞爆备份区）


# ----------------------------------------------------------------------------
# move_file
# ----------------------------------------------------------------------------
@tool_spec(
    name="move_file",
    description=(
        "移动 / 重命名文件或目录（在沙箱内）。默认目标已存在则拒绝，避免误覆盖；"
        "确需覆盖时传 overwrite=true（会先备份目标）。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "src": {"type": "string", "description": "源路径，相对工作区"},
            "dst": {"type": "string", "description": "目标路径，相对工作区"},
            "overwrite": {"type": "boolean", "description": "目标已存在时是否覆盖，默认 false", "default": False},
        },
        "required": ["src", "dst"],
    },
    category="文件",
    when_not_to_use=(
        "只是想改文件内容用 edit_block / write_file，move 只挪位置不改动内容。"
        "跨工作区移动会被沙箱拒绝，别尝试。"
    ),
)
def move_file(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    src = ctx.resolve(args["src"], must_exist=True)
    dst = ctx.resolve(args["dst"])
    if src.resolve() == dst.resolve():
        raise ToolError("源和目标相同", tool="move_file")
    if dst.exists() and not args.get("overwrite"):
        raise ToolError(f"目标已存在：{args['dst']}（要覆盖请传 overwrite=true）", tool="move_file")

    backup = ""
    if dst.exists() and args.get("overwrite"):
        backup = _backup(ctx, dst)
        if dst.is_dir():
            shutil.rmtree(str(dst))
        else:
            dst.unlink()
    try:
        shutil.move(str(src), str(dst))
    except OSError as exc:
        raise ToolError(f"移动失败：{exc}", tool="move_file") from exc

    ctx.record_change("move", f"{_rel(ctx, src)} -> {_rel(ctx, dst)}", path=_rel(ctx, dst))
    detail = f"已移动 {_rel(ctx, src)} -> {_rel(ctx, dst)}"
    if backup:
        detail += f"\n原目标已备份至 {backup}"
    return ToolResult.success(detail, meta={"src": str(src), "dst": str(dst)})


# ----------------------------------------------------------------------------
# copy_file
# ----------------------------------------------------------------------------
@tool_spec(
    name="copy_file",
    description=(
        "复制文件或目录到目标（在沙箱内）。默认目标已存在则拒绝；"
        "确需覆盖时传 overwrite=true（会先备份目标）。复制目录会递归。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "src": {"type": "string", "description": "源路径，相对工作区（文件或目录）"},
            "dst": {"type": "string", "description": "目标路径，相对工作区"},
            "overwrite": {"type": "boolean", "description": "目标已存在时是否覆盖，默认 false", "default": False},
        },
        "required": ["src", "dst"],
    },
    category="文件",
    when_not_to_use=(
        "只是想「另存为再改」用 write_file 直接写新路径即可；"
        "大目录拷贝会拖慢并占上下文，先用 list_dir 确认范围。"
    ),
)
def copy_file(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    src = ctx.resolve(args["src"], must_exist=True)
    dst = ctx.resolve(args["dst"])
    if dst.exists() and not args.get("overwrite"):
        raise ToolError(f"目标已存在：{args['dst']}（要覆盖请传 overwrite=true）", tool="copy_file")

    backup = ""
    if dst.exists() and args.get("overwrite"):
        backup = _backup(ctx, dst)
        if dst.is_dir():
            shutil.rmtree(str(dst))
    try:
        if src.is_dir():
            shutil.copytree(str(src), str(dst))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
    except OSError as exc:
        raise ToolError(f"复制失败：{exc}", tool="copy_file") from exc

    ctx.record_change("copy", f"{_rel(ctx, src)} -> {_rel(ctx, dst)}", path=_rel(ctx, dst))
    detail = f"已复制 {_rel(ctx, src)} -> {_rel(ctx, dst)}"
    if backup:
        detail += f"\n原目标已备份至 {backup}"
    return ToolResult.success(detail, meta={"src": str(src), "dst": str(dst)})


# ----------------------------------------------------------------------------
# delete
# ----------------------------------------------------------------------------
@tool_spec(
    name="delete",
    description=(
        "删除文件或目录（在沙箱内）。删除前会先备份到 .agent_backups，可恢复。\n"
        "删除目录必须传 recursive=true；误删文件可用 .agent_backups 找回。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要删除的路径，相对工作区"},
            "recursive": {"type": "boolean", "description": "删除目录时需要 true，默认 false", "default": False},
        },
        "required": ["path"],
    },
    category="文件",
    when_not_to_use=(
        "删除不可恢复（虽先备份，但别滥用）。密钥 / 配置类文件（*.pem/.env/config.yaml）不要删。"
        "只是想撤销改动用 rollback，比删文件安全。"
    ),
)
def delete(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    target = ctx.resolve(args["path"], must_exist=True)
    if target.is_dir() and not args.get("recursive"):
        raise ToolError(
            f"{args['path']} 是目录，删除需传 recursive=true",
            tool="delete",
            hint="确认要递归删除整个目录吗？是则传 recursive=true。",
        )

    # 删除前备份（目录过大则跳过备份并提示）
    backup = ""
    try:
        if target.is_file():
            backup = _backup(ctx, target)
        elif target.is_dir():
            total = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            if total <= _MAX_DIR_BACKUP_BYTES:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                rel = Path(ctx.guard.relpath(target))
                task_name = Path(str(ctx.config.workspace)).expanduser().resolve().name
                dest = _backup_root(ctx) / ".overwrites" / task_name / stamp / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(str(target), str(dest))
                backup = ctx.guard.relpath(dest)
            else:
                backup = "（目录过大未备份，请手动确认后再删）"
    except OSError:
        backup = ""

    try:
        if target.is_dir():
            shutil.rmtree(str(target))
        else:
            target.unlink()
    except OSError as exc:
        raise ToolError(f"删除失败：{exc}", tool="delete") from exc

    ctx.record_change("delete", _rel(ctx, target), path=_rel(ctx, target))
    detail = f"已删除 {_rel(ctx, target)}"
    if backup and backup != "（目录过大未备份，请手动确认后再删）":
        detail += f"\n已备份至 {backup}"
    return ToolResult.success(detail, meta={"path": str(target), "backup": backup})


def register(registry) -> None:
    registry.register_many([move_file, copy_file, delete])
