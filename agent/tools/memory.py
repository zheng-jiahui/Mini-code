"""
项目记忆工具：memory —— 让 agent 跨会话积累「项目约定与历史结论」。

为什么需要它：
    商业编程智能体（Claude Code / Codex）普遍会读一份项目级记忆
    （CLAUDE.md / AGENTS.md），把「这个仓库的约定、踩过的坑、定过的结论」
    沉淀下来，下次打开直接带上，不必每次从零探索。
    本工具把这个能力自实现：记忆落盘到项目根 `.minicode/memory.md`，
    启动时被自动注入 system 提示词（见 agent/loop.py）；agent 也可用
    append 把新结论写回，越用越懂这个项目。

与 todo 的区别：todo 只活在本会话；memory 落盘到项目根，跨会话、跨任务持久。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ..errors import ToolError
from .base import ToolContext, ToolResult, tool_spec

__all__ = ["memory", "register", "read_memory_file", "format_memory_section",
           "MEMORY_DIR", "MEMORY_FILE"]

MEMORY_DIR = ".minicode"
MEMORY_FILE = "memory.md"
_SECTION_HEADER = "# 项目记忆（跨会话持久）"


# ----------------------------------------------------------------------------
# 辅助（必须放在 @tool_spec 之前）
# ----------------------------------------------------------------------------
def _memory_path(ctx: ToolContext) -> Path:
    """记忆文件固定落在项目根 `.minicode/memory.md`，且必须仍在项目根内（防越界）。"""
    root = Path(str(getattr(ctx.config, "workspace_root", None) or ctx.workspace)).expanduser().resolve()
    dest = (root / MEMORY_DIR / MEMORY_FILE).resolve()
    try:
        dest.relative_to(root)  # 防御：记忆路径绝不能逃出项目根
    except ValueError:
        raise ToolError(f"记忆文件路径越界：{dest}", tool="memory")
    return dest


def read_memory_file(workspace_root) -> str:
    """读取项目记忆原文（供主循环启动注入）。文件不存在返回空串。"""
    try:
        p = (Path(str(workspace_root)).expanduser().resolve() / MEMORY_DIR / MEMORY_FILE)
        if p.exists():
            return p.read_text(encoding="utf-8")
    except OSError:
        pass
    return ""


def format_memory_section(text: str) -> str:
    """把记忆原文包成要注入 system 提示词的段落；空则空串。"""
    if not text.strip():
        return ""
    return (f"{_SECTION_HEADER}\n"
            f"以下是过往会话沉淀的项目约定与历史结论，与本次任务相关时优先于通用建议采纳：\n"
            f"{text.strip()}\n")


# ----------------------------------------------------------------------------
# memory
# ----------------------------------------------------------------------------
@tool_spec(
    name="memory",
    description=(
        "读写项目级持久记忆（存于项目根 `.minicode/memory.md`，跨会话、跨任务保留）。\n"
        "action=read 查看当前全部记忆；action=append 追加一条新结论（自动带时间戳）；\n"
        "action=update 用 content 整体覆盖记忆；action=clear 清空记忆。\n"
        "适合沉淀：项目约定、反复踩的坑、已定的技术选型、某类任务的标准做法——"
        "下次会话会自动读回并注入提示词，不必每次从零探索。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "操作类型",
                "enum": ["read", "append", "update", "clear"],
            },
            "content": {
                "type": "string",
                "description": "action=append 时的笔记内容 / action=update 时的完整记忆（其它操作可省略）；update 时不能为空",
            },
        },
        "required": ["action"],
    },
    category="记忆",
    when_not_to_use=(
        "本会话的临时待办/计划用 todo，不要写进 memory（memory 是跨会话长期记忆，会一直带着）。"
        "不要往 memory 里塞机密或一次性草稿。append 一次写一条结论，别把整段日志灌进去。"
    ),
)
def memory(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    action = (args.get("action") or "").strip().lower()
    path = _memory_path(ctx)

    if action == "read":
        if not path.exists():
            return ToolResult.success("（暂无项目记忆。用 action=append 写下第一条约定或结论。）")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"读取记忆失败：{exc}", tool="memory") from exc
        if not text.strip():
            return ToolResult.success("（记忆文件为空。用 action=append 补充。）", meta={"bytes": 0})
        return ToolResult.success(f"{_SECTION_HEADER}\n{text}", meta={"bytes": len(text)})

    if action == "append":
        content = (args.get("content") or "").strip()
        if not content:
            raise ToolError("action=append 时必须提供 content（要记的结论）", tool="memory")
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        line = f"- {stamp} {content}\n"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():  # 新文件先写标题，保证格式一致
                path.write_text(_SECTION_HEADER + "\n", encoding="utf-8")
            with path.open("a", encoding="utf-8", newline="") as f:
                f.write(line)
        except OSError as exc:
            raise ToolError(f"写入记忆失败：{exc}", tool="memory") from exc
        return ToolResult.success(
            f"已追加到项目记忆：{content}\n（文件：{ctx.guard.relpath(path)}）",
            meta={"action": "append"},
        )

    if action == "update":
        content = args.get("content")
        if content is None or not str(content).strip():
            raise ToolError("action=update 时 content 不能为空（要清空请用 action=clear）", tool="memory")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(content), encoding="utf-8", newline="")
        except OSError as exc:
            raise ToolError(f"写入记忆失败：{exc}", tool="memory") from exc
        return ToolResult.success(
            f"已用新内容整体覆盖项目记忆（{len(str(content))} 字符）。",
            meta={"action": "update", "bytes": len(str(content))},
        )

    if action == "clear":
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            raise ToolError(f"清空记忆失败：{exc}", tool="memory") from exc
        return ToolResult.success("已清空项目记忆（文件已删除，下次会话将从空白开始）。",
                                  meta={"action": "clear"})

    raise ToolError(f"未知 action={action!r}", tool="memory",
                   hint="可选：read / append / update / clear。")


def register(registry) -> None:
    registry.register(memory)
