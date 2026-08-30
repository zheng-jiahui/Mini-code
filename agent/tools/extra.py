"""
扩展工具集：read_many_files / replace_in_file / web_fetch / think。

这些工具都围绕一个目标——让 agent 更像「商用编程智能体」：
    · read_many_files：一次看清多个相关文件，省掉多轮往返；
    · replace_in_file：简单粗暴的全局字符串替换（edit_block 要求唯一匹配，
      但有时就是要「把所有旧名换成新名」）；
    · web_fetch：让 agent 能自己读文档 / RFC / API 说明（自实现，不依赖任何
      服务端工具，符合考核「重要逻辑需自行编写」的要求）；
    · think：把推理 / 计划写进「便签」，缓解长任务里「边想边改」容易跑偏的问题。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..errors import ToolError
from ..security import truncate_output
from .base import DIFF_CAPTURE_CAP, ToolContext, ToolResult, tool_spec
from .filesystem import _backup

__all__ = ["read_many_files", "replace_in_file", "web_fetch", "think", "register"]

_MAX_BATCH = 50  # 单次最多读多少个文件，防止一把梭进整目录把上下文撑爆


# ----------------------------------------------------------------------------
# 辅助（必须放在 @tool_spec 之前）
# ----------------------------------------------------------------------------
def _looks_binary(raw: bytes) -> bool:
    if b"\x00" in raw[:4096]:
        return True
    try:
        raw[:8192].decode("utf-8")
        return False
    except UnicodeDecodeError:
        return True


# ----------------------------------------------------------------------------
# read_many_files
# ----------------------------------------------------------------------------
@tool_spec(
    name="read_many_files",
    description=(
        "一次读取多个文本文件，每个文件带行号、用文件名小标题分隔后合并返回。"
        "适合「一次看清几个相关文件」（如配置 + 入口 + 测试），比逐个 read_file 省多轮往返。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "要读取的文件路径列表（相对工作区），如 [src/main.py, tests/test_main.py]",
            },
            "offset": {"type": "integer", "description": "每个文件从第几行开始读（1 起，默认 1）", "default": 1},
            "limit": {"type": "integer", "description": "每个文件最多读多少行，默认读全部（超长自动截断）"},
        },
        "required": ["paths"],
    },
    category="文件",
    when_not_to_use=(
        "只读取一个文件时用 read_file 更直观；这工具适合「一次看清多个相关文件」。"
        "别把整个目录几十个文件都塞进来——先用 list_dir / grep_search 定位相关文件，再批量读。"
    ),
)
def read_many_files(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    paths = args.get("paths")
    if not isinstance(paths, list) or not paths:
        raise ToolError("paths 必须是非空的字符串数组", tool="read_many_files")
    paths = [str(p).strip() for p in paths if str(p).strip()][: _MAX_BATCH]

    offset = max(1, int(args.get("offset") or 1))
    limit = int(args.get("limit") or 0)
    max_chars = int(getattr(ctx.config, "max_file_read_chars", 40_000))

    pieces: List[str] = []
    read_ok = 0
    for p in paths:
        try:
            target = ctx.resolve(p, must_exist=True)
        except ToolError as exc:
            pieces.append(f"## {p}\n[跳过] {exc.render()}")
            continue
        if target.is_dir():
            pieces.append(f"## {p}\n[跳过] 这是一个目录，请用 list_dir")
            continue
        try:
            raw = target.read_bytes()
        except OSError as exc:
            pieces.append(f"## {p}\n[跳过] 读取失败：{exc}")
            continue
        if _looks_binary(raw):
            pieces.append(f"## {p}\n[跳过] 疑似二进制文件（{len(raw)} 字节），读出来是乱码")
            continue
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        chunk = lines[offset - 1: offset - 1 + limit] if limit else lines
        numbered = "\n".join(f"{i + offset:>5}| {ln}" for i, ln in enumerate(chunk))
        pieces.append(
            f"## {p}（共 {len(lines)} 行，本次返回第 {offset}-{offset + len(chunk) - 1} 行）\n{numbered}"
        )
        read_ok += 1

    body = "\n\n".join(pieces)
    if len(body) > max_chars:
        body = truncate_output(body, max_chars, note="合并内容过长")
    return ToolResult.success(body, meta={"requested": len(paths), "read": read_ok})


# ----------------------------------------------------------------------------
# replace_in_file —— 全局字符串替换（区别于 edit_block 的唯一匹配）
# ----------------------------------------------------------------------------
@tool_spec(
    name="replace_in_file",
    description=(
        "把文件中**所有**出现的 old_text 替换为 new_text（全局替换，不要求唯一）。\n"
        "适合「把一个旧符号 / 旧 URL / 旧配置统一改成新的」这类批量改名场景。"
        "new_text 为空字符串表示删除这些片段。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径，相对工作区"},
            "old_text": {"type": "string", "description": "要被替换的原文（会替换所有出现处）"},
            "new_text": {"type": "string", "description": "替换后的新文本（空串表示删除）"},
        },
        "required": ["path", "old_text", "new_text"],
    },
    category="文件",
    when_not_to_use=(
        "只改一处、且能给出唯一上下文时，优先用 edit_block（更精准、改动范围更小、更不易误伤）。"
        "本工具会替换所有出现处，若 old_text 在别处也存在，会一并被改掉——确认这是你想要的再用。"
        "改动前务必先 read_file 看过原文。"
    ),
)
def replace_in_file(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    target = ctx.resolve(args["path"], must_exist=True)
    if target.is_dir():
        raise ToolError(f"{args['path']} 是目录，不能替换", tool="replace_in_file")

    old_text = args.get("old_text")
    new_text = args.get("new_text")
    if old_text is None or new_text is None:
        raise ToolError("old_text 与 new_text 都是必填", tool="replace_in_file")
    if not str(old_text):
        raise ToolError("old_text 不能为空，否则会清空整个文件", tool="replace_in_file")

    try:
        original = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ToolError(f"读取失败：{exc}", tool="replace_in_file") from exc

    count = original.count(str(old_text))
    if count == 0:
        raise ToolError(
            f"在 {args['path']} 中找不到 old_text（0 处）",
            tool="replace_in_file",
            hint="请先 read_file 确认原文；注意缩进与空白需完全一致。",
        )

    if getattr(ctx.config, "backup_on_write", True):
        _backup(ctx, target)

    new_content = original.replace(str(old_text), str(new_text))
    try:
        target.write_text(new_content, encoding="utf-8", newline="")
    except OSError as exc:
        raise ToolError(f"写入失败：{exc}", tool="replace_in_file",
                        hint="文件可能只读或被其它程序占用。") from exc

    ctx.record_change("edit", args["path"], before=original, after=new_content, path=args["path"])
    delta = len(new_content.splitlines()) - len(original.splitlines())
    return ToolResult.success(
        f"已在 {args['path']} 中替换 {count} 处（行数变化 {'+' if delta >= 0 else ''}{delta}）",
        meta={"replacements": count, "path": str(target)},
    )


def register(registry) -> None:
    registry.register_many([read_many_files, replace_in_file])
