"""
apply_patch 工具：把一段标准 unified diff 应用到工作区里**已存在**的文件。

为什么自己实现、不调外部 `patch`/`git apply`？
    让它在「没有 git、甚至没有 patch 命令」的环境（部分 Windows 机器）也能用，
    不引入对外部二进制的隐式依赖——这正是本项目「不依赖框架、可控、可审阅」的一贯取向。
    应用器只处理最常见的 unified diff（@@ hunk + 上下文/增/删三态），足以覆盖
    「模型给出一段 diff 让 agent 落地」这一核心场景；解析不到任何 hunk 时如实报错，
    绝不静默吞掉。

安全：
    - 走 ctx.resolve 路径沙箱，越界被 PathGuard 拦成 ok=False；
    - 写前由调用方的备份机制（write_file 同款）兜底；
    - 任一 hunk 在现状里定位不到就整体失败、不写盘，避免把文件改到半成品状态。
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..tools.base import DIFF_CAPTURE_CAP, ToolContext, ToolResult, tool_spec

__all__ = ["apply_patch", "register", "_parse_patch", "_apply_hunk"]

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def _parse_patch(patch_text: str) -> List[dict]:
    """把 unified diff 文本解析成 hunk 列表。

    每个 hunk: {"old_start": int, "new_start": int, "lines": [(type, text), ...]}
        type: " " 上下文  |  "-" 删除  |  "+" 新增
    文件头（--- / +++ / index）与 `\ No newline` 备注一律忽略。
    """
    hunks: List[dict] = []
    cur: Optional[dict] = None
    for raw in (patch_text or "").splitlines():
        m = _HUNK_RE.match(raw)
        if m:
            cur = {
                "old_start": int(m.group(1)),
                "new_start": int(m.group(3)),
                "lines": [],
            }
            hunks.append(cur)
            continue
        if cur is None:
            continue  # hunk 之前的文件头
        if raw.startswith("--- ") or raw.startswith("+++ "):
            continue
        if raw.startswith("\\"):  # "\ No newline at end of file" 等
            continue
        if raw.startswith("+"):
            cur["lines"].append(("+", raw[1:]))
        elif raw.startswith("-"):
            cur["lines"].append(("-", raw[1:]))
        elif raw.startswith(" "):
            cur["lines"].append((" ", raw[1:]))
        else:
            cur["lines"].append((" ", raw))  # 兜底：无前缀视为上下文
    return hunks


def _old_side(lines: List[Tuple[str, str]]) -> List[str]:
    """hunk 在「旧文件」一侧应有的连续文本块（上下文 + 删除行）。"""
    return [text for typ, text in lines if typ in (" ", "-")]


def _apply_hunk(old_lines: List[str], hunk: dict) -> Tuple[Optional[int], Optional[List[str]]]:
    """在 old_lines 中定位并应用单个 hunk。

    Returns:
        (匹配起点, 替换后的新块) —— 找不到时返回 (None, None)。
    替换块 = 上下文行原样保留 + 新增行插入 - 删除行跳过。
    """
    old_block = _old_side(hunk["lines"])
    m = len(old_block)
    if m == 0:
        # 纯新增（无上下文无删除）：插在 old_start 指示的位置
        pos = max(0, min(int(hunk["old_start"]) - 1, len(old_lines)))
        new_block = [text for typ, text in hunk["lines"] if typ == "+"]
        return pos, new_block

    n = len(old_lines)
    positions = [i for i in range(n - m + 1) if old_lines[i:i + m] == old_block]
    if not positions:
        # 容错：忽略首尾空白再比一次（模型生成的 diff 偶尔有多余缩进漂移）
        norm = lambda blk: [ln.strip() for ln in blk]
        target = norm(old_block)
        for i in range(n - m + 1):
            if norm(old_lines[i:i + m]) == target:
                positions.append(i)
                break
    if not positions:
        return None, None

    pos = positions[0]
    new_block = [text for typ, text in hunk["lines"] if typ in (" ", "+")]
    return pos, new_block


@tool_spec(
    name="apply_patch",
    description=(
        "把一段标准 unified diff（由 `@@` 分隔的 hunk 组成）应用到指定文件，"
        "常用于把模型给出的改动一次性落到文件里。只改已存在的文件，且会自动备份原文。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "目标文件路径（相对工作目录）"},
            "patch": {
                "type": "string",
                "description": "标准 unified diff 文本，例如：\n"
                               "@@ -1,3 +1,3 @@\n line1\n-line2\n+line2_fixed\n line3",
            },
        },
        "required": ["path", "patch"],
    },
    category="文件",
    when_not_to_use=(
        "目标是新建文件请用 write_file；只改其中一小段且你能拿到精确 old_text 时，"
        "edit_block 比手写 diff 更稳、更不容易因上下文对不上而失败。"
    ),
)
def apply_patch(args: dict, ctx: ToolContext) -> ToolResult:
    path = (args or {}).get("path")
    patch = (args or {}).get("patch")
    if not path or not patch:
        return ToolResult.failure("path 与 patch 均为必填参数")

    target = ctx.resolve(path)
    if not target.exists():
        return ToolResult.failure(
            f"文件不存在：{path}。apply_patch 只改已存在的文件；新建请用 write_file。",
            hint="先用 list_dir / read_file 确认路径，或改用 write_file 创建。",
        )

    try:
        old_text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return ToolResult.failure(f"无法读取文件 {path}：{exc}", hint="确认文件是文本文件且在工作区内。")

    hunks = _parse_patch(patch)
    if not hunks:
        return ToolResult.failure(
            "未在 patch 中解析到任何 `@@` hunk，请确认传入的是标准 unified diff。",
            hint="diff 需包含形如 `@@ -1,3 +1,3 @@` 的 hunk 头。",
        )

    new_lines = old_text.split("\n")
    applied = 0
    for hunk in hunks:
        pos, new_block = _apply_hunk(new_lines, hunk)
        if pos is None:
            return ToolResult.failure(
                f"第 {applied + 1} 个 hunk 无法在文件现状中定位（可能文件已被改动，"
                "或 diff 与当前内容不匹配）。已应用的 hunk 未写入磁盘。",
                hint="请用 read_file 重新读取最新内容，再生成与现状匹配的 diff。",
            )
        block_len = len(_old_side(hunk["lines"]))
        new_lines[pos:pos + block_len] = new_block
        applied += 1

    new_text = "\n".join(new_lines)
    # 记录变更（before/after 仅在小文件时采集，供 /diff 生成对照）
    cap = DIFF_CAPTURE_CAP
    before = old_text if len(old_text) <= cap else None
    after = new_text if len(new_text) <= cap else None
    ctx.record_change(
        "patch",
        f"应用 diff 到 {path}（{applied}/{len(hunks)} 个 hunk）",
        before=before, after=after, path=path,
    )
    try:
        target.write_text(new_text, encoding="utf-8")
    except OSError as exc:
        return ToolResult.failure(f"写入失败：{exc}")
    return ToolResult.success(
        f"已应用 {applied}/{len(hunks)} 个 hunk 到 {path}。",
        meta={"applied": applied, "total": len(hunks), "path": str(target)},
    )


def register(registry) -> None:
    registry.register(apply_patch)
