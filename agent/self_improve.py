"""
自我改进钩子（self-improvement）

让 agent 从「本次任务的失败 / 修复 / 中断」里沉淀出可泛化的经验，落盘到项目记忆
`.minicode/memory.md`，下次启动时被自动注入 system 提示词——形成
「失败 → 记忆 → 下次更聪明」的闭环，类比 Claude Code / Codex 的「记忆自更新」。

设计要点
--------
- 纯本地、确定性、**不依赖模型**：用规则把"本次会话的可观测信号"映射成经验句，
  因此完全离线可测、不引入任何 agent 框架（符合考核硬约束）。
- 与 `memory` 工具共用同一份 `.minicode/memory.md`：自改进追加的 `[auto]` 条目
  同样是项目记忆的一部分，会被启动注入——不必新开一条记忆通道。
- 去重 + 封顶：同一经验不重复写；自动条目过多就停止追加，避免撑爆提示词。
- 永不把异常抛给主循环：loop 调用处已 try/except，本模块内部读写也兜底。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .tools.memory import MEMORY_DIR, MEMORY_FILE

_LESSON_SECTION = "## 自动沉淀的经验（self-improve）"
_MAX_AUTO_ENTRIES = 60
_AUTO_TAG = "[auto]"


@dataclass
class SessionSignal:
    """一次任务结束后的可观测信号（由主循环采集，不直接依赖模型输出文本）。"""

    repair_rounds: int = 0       # 自修复成功回合数
    tool_errors: int = 0         # 工具调用失败次数
    permission_denied: int = 0   # 被权限系统拒绝次数
    empty_responses: int = 0     # 模型空响应次数
    finish_blocked: int = 0      # finish 被「先验证」拦截次数
    compactions: int = 0         # 上下文压缩触发次数
    aborted: bool = False        # 是否被用户中断
    llm_error: bool = False      # 是否模型调用失败


# 信号 → 可泛化的经验句（写给下个会话看的「结论」，不是原始日志）
_LESSON_TEMPLATES: Dict[str, str] = {
    "repair": (
        "本次任务触发了自修复闭环（运行/测试失败→定位 traceback→改源码→再验证）。"
        "下次遇到报错，先读完整 traceback 再改，别盲改；改完务必跑一次验证命令确认。"
    ),
    "permission": (
        "存在被权限系统拒绝的写/破坏性操作。涉及这类操作时，先确认当前 permission_mode"
        "（auto/ask/read_only）；需要落盘就显式走受控流程，别假设一定能写。"
    ),
    "empty_response": (
        "模型出现过空响应（未计入成功）。若连续空响应，应明确重试并给出提示，"
        "不要把「无输出」误判为任务完成。"
    ),
    "finish_blocked": (
        "finish 曾因「改了文件却没验证」被拦截。收尾前若动过文件，必须先跑验证/运行命令，"
        "再 finish 总结，避免交付未经验证的产物。"
    ),
    "compaction": (
        "本次会话触发了上下文压缩。长任务要善用 plan/todo 把目标拆小，"
        "并优先用 read_many_files/recall 按需取文件，降低上下文压力。"
    ),
    "aborted": (
        "任务曾被用户中断。涉及长耗时操作时，考虑用 run_command 的 background 模式异步跑，"
        "边等边推进其它子目标，减少被整体打断的概率。"
    ),
    "llm_error": (
        "出现过模型调用失败。应做好重试与降级（网关不支持流式就退回整包），"
        "并把失败记进结果而非让 CLI 崩溃。"
    ),
}

_DEDUPE_CHARS = "，。、；：,.;: \t\n（）()\"'「」"


def _normalize(line: str) -> str:
    """归一化用于去重：去标签、去空白、去常见标点，做粗略去重。"""
    s = line.strip()
    if s.startswith("-"):
        s = s[1:].strip()
    if s.startswith(_AUTO_TAG):
        s = s[len(_AUTO_TAG):].strip()
    for ch in _DEDUPE_CHARS:
        s = s.replace(ch, "")
    return s


def derive_lessons(signal: SessionSignal) -> List[str]:
    """把会话信号映射成去重后的经验句列表（顺序稳定）。"""
    lessons: List[str] = []
    seen = set()
    order = ["repair", "permission", "empty_response", "finish_blocked",
             "compaction", "aborted", "llm_error"]
    for key in order:
        if key == "repair":
            triggered = (signal.repair_rounds or 0) > 0 or (signal.tool_errors or 0) > 0
        elif key == "permission":
            triggered = (signal.permission_denied or 0) > 0
        elif key == "empty_response":
            triggered = (signal.empty_responses or 0) > 0
        elif key == "finish_blocked":
            triggered = (signal.finish_blocked or 0) > 0
        elif key == "compaction":
            triggered = (signal.compactions or 0) > 0
        elif key == "aborted":
            triggered = bool(signal.aborted)
        elif key == "llm_error":
            triggered = bool(signal.llm_error)
        else:
            triggered = False
        if not triggered:
            continue
        text = _LESSON_TEMPLATES[key]
        norm = _normalize(text)
        if norm in seen:
            continue
        seen.add(norm)
        lessons.append(text)
    return lessons


def _memory_path(workspace_root) -> Path:
    """记忆文件固定落在 `<workspace_root>/.minicode/memory.md`（与 memory 工具同址）。"""
    root = Path(str(workspace_root)).expanduser().resolve()
    return (root / MEMORY_DIR / MEMORY_FILE).resolve()


def _find_section(lines: List[str]) -> int:
    for i, ln in enumerate(lines):
        if ln.strip() == _LESSON_SECTION:
            return i
    return None


def record_lessons(workspace_root, lessons: List[str],
                   max_entries: int = _MAX_AUTO_ENTRIES) -> int:
    """把经验句追加进项目记忆（同一份 memory.md 的专用小节），去重并封顶。

    Returns:
        实际新增的条目数（0 表示无需新增或写入失败）。
    """
    lessons = [x for x in (lessons or []) if x and x.strip()]
    if not lessons:
        return 0
    path = _memory_path(workspace_root)
    try:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError:
        return 0

    lines = existing.splitlines()
    existing_auto = [ln for ln in lines if ln.strip().startswith(f"- {_AUTO_TAG}")]
    existing_norm = {_normalize(ln) for ln in existing_auto}
    current_count = len(existing_auto)

    new_lines: List[str] = []
    added = 0
    for lesson in lessons:
        norm = _normalize(lesson)
        if norm in existing_norm:
            continue
        if current_count + added >= max_entries:
            break  # 触顶即停，避免撑爆记忆 / 提示词
        existing_norm.add(norm)
        new_lines.append(f"- {_AUTO_TAG} {lesson}")
        added += 1

    if added == 0:
        return 0

    # 去掉结尾空行，再拼装
    while lines and not lines[-1].strip():
        lines.pop()
    out = list(lines)
    section_idx = _find_section(out)
    if section_idx is None:
        if out:
            out.append("")
        out.append(_LESSON_SECTION)
        out.append("")
        out.extend(new_lines)
    else:
        insert_at = section_idx + 1
        if insert_at < len(out) and out[insert_at].strip() == "":
            insert_at += 1
        out[insert_at:insert_at] = new_lines
    out.append("")  # 末尾空行

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(out), encoding="utf-8", newline="")
    except OSError:
        return 0
    return added


def forget_lessons(workspace_root) -> int:
    """清空专用小节里的自动经验条目（保留 memory 工具写的其它内容）。返回删除条数。"""
    path = _memory_path(workspace_root)
    if not path.exists():
        return 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    section_idx = _find_section(lines)
    if section_idx is None:
        return 0
    removed = 0
    out: List[str] = lines[:section_idx]
    i = section_idx + 1
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("## "):  # 遇到下一个小节标题，后面整段保留
            out.append("")
            out.extend(lines[i:])
            break
        if ln.strip().startswith(f"- {_AUTO_TAG}"):
            removed += 1  # 跳过：删除该自动条目
        else:
            out.append(ln)
        i += 1
    while out and not out[-1].strip():
        out.pop()
    try:
        path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8", newline="")
    except OSError:
        return 0
    return removed


def read_lessons(workspace_root) -> List[str]:
    """读取专用小节里的自动经验条目（纯文本行，去标签）。"""
    path = _memory_path(workspace_root)
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    section_idx = _find_section(lines)
    if section_idx is None:
        return []
    out: List[str] = []
    for ln in lines[section_idx + 1:]:
        if ln.strip().startswith("## "):
            break
        if ln.strip().startswith(f"- {_AUTO_TAG}"):
            out.append(_normalize(ln))  # 规范化展示，去掉 [auto] 与标点噪声
    return out
