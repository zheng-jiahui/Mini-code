"""
会话检查点：让一次会话能"停下来、下次接着跑"。

为什么需要
----------
V3 解决了上下文超窗，V7 解决了压缩导致的遗忘，但**进程一退就全丢**。
一个跑了几十步的长任务（或夜里跑到一半被中断/断网）要重跑，代价极高。

核心设计决策：区分「必须持久化的状态」与「可重建的状态」
----------------------------------------------------
    必须存（丢了就重建不出来）：
        history.messages / kinds / facts —— 发生过什么；
        task_name —— 否则恢复后代码会落到另一个目录；
        会话统计（usage、工具耗时与次数、压缩次数）—— 成本对账要跨进程连续。

    不存（恢复时重建）：
        backend client —— 从配置重建；
        registry —— 代码里注册，重建即可；
        system_prompt、project_profile —— 它们依赖**当前**的工具清单与工作区，
            存一份旧的反而会与现状不符（工具增删了、工作区文件变了），
            恢复时必须重新生成。

    所以恢复是「**重建骨架 + 回填历史**」，而不是"把整个对象反序列化"。
    这条边界是本模块唯一真正重要的设计，其余都是配套。

另外两条硬约束
--------------
1. **原子写**（临时文件 + `os.replace`）。中断恰恰会发生在写到一半的时候，
   留下一份半截 JSON，下次恢复直接崩掉——**坏掉的检查点比没有检查点更糟**。
2. **恢复后必须如实告诉模型这是在续跑**。否则它会以为自己刚执行过历史里的
   那些操作，于是要么重复执行、要么跳过本该做的验证步骤。

不做什么
--------
    不做"自动续跑上次没完成的任务"：恢复之后该做什么，仍交给模型判断，
    我们只负责把事实（历史 + 工作区现状）摆到它面前。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "CHECKPOINT_VERSION",
    "checkpoint_dir",
    "save",
    "latest",
    "load",
    "apply",
    "describe",
]

CHECKPOINT_VERSION = 1

# 检查点目录名：放在 .agent_sessions 下（已在 .gitignore 中，不会入库）
_CHECKPOINT_SUBDIR = "checkpoints"

RESUME_NOTE = (
    "【这是恢复的会话】以上的对话历史来自上次保存的检查点（{n_msgs} 条消息，"
    "曾压缩 {n_compact} 次，保存于 {saved_at}）。\n"
    "请注意：这些只是历史记录，**不代表你刚刚执行过其中的操作**——"
    "工作区可能已被改动过。继续动手前请先用 list_dir / read_file 确认当前实际状态，"
    "再决定下一步；不要仅凭历史记录就假定某个文件已经改好了。"
)


# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
def checkpoint_dir(loop) -> Path:
    """检查点目录：workspace_root/.agent_sessions/checkpoints/。

    用 workspace_root 而不是 config.workspace——后者会随任务切换而变，
    检查点需要一个稳定的家，否则"列出可恢复的会话"都做不到。
    """
    log_dir = getattr(loop.config, "session_log", None) or ".agent_sessions"
    root = Path(str(loop.workspace_root)).expanduser().resolve()
    return root / str(log_dir) / _CHECKPOINT_SUBDIR


def _path_for(loop, task_name: str) -> Path:
    safe = "".join(c for c in (task_name or "未命名任务") if c not in '\\/:*?"<>|').strip()
    return checkpoint_dir(loop) / f"{safe or '未命名任务'}.json"


# ----------------------------------------------------------------------------
# 保存
# ----------------------------------------------------------------------------
def save(loop, *, path: Optional[Path] = None) -> Optional[Path]:
    """把当前会话写入检查点。返回写入路径，未启用/无内容时返回 None。

    只在历史里确有内容时才写：一次都没跑过的空会话没有恢复价值，
    写一个空文件反而会让启动时多一句无意义的提示。
    """
    if not getattr(loop.config, "auto_checkpoint", True):
        return None
    messages: List[Dict[str, Any]] = list(loop.history.messages)
    if len(messages) <= 1:          # 只有 system 提示词
        return None
    if not loop.task_name:
        return None

    state: Dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "saved_at": time.time(),
        "saved_at_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_name": loop.task_name,
        "task": getattr(loop, "last_task", "") or "",
        "facts": loop.history.facts,
        # 不存 index 0（system）：它依赖当前工具清单与工作区，恢复时重建（见模块 docstring）
        "messages": messages[1:],
        "kinds": list(loop.history.kinds)[1:],
        "compact_count": loop.history.compact_count,
        "usage": dict(loop._usage),
        "tool_timings": dict(loop._tool_timings),
        "tool_calls_by_name": dict(loop._tool_calls_by_name),
        "total_tool_calls": loop._total_tool_calls,
        "output_compressions": loop._output_compressions,
        "model_time": loop._model_time,
        "model_calls": loop._model_calls,
    }

    dest = Path(path) if path else _path_for(loop, loop.task_name)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        # 原子写：先写临时文件再替换。中断只会发生在写到一半的时候，
        # 留下半截 JSON 会让下次恢复直接崩掉——坏掉的检查点比没有更糟。
        tmp = dest.with_name(dest.name + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(state, f, ensure_ascii=False, default=str)
        os.replace(tmp, dest)
    except (OSError, TypeError, ValueError):
        return None
    return dest


def latest(loop) -> Optional[Path]:
    """最近一次保存的检查点（按修改时间取最新）。"""
    d = checkpoint_dir(loop)
    if not d.is_dir():
        return None
    candidates = [p for p in d.iterdir() if p.is_file() and p.suffix == ".json"]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ----------------------------------------------------------------------------
# 读取与恢复
# ----------------------------------------------------------------------------
def load(path: Path) -> Dict[str, Any]:
    """读取检查点。格式不符或损坏时抛 ValueError（调用方负责兜底提示）。

    这里**不静默吞掉错误**：一份读不出来的检查点必须让用户知道它坏了，
    否则他会以为"没有可恢复的会话"，而实际上是有、但坏了——两者的处置完全不同。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"检查点无法解析：{exc}") from exc

    if not isinstance(state, dict):
        raise ValueError("检查点格式错误：顶层不是对象")
    if int(state.get("version") or 0) != CHECKPOINT_VERSION:
        raise ValueError(f"检查点版本不兼容：{state.get('version')}（当前 {CHECKPOINT_VERSION}）")
    if not isinstance(state.get("messages"), list) or not isinstance(state.get("kinds"), list):
        raise ValueError("检查点格式错误：缺少 messages / kinds")
    if len(state["messages"]) != len(state["kinds"]):
        raise ValueError("检查点已损坏：messages 与 kinds 长度不一致")
    return state


def apply(loop, state: Dict[str, Any]) -> int:
    """把检查点回填到 loop。返回恢复的历史消息条数。

    顺序很重要：先 prepare_task_dir（它会重建 system_prompt 与项目画像），
    再把历史接在 system 之后——保证 system 提示词永远反映**当前**的工具与工作区。
    """
    task_name = state.get("task_name") or ""
    task = state.get("task") or task_name or "恢复的任务"
    loop.prepare_task_dir(task, task_name=task_name or None, force_new=True)

    messages: List[Dict[str, Any]] = [loop.history.messages[0]] + list(state["messages"])
    kinds: List[str] = ["system"] + list(state["kinds"])
    loop.history.messages = messages
    loop.history.kinds = kinds
    loop.history.facts = state.get("facts") or ""
    loop.history.compact_count = int(state.get("compact_count") or 0)

    # 统计口径跨进程连续，否则 /stats 恢复后会从零开始，成本对账就断了
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        loop._usage[key] = int((state.get("usage") or {}).get(key) or 0)
    loop._tool_timings = {str(k): float(v) for k, v in (state.get("tool_timings") or {}).items()}
    loop._tool_calls_by_name = {str(k): int(v) for k, v in (state.get("tool_calls_by_name") or {}).items()}
    loop._total_tool_calls = int(state.get("total_tool_calls") or 0)
    loop._output_compressions = int(state.get("output_compressions") or 0)
    loop._model_time = float(state.get("model_time") or 0.0)
    loop._model_calls = int(state.get("model_calls") or 0)

    n_msgs = len(state["messages"])
    loop.history.add_note(RESUME_NOTE.format(
        n_msgs=n_msgs,
        n_compact=loop.history.compact_count,
        saved_at=state.get("saved_at_str") or "未知时间",
    ))

    # 与 V7 压缩后同理：模型手里的历史是旧的、磁盘是新的。
    # 补一份工作区真实清单，省得它靠历史记录去猜文件现在长什么样。
    # 措辞要贴合实际场景：这里是"恢复会话后"而不是"压缩后"。
    # 同一段提示被两处复用时，写死场景词迟早会说错——说错了就会误导模型。
    state_note = loop._workspace_state_note(reason="恢复会话后")
    if state_note:
        loop.history.add_note(state_note)

    return n_msgs


def describe(state: Dict[str, Any]) -> str:
    """一行摘要，给用户看"这个检查点里有什么"。"""
    return (
        f"任务「{state.get('task_name') or '未命名'}」"
        f" {len(state.get('messages') or [])} 条历史"
        f" / 压缩 {int(state.get('compact_count') or 0)} 次"
        f" / 累计 {int((state.get('usage') or {}).get('total_tokens') or 0):,} tokens"
        f" / 保存于 {state.get('saved_at_str') or '未知时间'}"
    )
