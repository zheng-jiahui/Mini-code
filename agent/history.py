"""
对话历史与上下文管理。

这是"自己实现 Agent"最容易做砸、也最能体现功底的一块。本模块负责三件事：

1. **结构化存储**：消息以 OpenAI 格式保存，另外维护 kind 标记
   （system / user / assistant / tool / note），压缩时据此决定保留哪些。

2. **预算控制**：
      - 写回执时立刻截断（工具输出往往是上下文杀手）；
      - 每轮估算 token 数，超过阈值自动压缩。

3. **自动压缩（compaction）**：
      - 保留 system + 最近 N 条消息（可配置）；
      - 中间部分交给模型做一次"结构化摘要"，替换成一条精简消息；
      - 摘要失败则退化为硬截断 —— 宁可丢信息，也不能让请求超窗报错。

为什么不做"只保留最近 N 条"？
    那样会丢掉早期的关键决策（"用户要求用 Python 3.11"、"不要改 config.yaml"）。
    摘要能把这类约束沉淀下来。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .errors import LLMError
from .llm import AssistantMessage

__all__ = ["History", "estimate_tokens", "count_messages_tokens"]

try:  # tiktoken 可选：装了就更精准，没装就走启发式
    import tiktoken  # type: ignore

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _ENC = None

_CJK = re.compile(r"[　-〿㐀-䶿一-鿿豈-﫿＀-￯]")
_OVERHEAD_PER_MESSAGE = 4  # role/分隔符等固定开销（OpenAI 的经验值）


def estimate_tokens(text: str) -> int:
    """估算文本的 token 数。

    优先 tiktoken；不可用时用启发式：
        英文/数字 ≈ 4 字符 1 token；中日韩 ≈ 1 字符 1.5 token（保守偏高估）。
    """
    if not text:
        return 0
    if _ENC is not None:
        try:
            return len(_ENC.encode(text))
        except Exception:  # pragma: no cover
            pass
    cjk = len(_CJK.findall(text))
    ascii_len = len(text) - cjk
    return int(ascii_len / 4 + cjk * 1.5) + 1


def count_messages_tokens(messages: Sequence[Dict[str, Any]]) -> int:
    """估算整个消息列表的 token 数。"""
    total = 0
    for m in messages:
        total += _OVERHEAD_PER_MESSAGE
        content = m.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):  # 多模态内容块
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total += estimate_tokens(str(part["text"]))
        for tc in m.get("tool_calls") or []:
            total += estimate_tokens(json.dumps(tc, ensure_ascii=False))
        total += estimate_tokens(str(m.get("name") or ""))
    return total


@dataclass
class History:
    """对话历史。

    Attributes:
        system_prompt: 系统提示词，永远占据 index 0，压缩时不动。
        messages: OpenAI 风格消息列表（含 system）。
        kinds: 与 messages 等长，标记每条消息的类别。
    """

    system_prompt: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    kinds: List[str] = field(default_factory=list)
    compact_count: int = 0

    def __post_init__(self) -> None:
        if not self.messages:
            self.reset()

    # ---------------- 基础增删 ----------------
    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.kinds = ["system"]
        self.compact_count = 0

    def add_user(self, content: str) -> None:
        self._append("user", {"role": "user", "content": content})

    def add_assistant(self, msg: AssistantMessage) -> None:
        self._append("assistant", msg.to_history_item())

    def add_tool_result(self, call_id: str, name: str, content: str, style: str = "native") -> None:
        if style == "native":
            item = {"role": "tool", "tool_call_id": call_id, "content": content}
        else:
            item = {"role": "user", "content": f'<tool_result name="{name}">\n{content}\n</tool_result>'}
        self._append("tool", item)

    def add_note(self, content: str) -> None:
        """插入系统级提示（纠错、预算提醒、重复调用警告等）。"""
        self._append("note", {"role": "system", "content": content})

    def _append(self, kind: str, item: Dict[str, Any]) -> None:
        self.messages.append(item)
        self.kinds.append(kind)

    # ---------------- 查询 ----------------
    @property
    def tokens(self) -> int:
        return count_messages_tokens(self.messages)

    def payload(self) -> List[Dict[str, Any]]:
        """发给模型的消息列表。"""
        return list(self.messages)

    def turn_count(self) -> int:
        return sum(1 for k in self.kinds if k == "user")

    def last_assistant_text(self) -> str:
        for m in reversed(self.messages):
            if m.get("role") == "assistant":
                return m.get("content") or ""
        return ""

    def summary_of_changes(self, changes: Sequence[Dict[str, Any]]) -> str:
        """把本次会话的文件变更整理成文本。"""
        if not changes:
            return "（本次会话未修改任何文件）"
        lines = []
        for c in changes:
            lines.append(f"- [{c.get('kind')}] {c.get('detail')}")
        return "\n".join(lines)

    # ---------------- 压缩 ----------------
    def needs_compaction(self, budget: int, threshold: float = 0.75) -> bool:
        return self.tokens >= int(budget * threshold)

    def compact(self, llm, keep_recent: int = 8, budget: Optional[int] = None) -> bool:
        """压缩历史。

        策略：system + 摘要(middle) + 最近 keep_recent 条。
        摘要由模型生成；若模型调用失败，退化为"丢弃中间部分"的硬压缩。

        Returns:
            是否真的压缩了。
        """
        if len(self.messages) <= keep_recent + 2:
            return False

        head = self.messages[0]                    # system
        tail_start = max(1, len(self.messages) - keep_recent)
        middle = self.messages[1:tail_start]
        tail = self.messages[tail_start:]
        if not middle:
            return False

        summary = self._summarize(llm, middle)
        if summary is None:  # 硬压缩兜底
            summary = "[历史已截断] 早期对话被丢弃以腾出上下文空间；如需细节请重新读取相关文件。"

        self.messages = [head, {"role": "system", "content": _wrap_summary(summary)}, *tail]
        self.kinds = ["system", "note", *self.kinds[tail_start:]]
        self.compact_count += 1
        return True

    def _summarize(self, llm, middle: Sequence[Dict[str, Any]]) -> Optional[str]:
        """让模型把一段历史压缩成结构化摘要。失败返回 None。"""
        transcript = []
        for m in middle:
            role = m.get("role")
            content = m.get("content") or ""
            if isinstance(content, str):
                content = content[:1200]
            if role == "assistant" and m.get("tool_calls"):
                calls = ", ".join(
                    f"{tc.get('function', {}).get('name')}({tc.get('function', {}).get('arguments', '')[:120]})"
                    for tc in m["tool_calls"]
                )
                content = (content + " " if content else "") + f"[调用了：{calls}]"
            if content:
                transcript.append(f"{role}: {content}")

        if not transcript:
            return None

        prompt = (
            "下面是智能体与被用户任务之间的一段历史对话，请压缩为结构化摘要，供后续继续工作。\n"
            "必须保留：① 用户的原始需求与硬性约束；② 已经确认的关键事实（项目结构、依赖、版本）；\n"
            "③ 已经尝试过且失败的方案（避免重复踩坑）；④ 当前进展与待办。\n"
            "不要复述工具原始输出，只保留结论。用中文，控制在 400 字以内，分条列出。\n\n"
            "===== 历史开始 =====\n" + "\n".join(transcript)[:12000] + "\n===== 历史结束 ====="
        )
        try:
            msg = llm.chat(
                [{"role": "system", "content": "你是精确的对话摘要器，只输出摘要本身，不要寒暄。"},
                 {"role": "user", "content": prompt}],
                tools=None,
            )
        except LLMError:
            return None
        text = (msg.content or "").strip()
        return text or None

    # ---------------- 落盘 ----------------
    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(m, ensure_ascii=False) for m in self.messages)


def _wrap_summary(text: str) -> str:
    return (
        "<conversation_summary>\n"
        "以下是此前对话的压缩摘要（为节省上下文，原始消息已被移除）：\n"
        f"{text}\n"
        "</conversation_summary>"
    )
