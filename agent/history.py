"""
对话历史与上下文管理。

这是"自己实现 Agent"最容易做砸、也最能体现功底的一块。本模块负责四件事：

1. **结构化存储**：消息以 OpenAI 格式保存，另外维护 kind 标记
   （system / user / assistant / tool / note / facts），压缩时据此决定保留哪些。

2. **预算控制**：
      - 写回执时立刻截断（工具输出往往是上下文杀手）；
      - 每轮估算 token 数，超过阈值自动压缩。

3. **自动压缩（compaction）**：
      - 保留 system + 最近 N 条消息（可配置）；
      - 中间部分交给模型做一次"结构化摘要"，替换成一条精简消息；
      - 摘要失败则退化为硬截断 —— 宁可丢信息，也不能让请求超窗报错。

4. **常驻事实层（V7）**：
      - 把仍然成立的关键事实（用户硬约束、技术选型、已失败的方案）单独抽出来；
      - 它**不参与**后续压缩，因此不会出现"摘要的摘要"式信息衰减；
      - 每次压缩时连同旧清单一起交给模型**整体重写**（不是追加），避免无限膨胀。

为什么不做"只保留最近 N 条"？
    那样会丢掉早期的关键决策（"用户要求用 Python 3.11"、"不要改 config.yaml"）。
    摘要能把这类约束沉淀下来。

那为什么摘要还不够，还要再抽一层"事实"？
    因为摘要本身在下一次压缩时会被**再次摘要**。每压一轮，早期信息就被再压一轮，
    几轮之后"不要改 config.yaml"这类硬约束就悄悄消失了 —— 而这恰恰是最不能丢的。
    把事实抽出来常驻、只让"过程"参与压缩，信息衰减就被截断在这一层。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import LLMError
from .llm import AssistantMessage

__all__ = ["History", "estimate_tokens", "count_messages_tokens", "token_counter_name"]

try:  # tiktoken 可选：装了就更精准，没装就走启发式
    import tiktoken  # type: ignore

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _ENC = None

_CJK = re.compile(r"[　-〿㐀-䶿一-鿿豈-﫿＀-￯]")
_OVERHEAD_PER_MESSAGE = 4  # role/分隔符等固定开销（OpenAI 的经验值）

# ---- 常驻事实层（见 History.facts 与 compact）----
# 要求摘要模型按这两节输出，我们据此把"事实"与"过程"拆开存放、分别对待。
FACTS_HEADING = "【关键事实】"
SUMMARY_HEADING = "【过程摘要】"
# 事实块上限。它不是摘要，而是常驻且不参与再压缩的短清单，必须小到能一直挂在
# 上下文里；模型被要求"按重要性从高到低"输出，所以超限时保留头部。
MAX_FACTS_CHARS = 1_500


def estimate_tokens(text: str) -> int:
    """估算文本的 token 数。

    优先级：tiktoken(cl100k_base) > 启发式。

    启发式经验值（与 cl100k_base 量级接近，偏保守）：
        英文/数字 ≈ 4 字符 1 token；中日韩 ≈ 1.3 字符 1 token。
    预算控制宁可"早压缩"也不要"低估到请求超窗"，所以略偏高估是安全的。

    注意：这里是「发给模型之前」的估算，用于上下文预算。真正精确的消耗
    来自 API 返回的 usage 字段（见 AgentLoop 的 /stats 面板），两者用途不同。
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
    return int(ascii_len / 4 + cjk / 1.3) + 1


def token_counter_name() -> str:
    """当前 count_messages_tokens 实际使用的计数方式（决定估算精度）。"""
    if _ENC is not None:
        return "tiktoken:cl100k_base（精确）"
    return "启发式估算（未安装 tiktoken，偏保守）"


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
        facts: 常驻事实清单（纯文本，不带 <key_facts> 包裹）。非空时以一条
            独立消息固定在 index 1，压缩时**永不进入被摘要的中间段**。
    """

    system_prompt: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    kinds: List[str] = field(default_factory=list)
    compact_count: int = 0
    facts: str = ""

    def __post_init__(self) -> None:
        if not self.messages:
            self.reset()

    # ---------------- 基础增删 ----------------
    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.kinds = ["system"]
        self.compact_count = 0
        self.facts = ""

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

    def drop_notes(self, prefix: str) -> int:
        """删掉所有以 prefix 开头的提示消息，返回删除条数。

        用于"可重建的事实"类回灌（例如工作区文件清单）：这类信息的价值只在于
        **最新**，旧副本留在历史里既白占上下文，又可能与现状矛盾——连续压缩时
        实测会在上下文里堆出好几份内容几乎相同、却都已过期的清单。
        """
        keep = [
            (m, k) for m, k in zip(self.messages, self.kinds)
            if not (isinstance(m.get("content"), str) and m["content"].startswith(prefix))
        ]
        removed = len(self.messages) - len(keep)
        if removed:
            self.messages = [m for m, _ in keep]
            self.kinds = [k for _, k in keep]
        return removed

    def add_note(self, content: str) -> None:
        """插入一条提示（纠错、预算提醒、重复调用警告等）。

        这里**不能用 role="system"**：OpenAI 兼容协议要求 system 消息位于列表开头，
        把 system 追加到末尾会让网关直接返回
        `400 System message must be at the beginning`。改用 user 角色回灌，
        对模型同样有强引导作用，且各家网关都接受。
        """
        self._append("note", {"role": "user", "content": content})

    def _append(self, kind: str, item: Dict[str, Any]) -> None:
        self.messages.append(item)
        self.kinds.append(kind)

    # ---------------- 查询 ----------------
    @property
    def tokens(self) -> int:
        return count_messages_tokens(self.messages)

    def payload(self) -> List[Dict[str, Any]]:
        """发给模型的消息列表。

        防御性处理：除首条外，任何 system 消息都降级为 user。
        这样即便将来有别的路径插入了 system，也不会触发网关的 400。
        """
        # 只允许第 0 条是 system。实测 NSCC 的 new-api 网关比 OpenAI 更严格：
        # 连"开头连续两条 system"（压缩摘要那种写法）都会返回同样的 400。
        out: List[Dict[str, Any]] = []
        for idx, msg in enumerate(self.messages):
            if idx > 0 and msg.get("role") == "system":
                msg = {**msg, "role": "user"}
            out.append(msg)
        return out

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

        策略：system + 常驻事实 + 过程摘要 + 最近 keep_recent 条。
        摘要由模型生成；若 `llm is None` 或模型调用失败，退化为"丢弃中间部分"的硬压缩。

        **为什么事实块不参与压缩**：早先的实现把整段历史（含上一次生成的摘要）
        一视同仁地再摘要一次，于是每压缩一轮，早期信息就被"再摘要"一轮——
        摘要的摘要会持续衰减，几轮之后"用户要求用 Python 3.11""不要改 config.yaml"
        这类硬约束就悄悄消失了，而它们恰恰是最不能丢的信息。
        把事实单独抽出来常驻、只让"过程"参与压缩，信息衰减就被截断在这一层。

        Returns:
            是否真的压缩了。
        """
        protected = 2 if self.facts else 1     # system（+ 常驻事实）不参与压缩
        if len(self.messages) <= keep_recent + protected + 1:
            return False

        head = self.messages[0]                    # system
        tail_start = max(protected, len(self.messages) - keep_recent)
        middle = self.messages[protected:tail_start]
        tail = self.messages[tail_start:]
        if not middle:
            return False

        # llm=None → 硬压缩：不调模型、直接丢弃更早的历史。
        # 摘要压缩存在"最小体积"（system + 事实 + 摘要 + 最近 N 条），预算小于它时
        # 压完反而更超，只能靠硬压缩把体量真正砍下来。
        if llm is None:
            new_facts, summary = "", None
        else:
            new_facts, summary = self._summarize(llm, middle, self.facts)
        if new_facts:
            self.facts = new_facts
        if summary is None:  # 硬压缩兜底
            summary = "[历史已截断] 早期对话被丢弃以腾出上下文空间；如需细节请重新读取相关文件。"

        # 摘要与事实都用 user 角色：网关只接受首条为 system，第 2 条 system 会触发 400
        rebuilt = [head]
        kinds = ["system"]
        if self.facts:
            rebuilt.append({"role": "user", "content": _wrap_facts(self.facts)})
            kinds.append("facts")
        rebuilt.append({"role": "user", "content": _wrap_summary(summary)})
        kinds.append("note")
        rebuilt.extend(tail)
        kinds.extend(self.kinds[tail_start:])

        self.messages = rebuilt
        self.kinds = kinds
        self.compact_count += 1
        return True

    def _summarize(self, llm, middle: Sequence[Dict[str, Any]],
                   old_facts: str = "") -> Tuple[str, Optional[str]]:
        """让模型把一段历史压缩成「事实 + 过程摘要」两部分。

        事实这块是**整体重写**而不是追加：把旧清单一起喂给模型，要求它输出
        "更新后仍然成立的完整清单"。追加式会让事实块单调膨胀，且被推翻的旧结论
        （"已试过 X、失败" → 后来 X 其实可行）永远删不掉，反而误导后续决策。

        Returns:
            (facts, summary)。模型没按要求分节时 facts 为空串——调用方据此
            **保留旧事实**：宁可这次没更新，也不能因为一次格式不合规就让
            一份已经确认过的约束清单凭空消失。
        """
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
            return "", None

        facts_instruction = (
            "\n当前已有的事实清单如下，请在它的基础上**整体更新后重写**（不是追加）：\n"
            f"{old_facts}\n"
            if old_facts else ""
        )
        prompt = (
            "下面是智能体与用户之间的一段历史对话，请压缩成两段结构化输出。\n\n"
            f"第一段，标题必须严格为「{FACTS_HEADING}」：\n"
            "列出**到现在仍然成立**的关键事实，供后续继续工作。包括：用户的硬性约束、"
            "项目结构与关键技术选型、已经尝试过且失败的方案（避免重复踩坑）、当前待办。\n"
            "要求：只写仍然成立的；已被推翻的请删除而非保留；按重要性从高到低排序；"
            "分条列出，不超过 15 条、400 字。"
            + facts_instruction +
            f"\n第二段，标题必须严格为「{SUMMARY_HEADING}」：\n"
            "概述这段历史里**发生过什么**（做了哪些操作、结果如何），用中文、300 字以内。"
            "不要复述工具的原始输出，只保留结论。\n\n"
            "===== 历史开始 =====\n" + "\n".join(transcript)[:12000] + "\n===== 历史结束 ====="
        )
        try:
            msg = llm.chat(
                [{"role": "system", "content": "你是精确的对话摘要器，只输出摘要本身，不要寒暄。"},
                 {"role": "user", "content": prompt}],
                tools=None,
            )
        except LLMError:
            return "", None
        text = (msg.content or "").strip()
        if not text:
            return "", None
        return _split_facts_and_summary(text)

    # ---------------- 落盘 ----------------
    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(m, ensure_ascii=False) for m in self.messages)


def _split_facts_and_summary(text: str) -> Tuple[str, Optional[str]]:
    """把模型输出拆成（事实, 过程摘要）。

    模型没按格式分节时返回 ("", 原文)，调用方据此保留旧事实。
    """
    if FACTS_HEADING not in text:
        return "", text
    preamble, _, rest = text.partition(FACTS_HEADING)
    if SUMMARY_HEADING in rest:
        facts_part, _, summary_part = rest.partition(SUMMARY_HEADING)
    else:
        facts_part, summary_part = rest, ""
    return _clean_facts(facts_part), (summary_part.strip() or preamble.strip() or None)


def _clean_facts(raw: str) -> str:
    """规范化事实清单：去空行、统一项目符号、按字符上限截断。

    超限时保留**前面**——模型被要求按重要性降序输出，所以头部最重要；
    同时补一条提示，让下一次压缩知道该收敛了。
    """
    lines: List[str] = []
    for ln in (raw or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        ln = re.sub(r"^[-*+]\s*", "- ", ln)
        if not ln.startswith("- "):
            ln = "- " + ln
        lines.append(ln)
    if not lines:
        return ""
    text = "\n".join(lines)
    if len(text) <= MAX_FACTS_CHARS:
        return text

    kept: List[str] = []
    used = 0
    for ln in lines:
        if used + len(ln) + 1 > MAX_FACTS_CHARS:
            break
        kept.append(ln)
        used += len(ln) + 1
    kept.append("- …（事实块已满，下次压缩请只保留仍然成立且最重要的条目）")
    return "\n".join(kept)


def _wrap_facts(text: str) -> str:
    """把事实清单包成常驻消息（明确告知它不会被再次摘要）。"""
    return (
        "<key_facts>\n"
        "以下是本次会话已确认的关键事实，**在后续上下文压缩中始终保留、不会被再次摘要**：\n"
        f"{text}\n"
        "</key_facts>"
    )


def _wrap_summary(text: str) -> str:
    return (
        "<conversation_summary>\n"
        "以下是此前对话的压缩摘要（为节省上下文，原始消息已被移除）：\n"
        f"{text}\n"
        "</conversation_summary>"
    )
