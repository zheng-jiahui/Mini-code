"""任务级质量指标：回答「这个 agent 到底做对了没有」。

为什么需要它
------------
`/stats` 原有的成本面板只回答**效率**问题（花了多少 token、多少秒、时间花在哪一段），
回答不了**质量**问题：任务真的完成了吗？中途返工了几次？卡住过吗？

而质量指标是做任何改进的**前提**：改一句 system prompt、调整一次压缩策略，
如果没有可比的量化口径，就分不清是"变好了"还是"只是这次任务碰巧简单"。
所以这一层的定位不是"多显示几行数字"，而是给后续每一次调优提供对照基准。

口径设计的三条原则
------------------
1. **区分"模型自述完成"与"产物真的能跑"**。模型调用 finish 不等于任务完成——
   本项目已经有「假完成拦截」（改过文件却没验证就收尾要拦下来）在守这一条，
   这里把它延续到度量层：结局分布如实列出 finish / model_final / no_progress / …，
   不把 `model_final`（只吐正文、可能是"我做不了"）与 `finish` 混为一谈。
2. **自修复要按"回合"计，而不是按"失败次数"计**。失败 3 次才修好，和
   失败 1 次就修好，是两种完全不同的能力水平。所以定义为：
   一次「修复回合」= 一段连续的验证失败 + 紧随其后的一次验证成功；
   该回合的轮数 = 这段连续失败的次数。末尾仍在失败的（未修好）单独统计，
   不计入"平均修复轮数"——否则分母里混进没修好的，指标会被稀释得看不出问题。
3. **不采信单一信号**。"返工"用同一路径被写入的次数来近似，它不完美
   （改 3 次也可能是合理迭代），但与自修复轮数、结局分布交叉看就有意义：
   返工多 + 修复回合多 = 模型对当前状态掌握不准，通常指向上下文信息不足。
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

__all__ = ["TaskRecord", "SessionMetrics", "SUCCESS_REASONS"]

# 视为"成功收尾"的结局。注意 model_final 在列但置信度低于 finish：
# 后者是模型显式调用 finish（且已通过假完成拦截），前者只是没有再发工具调用。
SUCCESS_REASONS = ("finish", "model_final")

# 结局的人话解释，面板里直接给出来，避免看的人对着枚举名猜含义
REASON_LABELS = {
    "finish": "显式完成",
    "model_final": "正文收尾（未调用 finish）",
    "max_steps": "步数用尽",
    "no_progress": "原地打转",
    "too_many_errors": "连续失败过多",
    "parse_failed": "输出无法解析",
    "llm_error": "模型调用报错",
    "internal_error": "内部错误",
    "aborted": "被中断",
    "unknown": "未知",
}


@dataclass
class TaskRecord:
    """一次任务运行的量化记录。"""

    task: str = ""
    finish_reason: str = "unknown"
    steps: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    elapsed: float = 0.0
    tokens: int = 0
    compacted: int = 0
    files_changed: int = 0

    # 自修复（只统计 run_command 这一类"验证动作"的成败序列）
    verify_runs: int = 0            # 跑过的验证命令总数
    verify_failures: int = 0        # 其中失败的次数
    repair_rounds: int = 0          # 修成功的回合数（失败→成功算一回合）
    repair_attempts: int = 0        # 这些回合累计花掉的失败轮数
    unresolved_failures: int = 0    # 到任务结束仍在失败的验证次数

    rework_files: int = 0           # 被写了不止一次的文件数

    @property
    def succeeded(self) -> bool:
        return self.finish_reason in SUCCESS_REASONS

    @property
    def tool_error_rate(self) -> float:
        """工具调用失败率。没有调用时返回 0 而不是报错——空会话不该除以零。"""
        return (self.tool_errors / self.tool_calls) if self.tool_calls else 0.0

    @property
    def avg_repair_rounds(self) -> Optional[float]:
        """平均每次成功修复花几轮；从未成功修复过返回 None（不是 0，0 会被误读成"一次就修好"）。"""
        return (self.repair_attempts / self.repair_rounds) if self.repair_rounds else None

    def to_dict(self) -> Dict[str, object]:
        d = dict(self.__dict__)
        d["succeeded"] = self.succeeded
        d["tool_error_rate"] = round(self.tool_error_rate, 4)
        d["avg_repair_rounds"] = self.avg_repair_rounds
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "TaskRecord":
        """从序列化数据重建。只取当前 dataclass 认识的字段，
        这样早期检查点里多出来的键不会让恢复直接崩掉（反之缺键则用默认值）。"""
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class SessionMetrics:
    """跨任务累计的质量指标。会话级（REPL 多轮）聚合，也可跨进程恢复后继续累计。"""

    def __init__(self) -> None:
        self.tasks: List[TaskRecord] = []

        # 以下为「当前任务」的暂存状态，finish_task 时结算进 TaskRecord
        self._started = time.time()
        self._task = ""
        self._verify_seq: List[bool] = []     # 验证动作（run_command）的成败序列
        self._tokens_at_start: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # 采集
    # ------------------------------------------------------------------
    def start_task(self, task: str, usage: Optional[Dict[str, int]] = None) -> None:
        """开始一个新任务。usage 传入会话级 token 计数，用于算出本任务的增量。"""
        self._task = task
        self._started = time.time()
        self._verify_seq = []
        self._tokens_at_start = dict(usage or {})

    def record_verify(self, ok: bool) -> None:
        """记录一次验证动作（run_command）的结果。"""
        self._verify_seq.append(bool(ok))

    def finish_task(
        self,
        finish_reason: str,
        *,
        steps: int = 0,
        tool_calls: int = 0,
        tool_errors: int = 0,
        compacted: int = 0,
        usage: Optional[Dict[str, int]] = None,
        changes: Optional[Sequence[Dict[str, object]]] = None,
    ) -> TaskRecord:
        """结算当前任务并归档。返回值便于调用方直接用于报告。"""
        rec = TaskRecord(
            task=self._task,
            finish_reason=finish_reason,
            steps=steps,
            tool_calls=tool_calls,
            tool_errors=tool_errors,
            elapsed=time.time() - self._started,
            tokens=self._token_delta(usage or {}),
            compacted=compacted,
        )
        rec.verify_runs = len(self._verify_seq)
        rec.verify_failures = sum(1 for ok in self._verify_seq if not ok)
        rec.repair_rounds, rec.repair_attempts, rec.unresolved_failures = self._repair_stats()
        rec.files_changed, rec.rework_files = _file_stats(changes or [])

        self.tasks.append(rec)
        self._verify_seq = []
        return rec

    # ------------------------------------------------------------------
    # 聚合
    # ------------------------------------------------------------------
    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def success_count(self) -> int:
        return sum(1 for t in self.tasks if t.succeeded)

    @property
    def success_rate(self) -> float:
        return (self.success_count / self.task_count) if self.task_count else 0.0

    def outcome_counts(self) -> Dict[str, int]:
        """结局分布：出现过的结局 → 次数（按次数降序）。"""
        counts: Dict[str, int] = {}
        for t in self.tasks:
            counts[t.finish_reason] = counts.get(t.finish_reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

    def aggregate(self) -> Dict[str, object]:
        """跨任务汇总。所有比值都在分母为 0 时给出 None 而非 0，
        因为"没有数据"和"数据是 0"在决策上完全不同。"""
        n = self.task_count
        if n == 0:
            return {"tasks": 0}
        total_calls = sum(t.tool_calls for t in self.tasks)
        total_err = sum(t.tool_errors for t in self.tasks)
        rounds = sum(t.repair_rounds for t in self.tasks)
        attempts = sum(t.repair_attempts for t in self.tasks)
        return {
            "tasks": n,
            "succeeded": self.success_count,
            "success_rate": self.success_rate,
            "avg_steps": sum(t.steps for t in self.tasks) / n,
            "avg_tool_calls": total_calls / n,
            "avg_elapsed": sum(t.elapsed for t in self.tasks) / n,
            "avg_tokens": sum(t.tokens for t in self.tasks) / n,
            "tool_calls": total_calls,
            "tool_errors": total_err,
            "tool_error_rate": (total_err / total_calls) if total_calls else None,
            "repair_rounds": rounds,
            "repair_attempts": attempts,
            "avg_repair_rounds": (attempts / rounds) if rounds else None,
            "unresolved_failures": sum(t.unresolved_failures for t in self.tasks),
            "rework_files": sum(t.rework_files for t in self.tasks),
            "compacted": sum(t.compacted for t in self.tasks),
        }

    # ------------------------------------------------------------------
    # 持久化（跨进程续跑时保持口径连续）
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {"tasks": [t.to_dict() for t in self.tasks]}

    def restore(self, data: Optional[Dict[str, object]]) -> int:
        """从检查点恢复历史任务记录，返回恢复的条数。

        与成本统计同理：指标口径必须跨进程连续，否则恢复会话后成功率会被
        重新计算，长任务的统计就断掉了。损坏的记录跳过而不是整批丢弃——
        一条记录坏了不该带走所有历史。
        """
        if not data or not isinstance(data, dict):
            return 0
        raw = data.get("tasks") or []
        if not isinstance(raw, list):
            return 0
        n = 0
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                self.tasks.append(TaskRecord.from_dict(item))
                n += 1
            except (TypeError, ValueError):
                continue
        return n

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def render_panel(self, title: str = "【质量指标 /stats】") -> str:
        agg = self.aggregate()
        if not agg.get("tasks"):
            return title + "\n  本次会话还没有跑过任务。"
        lines = [title]

        rate = agg["success_rate"] * 100
        lines.append(
            f"  任务数：{agg['tasks']}    成功：{agg['succeeded']}（{rate:.1f}%）"
            f"    失败：{agg['tasks'] - agg['succeeded']}"
        )

        dist = "、".join(
            f"{REASON_LABELS.get(r, r)} {c}" for r, c in self.outcome_counts().items()
        )
        lines.append(f"  结局分布：{dist}")

        lines.append(
            f"  平均每任务：步数 {agg['avg_steps']:.1f}    "
            f"工具调用 {agg['avg_tool_calls']:.1f}    "
            f"耗时 {agg['avg_elapsed']:.1f}s    "
            f"token {agg['avg_tokens']:,.0f}"
        )

        err_rate = agg["tool_error_rate"]
        lines.append(
            "  工具失败率："
            + ("无工具调用" if err_rate is None
               else f"{err_rate * 100:.1f}%（{agg['tool_errors']} / {agg['tool_calls']}）")
        )

        # 自修复：这是最能反映"agent 是否具备纠错能力"的一项
        if agg["repair_rounds"]:
            lines.append(
                f"  自修复：成功修复 {agg['repair_rounds']} 次，"
                f"平均 {agg['avg_repair_rounds']:.1f} 轮/次（累计 {agg['repair_attempts']} 次失败尝试）"
            )
        elif agg["tool_errors"]:
            lines.append(
                f"  自修复：{agg['tool_errors']} 次工具失败中没有一次修回来"
                f"（连续失败 {agg['unresolved_failures']} 次仍未通过）"
            )
        else:
            lines.append("  自修复：本次会话未出现工具失败")

        if agg["rework_files"]:
            lines.append(
                f"  返工文件：{agg['rework_files']} 个（同一文件被写入多次，"
                "通常意味着模型对当前文件内容掌握不准）"
            )
        if agg["compacted"]:
            lines.append(f"  上下文压缩：累计 {agg['compacted']} 次")
        return "\n".join(lines)

    def render_table(self) -> str:
        """逐任务的明细表，供评测报告使用。"""
        if not self.tasks:
            return "（无任务记录）"
        head = f"{'#':>2}  {'结局':<8} {'步':>3} {'调用':>4} {'失败':>4} {'耗时':>7} {'token':>8}  任务"
        rows = [head, "-" * min(100, max(len(head), 60))]
        for i, t in enumerate(self.tasks, 1):
            rows.append(
                f"{i:>2}  {t.finish_reason:<8} {t.steps:>3} {t.tool_calls:>4} "
                f"{t.tool_errors:>4} {t.elapsed:>6.1f}s {t.tokens:>8,}  {_short(t.task, 40)}"
            )
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _token_delta(self, usage: Dict[str, int]) -> int:
        """本任务消耗的 token = 会话累计 usage 的增量。

        用增量而不是"本次返回的 usage"：后者只是最后一轮的，而一轮任务会调很多次模型。
        """
        total = usage.get("total_tokens", 0)
        if total:
            return int(total) - int(self._tokens_at_start.get("total_tokens", 0))
        # 网关没给 total 时，用 prompt+completion 兜底
        now = int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
        return now - (
            int(self._tokens_at_start.get("prompt_tokens", 0))
            + int(self._tokens_at_start.get("completion_tokens", 0))
        )

    def _repair_stats(self):
        """把验证成败序列切成"修复回合"。

        定义：一段连续失败 + 紧随其后的一次成功 = 一个修复回合，
        该回合的轮数 = 这段连续失败的长度。序列末尾仍在失败的属于"未修复"，
        不计入平均修复轮数——否则未修好的失败会把平均值稀释，掩盖问题。
        """
        rounds = 0
        attempts = 0
        pending = 0          # 当前这段连续失败的长度
        for ok in self._verify_seq:
            if ok:
                if pending:
                    rounds += 1
                    attempts += pending
                pending = 0
            else:
                pending += 1
        return rounds, attempts, pending      # pending 即"到结束仍未修复"的次数


# ----------------------------------------------------------------------------
# 辅助
# ----------------------------------------------------------------------------
def _file_stats(changes: Sequence[Dict[str, object]]):
    """(改动过的文件数, 被写了不止一次的文件数)。

    只看真正改动文件的操作（write / edit）；run_command 之类没有 path，天然被排除。
    """
    writes: Dict[str, int] = {}
    for c in changes:
        if c.get("kind") not in ("write", "edit"):
            continue
        path = c.get("path")
        if not path:
            continue
        writes[str(path)] = writes.get(str(path), 0) + 1
    rework = sum(1 for n in writes.values() if n > 1)
    return len(writes), rework


def _short(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"
