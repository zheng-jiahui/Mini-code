"""
★ 主循环（Agent Loop）—— 整个项目的中枢。

职责边界非常清晰：
    本模块只做"编排"——调用模型、解析输出、执行工具、把结果喂回去、判断何时停下。
    它不认识任何一个具体工具，也不认识任何一家模型厂商。

循环的核心状态机
----------------
    ┌─────────────────────────────────────────────────────┐
    │  用户输入 → history.add_user(task)                   │
    └──────────────────────┬──────────────────────────────┘
                           ▼
                 ┌───────────────────┐
        ┌───────▶│ 预算检查/上下文压缩 │
        │        └─────────┬─────────┘
        │                  ▼
        │        ┌───────────────────┐   失败且可重试
        │        │  backend.chat()   │◀──────────────┐
        │        └─────────┬─────────┘               │
        │                  ▼                          │
        │        ┌───────────────────┐                │
        │        │  parser.parse()   │──有 issues─────┘
        │        └─────────┬─────────┘  (注入纠错提示)
        │                  ▼
        │        ┌───────────────────┐
        │        │ 有工具调用？       │──否──▶ 判定为收尾回答 ──▶ 结束
        │        └─────────┬─────────┘
        │                  ▼ 是
        │        ┌───────────────────┐
        │        │ registry.execute()│（异常→ok=False 回执，绝不抛出）
        │        └─────────┬─────────┘
        │                  ▼
        │        ┌───────────────────┐
        │        │ 结果回灌 history   │
        │        └─────────┬─────────┘
        │                  ▼
        │        ┌───────────────────┐
        └────────┤ 终止条件检查       │──触发──▶ 结束
                 │ ·finish 工具      │
                 │ ·步数/预算上限     │
                 │ ·连续失败/无进展   │
                 └───────────────────┘

终止条件的四种触发方式：
    1. 模型主动调用 finish（最理想）
    2. 模型给出不带工具调用的收尾回答（文本通道下以 FINAL: 开头更明确）
    3. 触发配额：步数 / 上下文 / 连续失败 / 重复调用
    4. 用户 Ctrl-C 中断
"""

from __future__ import annotations

import json
import re
import shutil
import time
import concurrent.futures as _cf
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from .config import AgentConfig
from .errors import Aborted, BudgetExceeded, LLMError
from .history import History
from .llm import AssistantMessage, LLMBackend, ToolCall
from .parser import ToolCallParser
from .prompts import BUDGET_WARNING, NO_PROGRESS_HINT, build_system_prompt, build_task_message
from .profile import detect_project_profile, format_project_profile, looks_like_complex_task
from .tools import build_tool_context
from .tools.base import ToolContext, ToolRegistry, ToolResult
from .tools.meta import FINISH_SENTINEL

__all__ = ["RunResult", "AgentLoop"]

# 只读工具：无副作用、互不依赖，可在同一轮里并行发出（V5 自主性）
_READONLY_TOOLS = frozenset({"read_file", "list_dir", "grep_search", "find_files", "diff"})

_EMPTY_OUTPUT_HINT = (
    "你上一次的回复中没有包含任何工具调用，也没有明确的收尾标记。\n"
    "请二选一：① 继续工作 → 发起工具调用；② 已完成 → 调用 finish 给出总结。"
)


def _slugify(text: str, max_chars: int = 20) -> str:
    """把任务描述压成可作目录名的短标识。

    保留中文与英文单词，但剔除三类噪音：路径非法字符、源码扩展名（.py 之类）、
    中英文标点。否则任务里提到的 "xxx.py" 会把目录名搞成 "…-bubble_sort.py，可"。
    """
    s = (text or "").strip()
    s = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", s)
    s = re.sub(r"\.(py|js|jsx|ts|tsx|java|cpp|c|h|go|rs|html|css|md|json|ya?ml|txt|sh)\b",
               " ", s, flags=re.IGNORECASE)
    s = re.sub(r"[]，。！？、；：""''（）【】《》,.!?;:'\"(){}[\\~`@#$%^&+=]+", " ", s)
    s = re.sub(r"\s+", "-", s.strip())
    s = re.sub(r"-+", "-", s).strip("-")
    if len(s) <= max_chars:
        return s
    # 超长时丢掉末尾那个被截断的半截词，避免得到 "…-打印-Hello-M" 这种断尾
    s = s[:max_chars]
    if "-" in s:
        s = s[: s.rfind("-")]
    return s.rstrip("-")


# ----------------------------------------------------------------------------
# 运行结果
# ----------------------------------------------------------------------------
@dataclass
class RunResult:
    """一次任务运行的完整结果。"""

    answer: str = ""
    finish_reason: str = "unknown"
    steps: int = 0
    tool_calls: int = 0
    errors: int = 0
    changes: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    elapsed: float = 0.0
    compacted: int = 0
    error_message: str = ""
    task_dir: str = ""            # 任务目录：workplace/{任务名}/，始终是最新的
    backup_dir: str = ""          # 本次快照：.agent_backups/{任务名}_{时间戳}_{第N次}/

    @property
    def succeeded(self) -> bool:
        return self.finish_reason in ("finish", "model_final")

    def stats_line(self) -> str:
        return (
            f"步数={self.steps} 工具调用={self.tool_calls} 失败={self.errors} "
            f"压缩={self.compacted} 耗时={self.elapsed:.1f}s 结束原因={self.finish_reason}"
        )


# ----------------------------------------------------------------------------
# 主循环
# ----------------------------------------------------------------------------
class AgentLoop:
    """编程智能体的执行引擎。一个 AgentLoop 实例持有完整会话状态（可多轮复用）。"""

    def __init__(
        self,
        config: AgentConfig,
        profile,                       # LLMProfile
        backend: LLMBackend,
        registry: ToolRegistry,
        console=None,
        *,
        system_prompt: Optional[str] = None,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.config = config
        self.profile = profile
        self.backend = backend
        self.registry = registry
        self.console = console
        self.on_event = on_event

        self.native = bool(profile.native_tools) and backend.supports_native_tools
        self.parser = ToolCallParser(registry, use_native=self.native)
        self._project_profile = ""   # V5：自动识别的项目画像段落（随任务目录刷新）
        self.system_prompt = system_prompt or build_system_prompt(
            tool_list=registry.describe(),
            workspace=str(config.resolved_workspace()),
            native_tools=self.native,
            restrict_to_workspace=config.restrict_to_workspace,
            project_profile=self._project_profile,
        )
        self.history = History(self.system_prompt)
        self.ctx: ToolContext = build_tool_context(config, console=console, session={})

        # 任务级工作区：workspace_root 是"所有生成代码的家"，
        # 每个具体任务会在它下面拥有一个独立子目录（见 prepare_task_dir）。
        # 取自 config.workspace_root 而非 resolved_workspace()，因为后者会随任务切换而变。
        root = getattr(config, "workspace_root", None) or config.workspace
        self.workspace_root: Path = Path(str(root)).expanduser().resolve()
        self._task_dir: Optional[Path] = None
        self._task_name: Optional[str] = None

        self._fingerprints: Deque[str] = deque(maxlen=6)
        self._stale_steps = 0
        self._consecutive_errors = 0
        self._ran_command_this_run = False   # 本任务是否已跑过 run_command（假完成拦截用）
        self._finish_blocked = 0             # finish 被"先验证"拦截的次数（最多 2 次，防死循环）
        self._consecutive_run_failures = 0   # 连续 run_command 失败次数（自修复预算）
        self._usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # V3 成本治理：会话级画像（跨任务累计，REPL 里 /stats 给出整体视图）
        self._tool_timings: Dict[str, float] = {}       # 工具名 → 累计耗时（秒）
        self._tool_calls_by_name: Dict[str, int] = {}  # 工具名 → 调用次数
        self._total_tool_calls = 0
        self._output_compressions = 0     # 工具回执因超长被智能压缩的次数
        self._session_started = time.time()

    # ------------------------------------------------------------------
    # 任务级工作区
    # ------------------------------------------------------------------
    @property
    def task_dir(self) -> Optional[Path]:
        """当前任务的子目录（还没跑过任务时为 None）。"""
        return self._task_dir

    @property
    def task_name(self) -> Optional[str]:
        """当前任务名（还没跑过任务时为 None）。"""
        return self._task_name

    def prepare_task_dir(self, task: str, task_name: Optional[str] = None,
                         force_new: bool = False) -> Path:
        """为本次任务准备目录 workplace/{任务名}/，同名任务复用同一目录。

        任务名优先级：显式传入的 task_name > 上次指定的名字 > 由任务描述自动提取。
        workplace 下始终是该任务的**最新**代码；历史版本由 snapshot_to_backups()
        归档到与 workplace 同级的 .agent_backups/ 里。

        同一会话内的追问会沿用当前目录，直到 /new 开启新任务——否则追问
        （如"再改成降序"）时模型会看不到上一轮刚生成的文件。
        """
        if not getattr(self.config, "per_task_dir", True):
            # 关闭任务子目录：所有任务共用工作区根目录
            self._task_dir = self.workspace_root
            self._task_name = self.workspace_root.name
            return self._task_dir

        if self._task_dir and not force_new:
            return self._task_dir

        raw = (task_name or self._task_name or "").strip()
        name = _slugify(raw, max_chars=32) if raw else _slugify(task, max_chars=24)
        if not name:
            name = "未命名任务"

        self._task_name = name
        self._task_dir = self.workspace_root / name
        self._task_dir.mkdir(parents=True, exist_ok=True)   # 已存在就复用，保留最新代码
        self._apply_task_dir()
        return self._task_dir

    # ------------------------------------------------------------------
    # 历史备份
    # ------------------------------------------------------------------
    @property
    def backup_root(self) -> Path:
        """备份根目录：与 workplace 同级，例如 项目根/.agent_backups。"""
        name = getattr(self.config, "backup_dir", None) or ".agent_backups"
        return self.workspace_root.parent / name

    def snapshot_to_backups(self) -> Optional[Path]:
        """把当前任务目录的完整快照归档到 .agent_backups/{任务名}_{时间戳}_{第N次}/。

        只在本次任务确实改动过文件时才备份；N 取该任务已有备份数 + 1，
        因此同一任务反复生成会依次得到"第1次""第2次"……
        """
        if self._task_dir is None or not self._task_dir.exists():
            return None
        changes = self.ctx.session.get("changes", []) if self.ctx else []
        if not changes:
            return None                      # 本次没动过文件，无需备份

        name = self._task_name or self._task_dir.name
        root = self.backup_root
        seq = self._next_backup_seq(root, name)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        dest = root / f"{name}_{stamp}_第{seq}次"
        dest.mkdir(parents=True, exist_ok=True)

        for item in self._task_dir.iterdir():
            if item.name.startswith("."):    # 跳过 .agent_sessions 等内部目录
                continue
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
        return dest

    @staticmethod
    def _next_backup_seq(backup_root: Path, task_name: str) -> int:
        """该任务已有的备份数 + 1（目录名形如 任务名_时间戳_第N次）。"""
        if not backup_root.exists():
            return 1
        prefix = f"{task_name}_"
        return sum(1 for p in backup_root.iterdir()
                   if p.is_dir() and p.name.startswith(prefix)) + 1

    def _apply_task_dir(self) -> None:
        """把沙箱上下文与系统提示词切换到当前任务目录。"""
        assert self._task_dir is not None
        self.config.workspace = str(self._task_dir)
        # 复用 session，保住 changes / command_guard 等会话状态，只换沙箱根目录
        self.ctx = build_tool_context(
            self.config, console=self.console, session=self.ctx.session if self.ctx else {}
        )
        self.ctx.session.setdefault("changes", [])
        self.ctx.session["task_name"] = self._task_name   # 供 rollback 工具定位快照

        # V5 项目画像：每次切入任务目录都重新扫描，把语言/框架/测试命令喂给系统提示词
        prof = detect_project_profile(self._task_dir)
        self._project_profile = format_project_profile(prof)

        self.system_prompt = build_system_prompt(
            tool_list=self.registry.describe(),
            workspace=str(self._task_dir),
            native_tools=self.native,
            restrict_to_workspace=self.config.restrict_to_workspace,
            project_profile=self._project_profile,
        )
        self.history.system_prompt = self.system_prompt
        if self.history.messages and self.history.messages[0].get("role") == "system":
            self.history.messages[0]["content"] = self.system_prompt
        else:
            self.history.reset()

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def run(self, task: str, extra_context: Optional[Dict[str, Any]] = None) -> RunResult:
        """执行一个任务直到终止条件触发。

        Args:
            task: 用户用自然语言描述的编程任务。
            extra_context: 附加上下文（如上一轮改动过的文件列表）。

        Returns:
            RunResult。除了 KeyboardInterrupt 之外的异常都会被捕获并写入 error_message，
            保证 CLI 永远不会因为一次工具报错而崩掉。
        """
        started = time.time()
        result = RunResult()
        # 每个任务一个独立子目录：首次进入时创建，同会话后续追问沿用
        result.task_dir = str(self.prepare_task_dir(task))
        # 每个任务清零"本次会话的副作用"与假完成拦截计数，保证语义是"本任务"而非跨任务累积
        self.ctx.session["changes"] = []
        self._ran_command_this_run = False
        self._finish_blocked = 0
        self._consecutive_run_failures = 0
        self.ctx.session.pop("finished", None)
        self.ctx.session.pop("summary", None)

        self.history.add_user(build_task_message(task, extra_context))
        self._emit("task_start", {"task": task})

        # V5 自主性：复杂任务开头注入"先用 plan 工具列计划"的提示，引导先规划再动手
        if self.config.plan_hint and looks_like_complex_task(task):
            self.history.add_note(
                "这是一个较复杂的任务。建议先调用 `plan` 工具列出分步计划（看清现状 → 改动 → 验证），"
                "再开始动手；计划能帮你和用户对齐目标，也避免东改西改。"
            )

        try:
            self._loop_body(result)
        except KeyboardInterrupt:
            result.finish_reason = "aborted"
            result.answer = result.answer or "（用户中断）"
            if self.console:
                self.console.warn("已中断当前任务（历史保留，可继续追问）")
        except Aborted as exc:
            result.finish_reason = "aborted"
            result.answer = str(exc)
        except LLMError as exc:
            result.finish_reason = "llm_error"
            result.error_message = str(exc)
            result.answer = f"模型调用失败：{exc}"
            if self.console:
                self.console.error(str(exc))
        except Exception as exc:  # noqa: BLE001 —— 兜底，保证 CLI 不崩
            result.finish_reason = "internal_error"
            result.error_message = f"{type(exc).__name__}: {exc}"
            result.answer = f"内部错误：{type(exc).__name__}: {exc}"
            if self.console:
                self.console.error(result.error_message)
        finally:
            result.elapsed = time.time() - started
            result.changes = list(self.ctx.session.get("changes", []))
            result.usage = dict(self._usage)
            result.compacted = self.history.compact_count
            self._save_session(result)
            # 本次任务改过文件 → 归档一份完整快照到 .agent_backups/
            try:
                snap = self.snapshot_to_backups()
                if snap is not None:
                    result.backup_dir = str(snap)
            except OSError as exc:          # 备份失败不该把任务结果带走
                if self.console:
                    self.console.warn(f"备份失败（不影响任务结果）：{exc}")

        return result

    # ------------------------------------------------------------------
    # 循环主体
    # ------------------------------------------------------------------
    def _loop_body(self, result: RunResult) -> None:
        max_steps = int(self.config.max_steps)

        for step in range(1, max_steps + 1):
            result.steps = step
            self._emit("step_start", {"step": step, "max_steps": max_steps})
            if self.console:
                self.console.step(step, max_steps, note=f"上下文≈{self.history.tokens} tokens")

            # --- 0) 上下文预算 ---
            self._maybe_compact()
            if step >= max_steps * 0.8:
                self.history.add_note(BUDGET_WARNING.format(used=step, max=max_steps))

            # --- 1) 调模型 ---
            msg = self._chat()
            if msg.usage:
                for k, v in msg.usage.items():
                    self._usage[k] = self._usage.get(k, 0) + int(v or 0)
            self.history.add_assistant(msg)
            self._emit("assistant", {"content": msg.content, "tool_calls": [c.name for c in msg.tool_calls]})

            # --- 2) 解析 ---
            outcome, attempts = self._parse_with_retry(msg)
            if self.console:
                self.console.thinking(outcome.narration)
            self._emit("parsed", {"calls": [c.name for c in outcome.calls], "issues": outcome.issues})

            # --- 3) 无调用 → 视为收尾 ---
            if not outcome.calls:
                text = (outcome.narration or msg.content or "").strip()
                if outcome.is_final or not outcome.issues:
                    result.finish_reason = "model_final"
                else:
                    result.finish_reason = "parse_failed"
                    if self.console:
                        self.console.warn("多次解析失败，已放弃本轮纠错")
                result.answer = self._strip_final_marker(text) or "（模型未给出总结）"
                return

            # --- 4) 执行工具 ---
            stop_reason = self._execute_calls(outcome.calls, result)
            if stop_reason:
                # finish 被调用 / 用户中止 / 错误过多
                self._finalize_after_tools(stop_reason, result)
                return

            # --- 5) 无进展检测 ---
            if self._check_stagnation():
                result.finish_reason = "no_progress"
                result.answer = "连续多步无有效进展，已停止。已完成的部分改动见上方记录。"
                if self.console:
                    self.console.warn(result.answer)
                return

        result.finish_reason = "max_steps"
        result.answer = f"已达到步数上限 {max_steps}，任务未确认完成。建议拆小任务或提高 agent.max_steps。"
        if self.console:
            self.console.warn(result.answer)

    # ------------------------------------------------------------------
    # 各步骤实现
    # ------------------------------------------------------------------
    def _chat(self) -> AssistantMessage:
        """调一次模型，带 UI 反馈。"""
        tools = self.registry.schemas() if self.native else None
        if self.console:
            with self.console.spinner(f"调用 {self.profile.model}"):
                return self.backend.chat(self.history.payload(), tools=tools)
        return self.backend.chat(self.history.payload(), tools=tools)

    def _parse_with_retry(self, msg: AssistantMessage):
        """解析；解析不出调用时把问题回灌给模型让它自我修正。

        Returns:
            (ParseOutcome, 纠错轮数)
        """
        outcome = self.parser.parse(msg)
        attempts = 0
        max_retries = int(self.config.max_parse_retries)

        while (not outcome.calls) and (not outcome.is_final) and attempts < max_retries:
            attempts += 1
            if outcome.issues:
                feedback = self.parser.build_correction_prompt(outcome, use_native=self.native)
            else:
                feedback = _EMPTY_OUTPUT_HINT
            if self.console:
                self.console.warn(f"输出无法执行（第 {attempts} 次纠错）：{outcome.issue_text() or '未给出工具调用'}")
            self.history.add_note(feedback)
            msg = self._chat()
            if msg.usage:
                for k, v in msg.usage.items():
                    self._usage[k] = self._usage.get(k, 0) + int(v or 0)
            self.history.add_assistant(msg)
            outcome = self.parser.parse(msg)

        return outcome, attempts

    def _execute_calls(self, calls: List[ToolCall], result: RunResult) -> Optional[str]:
        """执行一批工具调用，把回执写回历史。

        一轮里若全是只读工具调用（read_file / list_dir / grep_search / find_files / diff），
        且无 finish，则并行发出以缩短等待；其余（含写文件、run_command、finish）顺序执行，
        因为后者有副作用或会改变终止状态，必须保持顺序与即时拦截。

        Returns:
            None 表示继续循环；否则返回终止原因字符串。
        """
        # 并行分支：多个只读调用一起发（V5 自主性）
        if (self.config.parallel_tools and len(calls) > 1
                and all(c.name in _READONLY_TOOLS for c in calls)):
            return self._execute_parallel(calls, result)
        for call in calls:
            result.tool_calls += 1
            self._total_tool_calls += 1
            self._emit("tool_call", {"name": call.name, "args": call.arguments})
            if self.console:
                self.console.tool_call(call.name, call.arguments)

            if call.name == "run_command":
                self._ran_command_this_run = True

            # finish 是"控制类"工具：直接结束，不再把结果喂回模型
            if call.name == "finish":
                summary = (call.arguments.get("summary") or "").strip()
                # 「假完成」拦截：改过文件却从没运行过任何验证命令就收尾，
                # 大概率是没测过的半成品。回灌提示逼它先验证（最多拦 2 次，避免死循环）。
                wrote_files = any(
                    c.get("kind") in ("write", "edit")
                    for c in self.ctx.session.get("changes", [])
                )
                if wrote_files and not self._ran_command_this_run and self._finish_blocked < 2:
                    self._finish_blocked += 1
                    self.history.add_note(
                        "你修改/创建了文件，但本次会话还没运行过任何命令来验证改动确实可工作"
                        "（编译 / 测试 / 运行）。请先用 run_command 跑一遍验证，确认无误后再调用 finish 收尾；"
                        "若确实无法运行验证（例如纯文本/数据文件），也请先自行检查输出是否符合要求。"
                    )
                    return None  # 不收尾，继续循环
                self.ctx.session["finished"] = True
                self.ctx.session["summary"] = summary
                return "finish"

            tool_result = self.registry.execute(call.name, call.arguments, self.ctx, call_id=call.id)
            # 累计各工具耗时与调用次数（成本面板的"各工具耗时占比"用）
            self._tool_timings[call.name] = self._tool_timings.get(call.name, 0.0) + tool_result.elapsed
            self._tool_calls_by_name[call.name] = self._tool_calls_by_name.get(call.name, 0) + 1

            # 自修复闭环：run_command 失败时，把「出错位置 + 附近源码」直接回灌给模型，
            # 省掉它多一轮 read_file；连续失败达到预算时提醒停止乱试、考虑回滚。
            if call.name == "run_command":
                if tool_result.ok:
                    self._consecutive_run_failures = 0
                else:
                    self._on_command_failure(tool_result)

            # 回执写回历史：原生调用用 tool 角色，文本协议调用用 user 角色
            style = "native" if call.source == "native" else "text"
            rendered = tool_result.render(max_chars=int(self.config.max_tool_output_chars))
            # 超长回执被智能压缩，记一次（成本面板展示"回执智能压缩次数"）
            if len(tool_result.output or tool_result.error or "") > int(self.config.max_tool_output_chars):
                self._output_compressions += 1
            self.history.add_tool_result(call.id, call.name, rendered, style=style)

            self._emit("tool_result", {"name": call.name, "ok": tool_result.ok, "output": rendered[:400]})
            if self.console:
                self.console.tool_result(
                    call.name,
                    tool_result.ok,
                    rendered,
                    meta=f"{tool_result.elapsed:.2f}s",
                )

            # 统计与指纹
            if tool_result.ok:
                self._consecutive_errors = 0
            else:
                result.errors += 1
                self._consecutive_errors += 1
                if self._consecutive_errors >= int(self.config.max_consecutive_errors):
                    return "too_many_errors"

            self._fingerprints.append(call.fingerprint())

        return None

    def _execute_parallel(self, calls: List[ToolCall], result: RunResult) -> Optional[str]:
        """并行执行一批只读工具调用（见 _execute_calls 的并行分支说明）。

        只读工具不写 session、不触发 finish / 自修复，因此并行安全。回执按原顺序写回历史，
        统计与指纹在主线程里顺序累计，避免竞态。
        """
        futures = {}
        with _cf.ThreadPoolExecutor(max_workers=min(len(calls), 8)) as ex:
            for c in calls:
                fut = ex.submit(self.registry.execute, c.name, c.arguments, self.ctx, call_id=c.id)
                futures[fut] = c
            results = [None] * len(calls)
            for fut in _cf.as_completed(futures):
                c = futures[fut]
                results[calls.index(c)] = fut.result()

        for c, tool_result in zip(calls, results):
            result.tool_calls += 1
            self._total_tool_calls += 1
            self._tool_timings[c.name] = self._tool_timings.get(c.name, 0.0) + tool_result.elapsed
            self._tool_calls_by_name[c.name] = self._tool_calls_by_name.get(c.name, 0) + 1

            style = "native" if c.source == "native" else "text"
            rendered = tool_result.render(max_chars=int(self.config.max_tool_output_chars))
            if len(tool_result.output or tool_result.error or "") > int(self.config.max_tool_output_chars):
                self._output_compressions += 1
            self.history.add_tool_result(c.id, c.name, rendered, style=style)

            self._emit("tool_result", {"name": c.name, "ok": tool_result.ok, "output": rendered[:400]})
            if self.console:
                self.console.tool_result(c.name, tool_result.ok, rendered, meta=f"{tool_result.elapsed:.2f}s")

            if tool_result.ok:
                self._consecutive_errors = 0
            else:
                result.errors += 1
                self._consecutive_errors += 1
                if self._consecutive_errors >= int(self.config.max_consecutive_errors):
                    return "too_many_errors"
            self._fingerprints.append(c.fingerprint())

        return None

    def _finalize_after_tools(self, reason: str, result: RunResult) -> None:
        """工具阶段触发的结束处理。"""
        result.finish_reason = reason
        if reason == "finish":
            result.answer = self.ctx.session.get("summary") or "（模型未给出总结）"
        elif reason == "too_many_errors":
            result.answer = (
                f"连续 {self._consecutive_errors} 次工具执行失败，已停止以避免无效循环。\n"
                "最近的错误信息已在上方列出，请据此修正任务描述或环境后重试。"
            )
            if self.console:
                self.console.error(result.answer)
        elif reason == "aborted":
            result.answer = "（操作被取消）"

    def _on_command_failure(self, tool_result) -> None:
        """run_command 失败后的自修复感知：回灌「出错位置 + 源码上下文」，并在预算耗尽时提醒回滚。"""
        from .selfrepair import build_failure_note, detect_test_command

        output = tool_result.output or tool_result.error or ""
        note = build_failure_note(output, self.ctx)

        # 没有 traceback，但可能是「没测到」——给出该项目的测试命令提示
        if note is None and ("no tests ran" in output or "collected 0 items" in output):
            cmd = detect_test_command(self.workspace_root)
            if cmd:
                note = f"没有收集到任何测试用例。该项目可用的测试命令是 `{cmd}`，请确认测试文件命名/位置后重试。"

        if note:
            self.history.add_note(note)

        self._consecutive_run_failures += 1
        budget = int(getattr(self.config, "max_repair_retries", 5))
        if self._consecutive_run_failures >= budget:
            self.history.add_note(
                f"已连续 {self._consecutive_run_failures} 次运行失败。请停止无方向地乱试："
                f"回顾最近一次错误的根因、定向修复；若仍无法修复，可调用 rollback 工具"
                f"把代码恢复到上一次能跑通的快照，再从那时起重新分析。"
            )

    # ------------------------------------------------------------------
    # 上下文与停滞检测
    # ------------------------------------------------------------------
    def _maybe_compact(self) -> None:
        """超预算时压缩历史。"""
        budget = int(self.config.max_context_tokens) - int(self.config.reserve_tokens)
        if not self.config.auto_compact:
            return
        if not self.history.needs_compaction(budget, float(self.config.compact_threshold)):
            return
        if self.console:
            self.console.info(f"上下文 {self.history.tokens}/{budget} tokens，触发自动压缩…")
        ok = self.history.compact(
            llm=self.backend,
            keep_recent=int(self.config.compact_keep_recent),
            budget=budget,
        )
        if ok:
            self.history.add_note("（上下文已压缩，请基于摘要继续工作，不要重复已完成的操作。）")
            if self.console:
                self.console.info(f"压缩完成，当前 {self.history.tokens} tokens")

    def _check_stagnation(self) -> bool:
        """检测"原地打转"：最近 3 次调用完全相同，或连续多步无新变更。"""
        if len(self._fingerprints) >= 3:
            last3 = list(self._fingerprints)[-3:]
            if len(set(last3)) == 1:
                self.history.add_note(NO_PROGRESS_HINT.format(n=3))
                self._stale_steps += 1
                self._fingerprints.clear()
        if self._stale_steps >= 2:
            return True
        return False

    # ------------------------------------------------------------------
    # 成本面板（V3）
    # ------------------------------------------------------------------
    def build_stats_panel(self) -> str:
        """生成会话成本面板，供 CLI 的 `/stats` 命令与答辩演示使用。

        包含：真实 token 消耗（API usage）、当前上下文估算、会话耗时、
        工具调用总数、上下文/回执压缩次数、各工具耗时占比。

        设计要点：token 以「API 实际返回的 usage」为准（最精确），
        上下文估算只用于"发送前"的预算判断，二者来源不同、用途不同。
        """
        from .history import token_counter_name

        u = self._usage
        lines = ["【会话成本面板 /stats】",
                 "  真实 token 消耗（来自 API usage，最精确）：",
                 f"    prompt     = {u.get('prompt_tokens', 0):,}",
                 f"    completion = {u.get('completion_tokens', 0):,}",
                 f"    total      = {u.get('total_tokens', 0):,}",
                 f"  当前上下文估算：{self.history.tokens:,} tokens（计数方式：{token_counter_name()}）",
                 f"  会话耗时：{time.time() - self._session_started:.1f}s",
                 f"  工具调用总数：{self._total_tool_calls}",
                 f"  上下文压缩次数：{self.history.compact_count}    回执智能压缩次数：{self._output_compressions}"]
        if self._tool_timings:
            total_t = sum(self._tool_timings.values())
            lines.append("  各工具耗时占比：")
            for name, t in sorted(self._tool_timings.items(), key=lambda kv: kv[1], reverse=True):
                pct = (t / total_t * 100) if total_t > 0 else 0.0
                n = self._tool_calls_by_name.get(name, 0)
                lines.append(f"    {name:<14} {t:7.2f}s  {pct:5.1f}%  ({n} 次)")
        else:
            lines.append("  各工具耗时占比：本次会话尚未调用任何工具")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_final_marker(text: str) -> str:
        import re
        return re.sub(r"^\s*(FINAL|最终答案|完成)\s*[:：]\s*", "", text or "").strip()

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        if self.on_event:
            try:
                self.on_event(event, payload)
            except Exception:  # noqa: BLE001 —— 回调不能影响主流程
                pass

    def _save_session(self, result: RunResult) -> None:
        """会话落盘（JSONL），便于复盘与答辩演示。"""
        log_dir = getattr(self.config, "session_log", None)
        if not log_dir:
            return
        try:
            d = Path(self.config.workspace).expanduser().resolve() / log_dir
            d.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = d / f"session-{stamp}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                f.write(self.history.to_jsonl())
                f.write("\n" + json.dumps({"__result__": result.stats_line()}, ensure_ascii=False) + "\n")
            self._emit("session_saved", {"path": str(path)})
        except OSError:
            pass
