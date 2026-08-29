"""
自建工具系统：工具的定义（JSON Schema）、注册、执行与结果封装。

这是"不使用框架"时最核心的一块——框架通常提供 @tool 装饰器与执行器，这里全部手写：
    · ToolSpec     —— 一个工具的元数据（名字、描述、参数 JSON Schema、处理函数）
    · ToolRegistry —— 名字 → ToolSpec 的映射 + 统一的执行入口（带异常兜底）
    · ToolResult   —— 统一的回执结构，负责截断、脱敏、转 history 消息

关键约定：工具函数**永远不向主循环抛异常**。
任何异常都在 registry.execute 里被捕获并转成 ok=False 的 ToolResult，
因为"工具报错"本身就是给模型的重要信号（比崩溃有价值得多）。
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..errors import SecurityError, ToolError
from ..security import PathGuard, redact_secrets, truncate_output

__all__ = ["ToolContext", "ToolResult", "ToolSpec", "ToolRegistry", "tool_spec"]

ToolHandler = Callable[[Dict[str, Any], "ToolContext"], "ToolResult"]


# ----------------------------------------------------------------------------
# 执行上下文
# ----------------------------------------------------------------------------
@dataclass
class ToolContext:
    """注入给每个工具的运行时上下文（替代全局变量，便于测试）。"""

    workspace: Path
    guard: PathGuard
    config: Any                       # AgentConfig
    console: Any = None               # Console（可为 None，便于单测）
    session: Dict[str, Any] = field(default_factory=dict)

    # --- 便捷方法 ---
    def resolve(self, path: str, **kwargs) -> Path:
        return self.guard.resolve(path, **kwargs)

    def confirm(self, question: str, default: bool = False) -> bool:
        if self.console is None:
            return default
        return self.console.confirm(question, default=default)

    def record_change(self, kind: str, detail: str) -> None:
        """记录本次会话产生的副作用（用于结尾汇总"改了哪些文件"）。"""
        self.session.setdefault("changes", []).append({"kind": kind, "detail": detail, "ts": time.time()})


# ----------------------------------------------------------------------------
# 回执
# ----------------------------------------------------------------------------
@dataclass
class ToolResult:
    """工具执行结果。"""

    ok: bool
    output: str = ""
    error: str = ""
    data: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)
    elapsed: float = 0.0

    # ---------- 构造辅助 ----------
    @classmethod
    def success(cls, output: str = "", **kwargs) -> "ToolResult":
        meta = dict(kwargs.pop("meta", {}) or {})
        return cls(ok=True, output=output, meta=meta, **kwargs)

    @classmethod
    def failure(cls, error: str, hint: str = "", **kwargs) -> "ToolResult":
        meta = {"hint": hint}
        meta.update(kwargs.pop("meta", {}) or {})   # 允许调用方补充 meta，而不是覆盖报错提示
        return cls(ok=False, output="", error=error, meta=meta, **kwargs)

    @classmethod
    def from_exception(cls, exc: BaseException, tool: str = "") -> "ToolResult":
        if isinstance(exc, ToolError):
            return cls(ok=False, error=exc.render(), meta={"tool": exc.tool or tool})
        if isinstance(exc, KeyboardInterrupt):
            raise exc
        tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return cls(ok=False, error=f"[{tool}] {tb}", meta={"tool": tool})

    # ---------- 输出 ----------
    def render(self, max_chars: int = 0) -> str:
        """渲染成给模型看的字符串（含错误标记、脱敏、截断）。"""
        if not self.ok:
            body = self.error or "工具执行失败（无错误信息）"
            hint = self.meta.get("hint")
            if hint:
                body += f"\n提示：{hint}"
            return f"[TOOL ERROR] {body}"

        body = self.output or "(无输出)"
        body = redact_secrets(body)
        if max_chars and len(body) > max_chars:
            body = truncate_output(body, max_chars, note="工具回执过长")
        return body

    def to_message(self, call_id: str, name: str, style: str = "native") -> Dict[str, Any]:
        """转成 OpenAI 风格消息。

        style="native" → {"role": "tool", "tool_call_id": ...}
        style="text"   → {"role": "user", "content": "<tool_result ...>...</tool_result>"}
        """
        content = self.render(max_chars=self.meta.get("max_chars", 0))
        if style == "native":
            return {"role": "tool", "tool_call_id": call_id, "content": content}
        status = "ok" if self.ok else "error"
        return {
            "role": "user",
            "content": f'<tool_result name="{name}" status="{status}">\n{content}\n</tool_result>',
        }

    def __str__(self) -> str:  # pragma: no cover
        return self.render()


# ----------------------------------------------------------------------------
# 工具定义
# ----------------------------------------------------------------------------
@dataclass
class ToolSpec:
    """一个工具的完整定义。parameters 是标准 JSON Schema（object 类型）。"""

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: ToolHandler
    dangerous: bool = False      # True 时执行前需按 command_policy 处理
    hidden: bool = False         # True 时不出现在系统提示词的工具清单里
    category: str = "general"

    def openai_schema(self) -> Dict[str, Any]:
        """转成 OpenAI function calling 的 schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }

    def signature(self) -> str:
        """一行签名，用于系统提示词。"""
        props: Dict[str, Any] = (self.parameters or {}).get("properties", {})
        required = set((self.parameters or {}).get("required", []) or [])
        parts = []
        for key, spec in props.items():
            t = spec.get("type", "any")
            mark = "" if key in required else "?"
            parts.append(f"{key}{mark}: {t}")
        return f"{self.name}({', '.join(parts)})"

    def doc(self) -> str:
        return f"- {self.signature()} —— {self.description}"


def tool_spec(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    *,
    dangerous: bool = False,
    hidden: bool = False,
    category: str = "general",
) -> Callable[[ToolHandler], ToolSpec]:
    """装饰器：把普通函数标记为 ToolSpec（函数本身不注册，注册由 registry 完成）。"""

    def deco(fn: ToolHandler) -> ToolSpec:
        spec = ToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            handler=fn,
            dangerous=dangerous,
            hidden=hidden,
            category=category,
        )
        spec.__doc__ = fn.__doc__ or description
        return spec

    return deco


# ----------------------------------------------------------------------------
# 注册表
# ----------------------------------------------------------------------------
class ToolRegistry:
    """工具注册表 + 统一执行器。"""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    # ---------- 注册 ----------
    def register(self, spec: ToolSpec, *, override: bool = False) -> ToolSpec:
        if not spec.name:
            raise ValueError("工具名不能为空")
        if spec.name in self._tools and not override:
            raise ValueError(f"工具已注册：{spec.name}")
        self._tools[spec.name] = spec
        return spec

    def register_many(self, specs) -> None:
        for s in specs:
            self.register(s)

    # ---------- 查询 ----------
    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def specs(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def visible_specs(self) -> List[ToolSpec]:
        return [s for s in self._tools.values() if not s.hidden]

    def schemas(self) -> List[Dict[str, Any]]:
        """给 OpenAI 接口用的工具 schema 列表。"""
        return [s.openai_schema() for s in self._tools.values() if not s.hidden]

    def describe(self) -> str:
        """生成给系统提示词用的工具清单文本。"""
        buckets: Dict[str, List[ToolSpec]] = {}
        for s in self.visible_specs():
            buckets.setdefault(s.category, []).append(s)
        lines = []
        for cat, items in buckets.items():
            lines.append(f"【{cat}】")
            lines.extend(s.doc() for s in items)
        return "\n".join(lines)

    # ---------- 执行 ----------
    def execute(self, name: str, args: Dict[str, Any], ctx: ToolContext, *, call_id: str = "") -> ToolResult:
        """执行工具，**保证不抛异常**（KeyboardInterrupt 除外）。

        Args:
            name: 工具名。
            args: 已校验的参数字典。
            ctx: 执行上下文。
            call_id: 调用 ID，仅用于日志。

        Returns:
            ToolResult；未知工具 / 参数异常 / 越权都会变成 ok=False 的结果。
        """
        started = time.perf_counter()
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult.failure(
                f"未知工具 `{name}`。可用工具：{', '.join(self.names())}",
                hint="请只使用系统提示词中列出的工具名。",
                meta={"tool": name},
            )

        try:
            result = spec.handler(args or {}, ctx)
            if not isinstance(result, ToolResult):  # 允许 handler 直接返回字符串
                result = ToolResult.success(str(result))
        except SecurityError as exc:
            result = ToolResult.failure(exc.render(), hint=exc.hint or "", meta={"tool": name, "blocked": True})
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 —— 有意宽泛捕获，错误要回灌给模型
            result = ToolResult.from_exception(exc, tool=name)

        result.elapsed = time.perf_counter() - started
        result.meta.setdefault("tool", name)
        result.meta.setdefault("max_chars", getattr(ctx.config, "max_tool_output_chars", 0))
        return result

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def _json_type_hint(value: Any) -> str:  # pragma: no cover - 调试辅助
    return type(value).__name__
