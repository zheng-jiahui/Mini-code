"""
错误类型分层。

分层的原因：不同层的错误，处理策略完全不同——
    ConfigError    → 直接终止，属于使用者配置问题，模型救不回来
    LLMError       → 可重试（网络/限流/5xx）；重试耗尽才终止
    ParseError     → 把错误原文回灌给模型，让它在下一轮自我修正
    ToolError      → 绝不抛出到主循环外，必须转成"工具回执"交给模型看
    SecurityError  → ToolError 的子类，额外记审计日志
    BudgetExceeded → 触发上下文压缩或优雅终止
    Aborted        → 用户中断，正常退出
"""

from __future__ import annotations

from typing import Optional


class AgentError(Exception):
    """所有本项目异常的基类。"""

    default_message = "智能体内部错误"

    def __init__(self, message: Optional[str] = None, **detail):
        self.message = message or self.default_message
        self.detail = detail
        super().__init__(self.message)

    def __str__(self) -> str:  # pragma: no cover - 简单可读化
        if self.detail:
            extra = ", ".join(f"{k}={v}" for k, v in self.detail.items())
            return f"{self.message} ({extra})"
        return self.message


class ConfigError(AgentError):
    """配置缺失或非法（如 API key 为空、workspace 不存在）。"""

    default_message = "配置错误"


class LLMError(AgentError):
    """模型调用失败。retryable=False 表示重试也没用（如 401、400）。"""

    def __init__(self, message=None, *, retryable: bool = True, status: Optional[int] = None, **detail):
        super().__init__(message, **detail)
        self.retryable = retryable
        self.status = status


class ParseError(AgentError):
    """模型输出无法解析为合法工具调用。

    feedback 字段会被回灌给模型，引导其下一轮输出正确格式。
    """

    def __init__(self, message=None, *, raw: str = "", feedback: str = "", **detail):
        super().__init__(message or "无法解析模型输出", **detail)
        self.raw = raw
        self.feedback = feedback or f"输出解析失败：{self.message}"


class ToolError(AgentError):
    """工具执行失败。主循环会把它渲染成一条 tool 消息回灌给模型。"""

    def __init__(self, message=None, *, tool: str = "", hint: str = "", **detail):
        super().__init__(message, **detail)
        self.tool = tool
        self.hint = hint

    def render(self) -> str:
        """渲染成给模型看的错误回执。"""
        parts = [f"[{self.tool or 'tool'} 执行失败] {self.message}"]
        if self.hint:
            parts.append(f"提示：{self.hint}")
        return "\n".join(parts)


class SecurityError(ToolError):
    """被安全策略拦截（越界路径 / 危险命令）。"""

    default_message = "操作被安全策略拦截"


class BudgetExceeded(AgentError):
    """超出上下文或步数预算。"""

    def __init__(self, message=None, *, kind: str = "steps", **detail):
        super().__init__(message or "超出预算", **detail)
        self.kind = kind  # steps | tokens


class Aborted(AgentError):
    """用户主动中断（Ctrl-C / 输入 /exit / 危险命令选择 n）。"""

    default_message = "任务被用户中止"
