"""
MiniCode —— 从零实现的编程智能体。

设计原则（也是本项目区别于"调框架"的地方）：
    1. 不引入任何 Agent 框架 / SDK，只使用 OpenAI 兼容的聊天补全客户端；
    2. 对话历史、工具定义与本地执行、输出解析、循环终止、错误处理、上下文压缩全部自行实现；
    3. 一切文件/命令操作都在本地进程内完成，不依赖 API 服务端托管能力。
"""

__version__ = "0.1.0"
__author__ = "MiniCode"

from .errors import (  # noqa: F401
    Aborted,
    AgentError,
    BudgetExceeded,
    ConfigError,
    LLMError,
    ParseError,
    SecurityError,
    ToolError,
)

__all__ = [
    "__version__",
    "AgentError",
    "ConfigError",
    "LLMError",
    "ParseError",
    "ToolError",
    "SecurityError",
    "BudgetExceeded",
    "Aborted",
]
