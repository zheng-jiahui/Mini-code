"""
安全层：路径沙箱、危险命令识别、输出净化。

这是"自己实现工具执行"时必须自己补上的一环——框架通常帮你做了，这里没有框架。

三条防线：
    1. 路径沙箱：所有文件读写先解析为绝对路径，必须落在 workspace 内（可关闭但默认开启）；
    2. 命令审查：按 command_policy（allow/confirm/deny）处理命中的危险命令；
    3. 输出净化：截断超长输出（防止一次命令吃掉整个上下文），并对常见密钥形态打码。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .errors import SecurityError

__all__ = ["PathGuard", "CommandGuard", "truncate_output", "redact_secrets", "check_command"]

# 默认的敏感目录/文件（连读都不建议）
_SENSITIVE_NAMES = {".env", ".env.local", "id_rsa", "id_ed25519", ".npmrc", ".pypirc", "credentials", "secrets.yaml", "config.yaml"}
_SENSITIVE_SUFFIX = (".pem", ".key", ".p12", ".pfx", ".keystore")

# 输出里疑似密钥的形态（打码，避免模型把密钥写进文件/回显到终端）
_SECRET_PATTERNS = [
    (re.compile(r"(sk-[A-Za-z0-9_\-]{16,})"), "sk-***"),
    (re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)(['\"]?)([A-Za-z0-9_\-]{12,})\2"), r"\1\2***\2"),
    (re.compile(r"(?i)(bearer\s+)([A-Za-z0-9._\-]{12,})"), r"\1***"),
    (re.compile(r"(?i)(token\s*[=:]\s*)(['\"]?)([A-Za-z0-9._\-]{12,})\2"), r"\1\2***\2"),
]


# ----------------------------------------------------------------------------
# 路径沙箱
# ----------------------------------------------------------------------------
class PathGuard:
    """把所有用户/模型给出的路径约束到 workspace 之内。"""

    def __init__(self, workspace: Path, enabled: bool = True, extra_roots: Optional[Iterable[Path]] = None):
        self.workspace = Path(workspace).resolve()
        self.enabled = enabled
        self.extra_roots = [Path(p).resolve() for p in (extra_roots or [])]

    def resolve(self, user_path: str, *, must_exist: bool = False, allow_outside: bool = False) -> Path:
        """解析并校验路径。

        Args:
            user_path: 相对或绝对路径字符串。
            must_exist: True 时路径不存在直接抛错。
            allow_outside: 临时放宽沙箱（仅内部工具使用）。

        Returns:
            安全的绝对路径。

        Raises:
            SecurityError: 越界、非法路径或不存在。
        """
        if not user_path or not str(user_path).strip():
            raise SecurityError("路径为空", tool="path", hint="请给出相对于工作区的路径，如 `src/main.py`。")

        raw = os.path.expandvars(str(user_path).strip().strip('"').strip("'"))
        p = Path(os.path.expanduser(raw))
        if not p.is_absolute():
            p = self.workspace / p

        try:
            resolved = p.resolve()
        except (OSError, RuntimeError) as exc:
            raise SecurityError(f"路径无法解析：{user_path}（{exc}）", tool="path") from exc

        if self.enabled and not allow_outside:
            if not self._inside(resolved):
                raise SecurityError(
                    f"路径越界：{user_path} -> {resolved}",
                    tool="path",
                    hint=f"所有文件操作必须位于工作区内：{self.workspace}。若要访问外部路径，请把 agent.workspace 设为更大范围。",
                )

        if must_exist and not resolved.exists():
            raise SecurityError(
                f"路径不存在：{user_path}",
                tool="path",
                hint="先用 list_dir 确认目录结构，或检查大小写与斜杠方向。",
            )
        return resolved

    def _inside(self, path: Path) -> bool:
        for root in [self.workspace, *self.extra_roots]:
            try:
                path.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def is_sensitive(self, path: Path) -> bool:
        """是否为疑似存放凭据的文件。"""
        name = path.name.lower()
        return name in _SENSITIVE_NAMES or name.endswith(_SENSITIVE_SUFFIX)

    def relpath(self, path: Path) -> str:
        """转成相对工作区的展示路径。"""
        try:
            return str(Path(path).resolve().relative_to(self.workspace))
        except ValueError:
            return str(path)


# ----------------------------------------------------------------------------
# 命令审查
# ----------------------------------------------------------------------------
class CommandGuard:
    """危险命令识别。"""

    def __init__(self, patterns: Optional[Iterable[str]] = None, policy: str = "confirm"):
        # 配置未给黑名单时回落到内置默认规则，避免出现"空规则 = 全部放行"的危险默认态
        if not patterns:
            from .config import DEFAULT_DANGEROUS_COMMANDS
            patterns = DEFAULT_DANGEROUS_COMMANDS
        self.patterns: List[re.Pattern] = []
        for pat in (patterns or []):
            try:
                self.patterns.append(re.compile(pat))
            except re.error:
                continue  # 非法正则直接忽略，不因配置错误阻塞启动
        self.policy = policy

    def check(self, command: str) -> Tuple[str, str]:
        """审查命令。

        Returns:
            (decision, reason)；decision ∈ {"ok", "confirm", "deny"}
        """
        if self.policy == "allow":
            return "ok", ""
        for pat in self.patterns:
            m = pat.search(command or "")
            if m:
                if self.policy == "deny":
                    return "deny", f"命中危险命令规则 /{pat.pattern}/ （片段：{m.group(0)!r}）"
                return "confirm", f"命中危险命令规则 /{pat.pattern}/ （片段：{m.group(0)!r}）"
        return "ok", ""


def check_command(guard: CommandGuard, command: str) -> Tuple[str, str]:  # 便捷函数
    return guard.check(command)


# ----------------------------------------------------------------------------
# 输出净化
# ----------------------------------------------------------------------------
def truncate_output(text: str, max_chars: int, *, note: str = "输出过长已截断") -> str:
    """head + tail 双端截断：保留开头（报错堆栈头部）与结尾（最新结果），丢中间。

    比"只保留开头"更好的原因：命令输出的关键结论通常在最后（如 pytest 的 summary）。
    """
    if max_chars <= 0 or text is None:
        return ""
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    omitted = len(text) - max_chars
    return (
        f"{text[:head]}\n"
        f"\n... [省略 {omitted} 字符：{note}] ...\n\n"
        f"{text[-tail:]}"
    )


def redact_secrets(text: str) -> str:
    """对输出中的疑似密钥打码。"""
    if not text:
        return text
    for pat, repl in _SECRET_PATTERNS:
        text = pat.sub(repl, text)
    return text
