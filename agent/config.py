"""
配置管理。

设计要点
--------
1. **凭据与代码分离**：`config.yaml` 存放 base_url / model / api_key，
   其中 api_key 推荐写成环境变量占位符 `${OPENAI_API_KEY}`；该文件已在 .gitignore 中。
2. **三层优先级**：命令行参数 > 环境变量 > YAML 文件 > 内置默认值。
3. **多档位（profile）**：可同时配置 deepseek / 本地 Ollama 等多个端点，运行时切换。
4. **失效即早**：加载时做完整校验，缺 key、路径不存在等问题在启动阶段就报错。

用法示例：
    cfg = load_config()
    cfg = load_config(cli_overrides={"model": "deepseek-chat", "temperature": 0.0})
    cfg = load_config(profile="deepseek")
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .errors import ConfigError

try:  # PyYAML 是可选依赖：没有它也能用 JSON 配置
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

__all__ = ["LLMProfile", "AgentConfig", "Config", "load_config", "DEFAULT_DANGEROUS_COMMANDS"]

# 支持 "${VAR}" / "$VAR" / "env:VAR" 三种占位写法
_ENV_PATTERN = re.compile(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$")
_ENV_PREFIX = "env:"

# config.yaml 的搜索顺序（第一个命中的生效）
_CONFIG_CANDIDATES: Iterable[str] = (
    "config.yaml",
    "config.yml",
    "config.json",
)


# ----------------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------------
@dataclass
class LLMProfile:
    """一个模型档位（端点 + 模型 + 采样参数）。"""

    name: str = "default"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 4096
    timeout: float = 120.0
    max_retries: int = 3          # 可重试错误（限流/网络/5xx）的重试次数
    retry_backoff: float = 2.0    # 指数退避基数（秒）
    native_tools: bool = True     # 是否支持 OpenAI 风格的 function calling
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_body: Dict[str, Any] = field(default_factory=dict)  # 透传给端点的私有字段

    def masked(self) -> Dict[str, Any]:
        """返回可安全打印（不含密钥）的视图。"""
        data = asdict(self)
        key = data.get("api_key") or ""
        data["api_key"] = (key[:6] + "***" + key[-4:]) if len(key) > 12 else ("***" if key else "<未设置>")
        return data


@dataclass
class AgentConfig:
    """Agent 运行期参数（循环控制、安全、上下文预算）。"""

    workspace: str = "."
    max_steps: int = 60
    max_steps_without_progress: int = 8
    max_consecutive_errors: int = 4
    max_parse_retries: int = 2

    command_timeout: int = 120
    max_tool_output_chars: int = 12_000
    max_file_read_chars: int = 40_000

    max_context_tokens: int = 96_000
    reserve_tokens: int = 4_000
    compact_keep_recent: int = 8
    auto_compact: bool = True
    compact_threshold: float = 0.75

    restrict_to_workspace: bool = True
    command_policy: str = "confirm"      # allow | confirm | deny
    backup_on_write: bool = True
    backup_dir: str = ".agent_backups"
    stream: bool = False
    session_log: Optional[str] = ".agent_sessions"

    dangerous_commands: List[str] = field(default_factory=list)

    def resolved_workspace(self) -> Path:
        """返回绝对化的工作区根目录。"""
        return Path(self.workspace).expanduser().resolve()


DEFAULT_DANGEROUS_COMMANDS: List[str] = [
    r"rm\s+-rf\s+(\/|\~|\*|$)",
    r"rm\s+-rf\s+\.\.?\s*$",
    r"(?i)\bformat\s+[a-z]:",
    r"(?i)\bdel\s+\/[fs]\b",
    r"(?i)\bshutdown\b|\breboot\b",
    r"(?i)git\s+push\s+(-{1,2}force|-f\b)",
    r"(?i)git\s+reset\s+--hard",
    r"(?i)git\s+clean\s+-[fdx]+",
    r"(?i)chmod\s+(-R\s+)?777",
    r"(?i)curl\s+[^\n|]*\|\s*(ba|z|fi)?sh",
    r"(?i)\bmkfs\b|\bdd\s+if=",
    r"(?i)taskkill\s+\/f\s+\/im\s+(system|csrss|winlogon)",
]


@dataclass
class Config:
    """顶层配置对象。"""

    llm: LLMProfile
    agent: AgentConfig
    profiles: Dict[str, LLMProfile] = field(default_factory=dict)
    source: Optional[str] = None

    def describe(self) -> str:
        """一行摘要，用于启动横幅。"""
        return (
            f"profile={self.llm.name} model={self.llm.model} "
            f"base_url={self.llm.base_url} temp={self.llm.temperature} "
            f"tools={'原生' if self.llm.native_tools else '文本协议'} "
            f"workspace={self.agent.resolved_workspace()}"
        )


# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
def _expand_env(value: Any) -> Any:
    """递归展开字符串中的环境变量占位符。"""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(_ENV_PREFIX):
            return os.environ.get(text[len(_ENV_PREFIX):].strip(), "")
        m = _ENV_PATTERN.match(text)
        if m:
            return os.environ.get(m.group(1), "")
        return value
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """原地递归合并，返回 base。"""
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _read_raw(path: Path) -> Dict[str, Any]:
    """读取 YAML/JSON 配置文件。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"无法读取配置文件 {path}: {exc}", path=str(path)) from exc

    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path} 不是合法 JSON: {exc}", path=str(path)) from exc
    else:
        if yaml is None:
            raise ConfigError(
                f"未安装 PyYAML，无法解析 {path}；请 pip install pyyaml，或改用 config.json",
                path=str(path),
            )
        try:
            data = yaml.safe_load(text)
        except Exception as exc:  # yaml.YAMLError 类型不稳定，宽泛捕获
            raise ConfigError(f"{path} 不是合法 YAML: {exc}", path=str(path)) from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} 顶层必须是对象/映射", path=str(path))
    return _expand_env(data)


def _locate_config(explicit: Optional[str]) -> Optional[Path]:
    """定位配置文件：显式路径 > 环境变量 AGENT_CONFIG > 当前目录候选 > 用户目录。"""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise ConfigError(f"指定的配置文件不存在：{p}", path=str(p))
        return p.resolve()

    env_path = os.environ.get("AGENT_CONFIG")
    if env_path and Path(env_path).expanduser().exists():
        return Path(env_path).expanduser().resolve()

    here = Path(__file__).resolve().parent.parent
    for name in _CONFIG_CANDIDATES:
        for base in (Path.cwd(), here):
            cand = base / name
            if cand.exists():
                return cand.resolve()

    user_cfg = Path.home() / ".minicode" / "config.yaml"
    if user_cfg.exists():
        return user_cfg
    return None


# ----------------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------------
def load_config(
    explicit: Optional[str] = None,
    profile: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    require_api_key: bool = True,
) -> Config:
    """加载配置。

    Args:
        explicit: 显式指定的配置文件路径。
        profile: 选择哪个模型档位，None 时读配置里的 `active_profile`。
        cli_overrides: 命令行覆盖项，支持 {"model":..., "temperature":...,
                       "base_url":..., "api_key":..., "workspace":..., "max_steps":...}。
        require_api_key: 是否强制要求 API key（离线演示/单测时可设为 False）。

    Returns:
        校验通过的 Config 对象。

    Raises:
        ConfigError: 配置非法（缺 API key、workspace 不存在、档位不存在等）。
    """
    path = _locate_config(explicit)
    raw: Dict[str, Any] = _read_raw(path) if path else {}

    # ---- 1) 解析所有档位 ----
    profiles_raw: Dict[str, Any] = dict(raw.get("profiles") or {})
    profiles: Dict[str, LLMProfile] = {}
    for name, body in profiles_raw.items():
        body = dict(body or {})
        body.setdefault("name", name)
        profiles[name] = LLMProfile(**{k: v for k, v in body.items() if k in LLMProfile.__annotations__})

    # 兼容"扁平写法"：顶层直接写 model/base_url/api_key
    flat = {k: v for k, v in raw.items() if k in LLMProfile.__annotations__ and k != "name"}
    if flat:
        merged = dict(profiles.get("default", LLMProfile()).__dict__)
        merged.update(flat)
        profiles["default"] = LLMProfile(**merged)

    if not profiles:
        profiles["default"] = LLMProfile()

    # ---- 2) 选择档位 ----
    active = profile or raw.get("active_profile") or os.environ.get("AGENT_PROFILE") or "default"
    if active not in profiles:
        raise ConfigError(
            f"模型档位 '{active}' 不存在，可选：{', '.join(sorted(profiles))}",
            profile=active,
        )
    llm = profiles[active]
    llm.name = active

    # ---- 3) 环境变量覆盖（优先级高于 YAML）----
    env_map = {
        "api_key": "OPENAI_API_KEY",
        "base_url": "OPENAI_BASE_URL",
        "model": "OPENAI_MODEL",
    }
    for field_name, env_name in env_map.items():
        val = os.environ.get(env_name)
        if val:
            setattr(llm, field_name, val)
    # AGENT_* 系列优先级更高，便于临时切换
    agent_env_map = {
        "api_key": "AGENT_API_KEY",
        "base_url": "AGENT_BASE_URL",
        "model": "AGENT_MODEL",
        "temperature": "AGENT_TEMPERATURE",
        "native_tools": "AGENT_NATIVE_TOOLS",
    }
    for field_name, env_name in agent_env_map.items():
        val = os.environ.get(env_name)
        if val:
            if field_name == "temperature":
                try:
                    llm.temperature = float(val)
                except ValueError as exc:
                    raise ConfigError(f"AGENT_TEMPERATURE 不是数字：{val}") from exc
            elif field_name == "native_tools":
                llm.native_tools = val.strip().lower() in ("1", "true", "yes", "on")
            else:
                setattr(llm, field_name, val)

    # ---- 4) Agent 运行参数 ----
    agent_raw = _deep_update({}, dict(raw.get("agent") or {}))
    agent = AgentConfig(**{k: v for k, v in agent_raw.items() if k in AgentConfig.__annotations__})

    ws_env = os.environ.get("AGENT_WORKSPACE")
    if ws_env:
        agent.workspace = ws_env

    # ---- 5) 命令行覆盖（最高优先级）----
    for key, value in (cli_overrides or {}).items():
        if value is None:
            continue
        if key in LLMProfile.__annotations__:
            setattr(llm, key, value)
        elif key in AgentConfig.__annotations__:
            setattr(agent, key, value)
        # 未知 key 静默忽略，避免 CLI 传多余字段时崩溃

    # ---- 6) 默认值补齐与校验 ----
    if not agent.dangerous_commands:
        agent.dangerous_commands = list(DEFAULT_DANGEROUS_COMMANDS)

    if agent.command_policy not in ("allow", "confirm", "deny"):
        raise ConfigError(
            f"command_policy 必须是 allow/confirm/deny 之一，当前为 {agent.command_policy!r}",
            policy=agent.command_policy,
        )

    ws = agent.resolved_workspace()
    if not ws.exists():
        try:
            ws.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(f"工作区不存在且无法创建：{ws}（{exc}）", workspace=str(ws)) from exc
    if not ws.is_dir():
        raise ConfigError(f"工作区不是目录：{ws}", workspace=str(ws))

    if require_api_key and not llm.api_key:
        raise ConfigError(
            "未配置 API key。请在 config.yaml 中设置 api_key，"
            "或设置环境变量 OPENAI_API_KEY / AGENT_API_KEY。",
            profile=llm.name,
        )
    if not llm.base_url:
        raise ConfigError("base_url 不能为空", profile=llm.name)

    return Config(llm=llm, agent=agent, profiles=profiles, source=str(path) if path else None)
