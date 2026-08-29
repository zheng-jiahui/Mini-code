"""
LLM 接入层。

只做一件事：把「消息列表 + 工具 schema」发给 OpenAI 兼容端点，拿回标准化的 AssistantMessage。
刻意保持"薄"——不含任何 Agent 逻辑（不决定调什么工具、不循环），那是 loop.py 的职责。

三种后端：
    OpenAIBackend  : 使用官方 openai SDK（默认）
    RawHTTPBackend : 只依赖 requests，自行拼 HTTP，便于审阅协议细节 / SDK 不可用时兜底
    MockBackend    : 离线演练，不发网络请求，用于演示与单元测试

重试策略：仅对「可重试」错误（连接失败、超时、408/409/429、5xx）做指数退避重试；
认证/参数类错误（400/401/403/404）立即失败，避免无意义等待。
"""

from __future__ import annotations

import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import LLMProfile
from .errors import ConfigError, LLMError

__all__ = ["ToolCall", "AssistantMessage", "LLMBackend", "OpenAIBackend", "RawHTTPBackend", "MockBackend", "build_backend"]

# 这些 HTTP 状态码值得重试；其余 4xx 视为不可恢复
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


# ----------------------------------------------------------------------------
# 统一数据模型
# ----------------------------------------------------------------------------
@dataclass
class ToolCall:
    """一次工具调用意图（与具体模型的表示解耦）。"""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""          # 模型给出的原始 arguments 字符串，便于报错回显
    malformed: bool = False          # True 表示 arguments 不是合法 JSON
    source: str = "native"           # native | text

    def fingerprint(self) -> str:
        """用于重复调用检测（同一工具 + 同一参数 = 同一指纹）。"""
        try:
            args = json.dumps(self.arguments, sort_keys=True, ensure_ascii=False)
        except TypeError:  # 含不可序列化对象
            args = repr(self.arguments)
        return f"{self.name}:{args}"


@dataclass
class AssistantMessage:
    """模型返回的 assistant 消息（与 provider 无关的规范形态）。"""

    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_history_item(self) -> Dict[str, Any]:
        """转成 OpenAI 消息格式，塞回历史。"""
        item: Dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            item["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.raw_arguments or json.dumps(c.arguments, ensure_ascii=False)},
                }
                for c in self.tool_calls
            ]
        return item


# ----------------------------------------------------------------------------
# 后端抽象
# ----------------------------------------------------------------------------
class LLMBackend(ABC):
    """所有后端的统一接口。"""

    def __init__(self, profile: LLMProfile):
        self.profile = profile
        # 密钥池：以 api_keys 为准，api_key 作为首个候选补在前面
        self._keys: List[str] = [k for k in (profile.api_keys or []) if k]
        if profile.api_key and profile.api_key not in self._keys:
            self._keys.insert(0, profile.api_key)
        self._key_index = 0
        self.on_key_rotate: Optional[Callable[[int, int], None]] = None

    @property
    def supports_native_tools(self) -> bool:
        return self.profile.native_tools

    # ---------------- 密钥轮换 ----------------
    def _rotate_key(self) -> bool:
        """切换到下一把密钥；没有更多候选时返回 False。"""
        if len(self._keys) < 2:
            return False
        nxt = self._key_index + 1
        if nxt >= len(self._keys):
            return False                      # 已用完，不再绕回第一个
        self._key_index = nxt
        self.profile.api_key = self._keys[nxt]
        self._rebuild_client()
        if self.on_key_rotate:
            try:
                self.on_key_rotate(nxt, len(self._keys))
            except Exception:                 # 回调出错不能影响主流程
                pass
        return True

    def _rebuild_client(self) -> None:
        """密钥变更后重建底层客户端；用不到长连接的子类无需实现。"""

    @staticmethod
    def _should_rotate(exc: "LLMError") -> bool:
        """哪些错误值得换 key：鉴权失败、限流、额度耗尽。"""
        if getattr(exc, "status", None) in (401, 403, 429):
            return True
        text = str(exc).lower()
        return any(k in text for k in
                   ("quota", "insufficient", "balance", "额度", "余额", "无效的令牌", "invalid token"))

    @abstractmethod
    def _send(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        on_delta=None,
    ) -> AssistantMessage:
        """真正发一次请求（由子类实现）。

        on_delta: 流式增量回调 (text_chunk: str) -> None。能流式的后端应在收到分片时
        即时调用；不能流式的后端（如 Mock）可以忽略它——调用方仍会拿到完整结果，
        所以"没有流式"只会退化成"一次性输出"，不会丢内容。
        """

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        on_delta=None,
    ) -> AssistantMessage:
        """带重试与错误归一化的对外入口。

        Args:
            messages: OpenAI 风格消息列表。
            tools: 工具 schema 列表；为 None 时不传 tools 字段（文本协议模式）。
            tool_choice: "auto" / "none" / 指定函数。
            on_delta: 流式增量回调（可选），签名 (text_chunk: str) -> None。
                流式期间会随收随调；若请求中途失败并重试，会先回调一条中断提示，
                避免用户把两段拼起来误读成一次完整输出。

        Returns:
            AssistantMessage。

        Raises:
            LLMError: 重试耗尽或不可恢复错误。
        """
        retries = max(0, int(self.profile.max_retries))
        last_exc: Optional[BaseException] = None
        rotations = 0
        max_rotations = max(0, len(self._keys) - 1)
        emitted = False        # 本次是否已经往回调里吐过内容

        def _delta(text: str) -> None:
            nonlocal emitted
            if text:
                emitted = True
                if on_delta:
                    on_delta(text)

        for attempt in range(retries + 1):
            try:
                return self._send(messages, tools=tools, tool_choice=tool_choice,
                                  on_delta=_delta if on_delta else None)
            except LLMError as exc:
                last_exc = exc
                # 凭据/配额类错误：换一把钥匙再试，对同一把 key 重试没有意义
                if rotations < max_rotations and self._should_rotate(exc) and self._rotate_key():
                    rotations += 1
                    continue
                if not exc.retryable or attempt >= retries:
                    raise
            except Exception as exc:  # 兜底：把 SDK/requests 的异常翻译成 LLMError
                last_exc = exc
                if attempt >= retries:
                    break
                llm_exc = LLMError(f"模型调用异常：{type(exc).__name__}: {exc}", retryable=True)
                if attempt >= retries:
                    raise llm_exc from exc
            # 已经吐过一半内容再重试，必须显式告知中断，否则用户会把两段拼成一次输出
            if emitted:
                _delta("\n[生成中断，正在重试]\n")
            # 指数退避 + 抖动，避免多个请求同时重试造成雪崩
            delay = (self.profile.retry_backoff ** attempt) + random.uniform(0, 0.4)
            time.sleep(min(delay, 30.0))

        raise LLMError(f"模型调用失败（已重试 {retries} 次）：{last_exc}")


def _tool_calls_from_raw(raw_calls) -> List[ToolCall]:
    """把 provider 形态的 tool_calls 解析成规范 ToolCall 列表。

    流式与非流式共用：无论 arguments 是整包还是分片累积来的，都走同一段解析，
    避免"两条通道各解析一遍、行为不一致"——那是本项目已经踩过的一类坑。
    """
    calls: List[ToolCall] = []
    for tc in raw_calls or []:
        raw_args = getattr(tc.function, "arguments", "") or "{}"
        try:
            args = json.loads(raw_args)
            if not isinstance(args, dict):
                raise ValueError("arguments 不是对象")
            malformed = False
        except Exception:  # noqa: BLE001 —— 参数不合法要如实回报给模型，不能抛给主循环
            args, malformed = {}, True
        calls.append(ToolCall(
            id=getattr(tc, "id", "") or "",
            name=getattr(tc.function, "name", "") or "",
            arguments=args,
            raw_arguments=raw_args,
            malformed=malformed,
        ))
    return calls


def _usage_from(resp: Any) -> Dict[str, int]:
    """提取 usage；没有就返回空 dict（调用方按"未知"处理，不编造数字）。"""
    usage = getattr(resp, "usage", None)
    if not usage:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


class _SimpleCall:
    """把流式累积出的 (id, name, arguments) 伪装成 provider 的 tool_call 对象，
    好让流式与非流式共用 _tool_calls_from_raw 这一段解析。"""

    def __init__(self, id_: str, name: str, arguments: str) -> None:
        self.id = id_
        self.function = _SimpleFn(name, arguments)


class _SimpleFn:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


# ----------------------------------------------------------------------------
# 后端 1：openai 官方 SDK
# ----------------------------------------------------------------------------
class OpenAIBackend(LLMBackend):
    """基于 openai SDK 的后端（允许使用模型厂商的 API 客户端库）。"""

    def __init__(self, profile: LLMProfile):
        super().__init__(profile)
        # 一旦确认该网关不支持流式就置 True，之后不再为每次调用白跑一次失败的流式请求
        self._stream_disabled = False
        try:
            from openai import OpenAI  # 延迟导入，用 RawHTTPBackend 时不必安装
        except ImportError as exc:  # pragma: no cover
            raise ConfigError("未安装 openai SDK，请 pip install openai，或改用 --backend http") from exc

        self._OpenAI = OpenAI
        self._client = self._build_client()

    def _build_client(self):
        return self._OpenAI(
            api_key=self.profile.api_key,
            base_url=self.profile.base_url,
            timeout=self.profile.timeout,
            max_retries=0,  # 重试由本模块统一控制，避免双重重试
            default_headers=self.profile.extra_headers or None,
        )

    def _rebuild_client(self) -> None:
        """轮换密钥后必须重建客户端——SDK 的 api_key 是在构造时固定的。

        空实现会让轮换失效：profile.api_key 换了，但 self._client 仍持有旧密钥。
        """
        self._client = self._build_client()

    def _base_kwargs(self, messages, tools, tool_choice) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.profile.model,
            "messages": messages,
            "temperature": self.profile.temperature,
            "top_p": self.profile.top_p,
            "max_tokens": self.profile.max_tokens,
        }
        if tools and self.supports_native_tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if self.profile.extra_body:
            kwargs.update(self.profile.extra_body)
        return kwargs

    def _send(self, messages, tools=None, tool_choice=None, on_delta=None) -> AssistantMessage:
        kwargs = self._base_kwargs(messages, tools, tool_choice)

        # 流式只在「调用方确实要增量」时启用；否则走最稳的整包返回。
        # 任何流式特有的失败都会自动退回整包，不会让用户卡住；
        # 且退回过一次后记下，后续调用不再重复浪费一次失败请求。
        if (on_delta and not getattr(self, "_stream_disabled", False)
                and getattr(self.profile, "stream", True)):
            try:
                return self._send_streaming(kwargs, on_delta)
            except LLMError as exc:
                # 网关不支持流式（含 stream_options）时退回整包，而不是直接失败
                if not self._looks_like_stream_unsupported(exc):
                    raise
                self._stream_disabled = True
            except Exception:
                self._stream_disabled = True

        return self._send_blocking(kwargs)

    @staticmethod
    def _looks_like_stream_unsupported(exc: BaseException) -> bool:
        """判断失败是否"像"网关不支持流式——是就退回整包，别让体验优化变成故障。"""
        text = str(exc).lower()
        markers = ("stream", "stream_options", "include_usage", "not support", "unsupported",
                   "unknown", "invalid_request", "400")
        return any(m in text for m in markers)

    def _send_blocking(self, kwargs: Dict[str, Any]) -> AssistantMessage:
        try:
            resp = self._client.chat.completions.create(stream=False, **kwargs)
        except Exception as exc:
            raise _translate_sdk_error(exc) from exc

        choice = (resp.choices or [None])[0]
        if choice is None:
            raise LLMError("模型返回空 choices", retryable=True)

        msg = choice.message
        return AssistantMessage(
            content=msg.content or "",
            tool_calls=_tool_calls_from_raw(getattr(msg, "tool_calls", None)),
            finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
            usage=_usage_from(resp),
            raw=resp,
        )

    def _send_streaming(self, kwargs: Dict[str, Any], on_delta) -> AssistantMessage:
        """流式请求：边收边吐内容，同时把 tool_calls 的分片按 index 攒起来。

        两个必须处理对的点：
        1. **tool_calls 是按 index 分片的**。一个函数调用的 id / name / arguments 会
           跨多个 chunk 到达，必须按 index 累积后再解析，否则 arguments 是残缺 JSON。
        2. **默认流式不带 usage**，而 /stats 的成本对账依赖 API 返回的 usage。
           所以要显式要 `stream_options={"include_usage": True}`；网关不支持时
           由 _send 退回整包（宁可不流式，也不能让成本统计悄悄失效）。
        """
        try:
            stream = self._client.chat.completions.create(
                stream=True, stream_options={"include_usage": True}, **kwargs
            )
        except Exception as exc:
            # 部分网关不接受 stream_options，去掉再试一次
            translated = _translate_sdk_error(exc)
            if not self._looks_like_stream_unsupported(translated):
                raise translated from exc
            try:
                stream = self._client.chat.completions.create(stream=True, **kwargs)
            except Exception as exc2:
                raise _translate_sdk_error(exc2) from exc2

        content_parts: List[str] = []
        acc: Dict[int, Dict[str, str]] = {}
        finish_reason = "stop"
        usage: Dict[str, int] = {}
        last_chunk: Any = None

        for chunk in stream:
            last_chunk = chunk
            if getattr(chunk, "usage", None):
                got = _usage_from(chunk)
                if got:
                    usage = got
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                on_delta(delta.content)
            for tc in getattr(delta, "tool_calls", None) or []:
                # index 缺失的网关（不规范但存在）一律归到 0，避免 KeyError 丢调用
                idx = getattr(tc, "index", None)
                idx = int(idx) if idx is not None else 0
                slot = acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                if getattr(tc, "id", None):
                    slot["id"] += tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] += fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"] += fn.arguments

        calls = _tool_calls_from_raw([
            _SimpleCall(slot["id"], slot["name"], slot["args"]) for _, slot in sorted(acc.items())
        ])
        return AssistantMessage(
            content="".join(content_parts),
            tool_calls=calls,
            finish_reason=finish_reason or "stop",
            usage=usage,
            raw=last_chunk,
        )


# ----------------------------------------------------------------------------
# 后端 2：requests 原生 HTTP
# ----------------------------------------------------------------------------
class RawHTTPBackend(LLMBackend):
    """不依赖任何厂商 SDK，直接发 HTTP 请求（便于看清协议、也方便排查网关差异）。

    端点约定：`POST {base_url}/chat/completions`，Header 带 `Authorization: Bearer <key>`。
    若 base_url 已包含 /chat/completions，则直接使用。
    """

    def __init__(self, profile: LLMProfile):
        super().__init__(profile)
        try:
            import requests  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ConfigError("未安装 requests，请 pip install requests，或改用 --backend sdk") from exc
        base = profile.base_url.rstrip("/")
        self._url = base if base.endswith("/chat/completions") else base + "/chat/completions"
        self._session = None

    def _client_session(self):
        import requests
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def _send(self, messages, tools=None, tool_choice=None, on_delta=None) -> AssistantMessage:
        # 本后端刻意保持"能看清协议"的极简实现，不做流式（SSE 解析会掩盖协议本身）；
        # 忽略 on_delta 是安全的：调用方仍会拿到完整的 AssistantMessage。
        import requests

        payload: Dict[str, Any] = {
            "model": self.profile.model,
            "messages": messages,
            "temperature": self.profile.temperature,
            "top_p": self.profile.top_p,
            "max_tokens": self.profile.max_tokens,
            "stream": False,
        }
        if tools and self.supports_native_tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if self.profile.extra_body:
            payload.update(self.profile.extra_body)

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.profile.api_key}"}
        headers.update(self.profile.extra_headers or {})

        try:
            resp = self._client_session().post(self._url, json=payload, headers=headers, timeout=self.profile.timeout)
        except requests.exceptions.Timeout as exc:
            raise LLMError(f"请求超时（{self.profile.timeout}s）", retryable=True) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMError(f"网络错误：{exc}", retryable=True) from exc

        if resp.status_code >= 400:
            raise LLMError(
                f"HTTP {resp.status_code}: {resp.text[:500]}",
                retryable=resp.status_code in _RETRYABLE_STATUS,
                status=resp.status_code,
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError(f"响应不是合法 JSON：{resp.text[:300]}", retryable=True) from exc

        return _parse_chat_completion_json(data)


def _parse_chat_completion_json(data: Dict[str, Any]) -> AssistantMessage:
    """解析 OpenAI 风格响应体（SDK 与裸 HTTP 共用）。"""
    choices = data.get("choices") or []
    if not choices:
        raise LLMError(f"响应缺少 choices：{str(data)[:300]}", retryable=True)
    choice = choices[0]
    msg = choice.get("message") or {}

    calls: List[ToolCall] = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args)
            if not isinstance(args, dict):
                raise ValueError("arguments 不是对象")
            malformed = False
        except Exception:
            args, malformed = {}, True
        calls.append(
            ToolCall(
                id=tc.get("id") or f"call_{i}",
                name=fn.get("name") or "",
                arguments=args,
                raw_arguments=raw_args,
                malformed=malformed,
            )
        )

    usage = data.get("usage") or {}
    return AssistantMessage(
        content=msg.get("content") or "",
        tool_calls=calls,
        finish_reason=choice.get("finish_reason") or "stop",
        usage={
            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
        },
        raw=data,
    )


def _translate_sdk_error(exc: BaseException) -> LLMError:
    """把 openai SDK 的异常归一化为 LLMError，并判断是否可重试。"""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    if status is not None:
        retryable = status in _RETRYABLE_STATUS
        return LLMError(f"HTTP {status}: {str(exc)[:400]}", retryable=retryable, status=status)
    # 常见的不可重试类型
    if "Authentication" in name or "Permission" in name or "NotFound" in name or "BadRequest" in name:
        return LLMError(f"{name}: {str(exc)[:400]}", retryable=False)
    if "RateLimit" in name or "APITimeout" in name or "Connection" in name or "InternalServer" in name:
        return LLMError(f"{name}: {str(exc)[:400]}", retryable=True)
    return LLMError(f"{name}: {str(exc)[:400]}", retryable=True)


# ----------------------------------------------------------------------------
# 后端 3：离线 Mock
# ----------------------------------------------------------------------------
class MockBackend(LLMBackend):
    """脚本化后端：按序返回预设响应，用于离线演示与测试（不发任何网络请求）。

    script 元素示例：
        {"content": "我先看一下目录", "tool_calls": [{"name": "list_dir", "arguments": {"path": "."}}]}
    脚本耗尽后，返回一段不带工具调用的收尾消息，主循环据此正常结束。
    """

    def __init__(self, profile: LLMProfile, script: Optional[List[Dict[str, Any]]] = None):
        super().__init__(profile)
        self.script = list(script or [])
        self.cursor = 0
        self.recorded: List[Dict[str, Any]] = []

    def _send(self, messages, tools=None, tool_choice=None, on_delta=None) -> AssistantMessage:
        self.recorded.append({"messages": messages, "tools": tools})
        if self.cursor >= len(self.script):
            return AssistantMessage(content="FINAL: 演示脚本执行完毕。", finish_reason="stop")

        item = self.script[self.cursor]
        self.cursor += 1
        raw_calls = item.get("tool_calls") or []

        # 仿真两种协议：不支持原生工具调用时，把调用渲染成文本协议的代码块，
        # 这样 Mock 后端能真实地走过 parser 的文本解析分支。
        if self.supports_native_tools:
            calls = []
            for i, c in enumerate(raw_calls):
                args = c.get("arguments") or {}
                calls.append(
                    ToolCall(
                        id=f"mock_{self.cursor}_{i}",
                        name=c.get("name", ""),
                        arguments=args,
                        raw_arguments=json.dumps(args, ensure_ascii=False),
                        source="native",
                    )
                )
            content = item.get("content", "")
            finish = "tool_calls" if calls else "stop"
        else:
            calls = []
            blocks = []
            for c in raw_calls:
                blocks.append(
                    "```json\n"
                    + json.dumps({"tool": c.get("name", ""), "args": c.get("arguments") or {}}, ensure_ascii=False, indent=2)
                    + "\n```"
                )
            content = (item.get("content", "") + "\n\n" + "\n\n".join(blocks)).strip()
            finish = "stop"

        return AssistantMessage(
            content=content,
            tool_calls=calls,
            finish_reason=finish,
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )


# ----------------------------------------------------------------------------
# 工厂
# ----------------------------------------------------------------------------
def build_backend(profile: LLMProfile, kind: str = "auto", script: Optional[List[Dict[str, Any]]] = None) -> LLMBackend:
    """构造后端。

    Args:
        profile: 模型档位配置。
        kind: auto | sdk | http | mock。auto 时优先 SDK，SDK 不可用退化为 HTTP。
        script: 仅 MockBackend 使用。
    """
    kind = (kind or "auto").lower()
    if kind == "mock":
        return MockBackend(profile, script)
    if kind == "http":
        return RawHTTPBackend(profile)
    if kind == "sdk":
        return OpenAIBackend(profile)
    # auto
    try:
        return OpenAIBackend(profile)
    except ConfigError:
        return RawHTTPBackend(profile)
