"""
联网读取工具：fetch_url —— 用标准库 urllib 自写 GET，把网页/接口内容取回给模型。

为什么值得做：商业 code agent（Claude Code / Codex）都能“查资料”——读官方文档、
查 API 用法、看报错对应的 issue。本项目此前完全离线，模型遇到不熟悉的库只能凭记忆瞎猜。
自写而非调用服务端 fetch 工具，符合题目“重要逻辑需自行编写、不得依赖 API 服务端托管工具”。

安全边界（把“模型的输出/外部输入”当不可信处理）：
    · 仅允许 http / https 两种 scheme，file:// / ftp:// 等一律拒绝；
    · 响应体设硬上限（默认 2MB 原始字节），避免把内存撑爆；
    · 整个请求设超时（默认 20s），绝不挂死；
    · 仅抽取文本内容（HTML 会轻量去标签），不执行任何脚本。
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from typing import Any, Dict, Tuple

from .base import ToolContext, ToolResult, tool_spec

__all__ = ["fetch_url", "register"]

_MAX_BYTES = 2 * 1024 * 1024          # 原始响应上限 2MB
_DEFAULT_TIMEOUT = 20
_DEFAULT_MAX_CHARS = 12_000           # 回给模型的文本上限（registry 还会再按 max_tool_output_chars 截断）

_USER_AGENT = "MiniCode-CodingAgent/1.0 (+https://github.com/)"
_SCRIPT_RE = re.compile(r"<script[\s\S]*?</script>", re.IGNORECASE)
_STYLE_RE = re.compile(r"<style[\s\S]*?</style>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_BLANK_RE = re.compile(r"\n{3,}")


def _strip_html(html: str) -> str:
    """把 HTML 轻量转成可读文本：去 script/style、去标签、压缩空白。"""
    text = _SCRIPT_RE.sub(" ", html)
    text = _STYLE_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    text = text.replace("\r", "")
    text = _BLANK_RE.sub("\n\n", text)
    return text.strip()


def _validate(url: str) -> Tuple[bool, str]:
    """只允许 http/https；其余 scheme 一律拒绝。返回 (ok, reason)。"""
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError as exc:
        return False, f"URL 无法解析：{exc}"
    if parsed.scheme not in ("http", "https"):
        return False, f"不支持的协议 {parsed.scheme!r}；fetch_url 只允许 http/https。"
    if not parsed.netloc:
        return False, "URL 缺少主机名（host）。"
    return True, ""


@tool_spec(
    name="fetch_url",
    description=(
        "抓取一个网址（http/https）的文本内容并返回，用于查官方文档、看 API 用法、"
        "读报错对应的资料。HTML 页面会自动抽取正文文本。\n"
        "适合在“不确定某个库/接口怎么用”时自查，而不是凭记忆瞎写代码。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "要抓取的网址，必须是以 http:// 或 https:// 开头的完整 URL"},
            "max_chars": {
                "type": "integer",
                "description": "返回文本最多多少字符，默认 12000，超长自动截断",
                "default": _DEFAULT_MAX_CHARS,
            },
            "timeout": {
                "type": "integer",
                "description": "请求超时秒数，默认 20",
                "default": _DEFAULT_TIMEOUT,
            },
        },
        "required": ["url"],
    },
    category="检索",
    when_not_to_use=(
        "要读的是工作区里的本地文件用 read_file，别拿 fetch_url 去读本地路径；"
        "file:// 等协议不被支持。也不要用它访问需要登录鉴权的内网页面（多半取不到内容）。"
        "读到的内容只是参考，关键接口仍需用 run_command 实际验证。"
    ),
)
def fetch_url(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    url = (args.get("url") or "").strip()
    if not url:
        return ToolResult.failure("url 不能为空", hint="请传入以 http:// 或 https:// 开头的完整网址。")
    ok, reason = _validate(url)
    if not ok:
        return ToolResult.failure(reason)

    max_chars = max(200, min(int(args.get("max_chars") or _DEFAULT_MAX_CHARS), 50_000))
    timeout = max(2, min(int(args.get("timeout") or _DEFAULT_TIMEOUT), 60))

    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "") or ""
            raw = resp.read(_MAX_BYTES)
            status = getattr(resp, "status", resp.getcode())
    except urllib.error.HTTPError as exc:
        return ToolResult.failure(
            f"抓取失败：HTTP {exc.code} {exc.reason}",
            hint="可能是地址不存在或服务器拒绝；检查 URL 是否正确。",
            meta={"http_status": exc.code},
        )
    except urllib.error.URLError as exc:
        return ToolResult.failure(
            f"抓取失败：无法连接（{exc.reason}）",
            hint="检查网络是否可达、URL 是否拼写正确。",
        )
    except Exception as exc:  # noqa: BLE001 —— 网络异常五花八门，统一兜底为失败回执
        return ToolResult.failure(f"抓取失败：{exc}", hint="该地址可能无法访问或超时。")

    # 解码：优先按响应声明的编码，失败再退 UTF-8（替换兜底）
    charset = "utf-8"
    m = re.search(r"charset=([\w-]+)", content_type, re.IGNORECASE)
    if m:
        charset = m.group(1).strip()
    try:
        text = raw.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        text = raw.decode("utf-8", errors="replace")

    if "html" in content_type.lower():
        text = _strip_html(text)
    else:
        # 非 HTML（如纯文本 / json / markdown）：保留原样，但清掉回车
        text = text.replace("\r", "")

    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    note = "（内容已截断）" if truncated else ""
    return ToolResult.success(
        f"[fetch_url] {url}（HTTP {status}，{len(raw)} 字节原始响应）{note}\n\n{text}",
        meta={"http_status": status, "content_type": content_type, "chars": len(text), "truncated": truncated},
    )


def register(registry) -> None:
    registry.register(fetch_url)
