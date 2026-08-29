"""
模型输出解析层 —— 把"模型说的话"变成"可执行的工具调用意图"。

双通道设计（这是本项目对模型兼容性的核心处理）：
    通道 A（首选）：模型原生 tool_calls 结构化输出 —— 精确、无需正则。
    通道 B（兜底）：文本协议。当端点/模型不支持 function calling
                   （如部分本地模型、网关被阉割）时启用，约定模型输出：

                        ```json
                        {"tool": "read_file", "args": {"path": "src/main.py"}}
                        ```

                    解析器负责从任意自然语言中把这些代码块抠出来、校验、转成 ToolCall。
                    校验失败的调用不会静默丢弃，而是生成一条"错误回执"回灌给模型自我修正。

解析器的三条硬规则：
    1. 绝不猜测 —— 无法确定的内容一律转成 issue，而不是编一个调用；
    2. 绝不静默 —— 每个被拒绝/修正的调用都要有给模型看的说明；
    3. 绝不越权 —— 只接受注册表中的工具名与参数类型。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import ParseError
from .llm import AssistantMessage, ToolCall

__all__ = ["ParseOutcome", "ToolCallParser", "extract_text_calls", "validate_arguments"]

# ----------------------------------------------------------------------------
# 文本协议识别
# ----------------------------------------------------------------------------
# 1) ```json ... ``` / ```tool ... ``` / ```tool_code ... ``` 代码块
_FENCE_RE = re.compile(
    r"```(?:json|tool|tool_call|tool_code|actions?)?\s*\n(?P<body>.*?)```",
    re.DOTALL | re.IGNORECASE,
)
# 2) <tool_call>...</tool_call> 标签（部分模型偏好这种写法）
_TAG_RE = re.compile(r"<tool_call\s*>(?P<body>.*?)</tool_call\s*>", re.DOTALL | re.IGNORECASE)
# 3) 形如 `finish: ...` 的收尾标记
_FINAL_RE = re.compile(r"^\s*(FINAL|最终答案|完成)\s*[:：]", re.IGNORECASE)

# JSON 里工具名的兼容写法
_NAME_KEYS = ("tool", "name", "tool_name", "function", "action")
# JSON 里参数体的兼容写法
_ARGS_KEYS = ("args", "arguments", "parameters", "params", "input")

_JSON_TYPES = {
    "string": str,
    "str": str,
    "integer": int,
    "int": int,
    "number": (int, float),
    "boolean": bool,
    "bool": bool,
    "array": list,
    "object": dict,
}


@dataclass
class ParseOutcome:
    """一次解析的结果。"""

    narration: str = ""                     # 模型的自然语言说明（给用户看）
    calls: List[ToolCall] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)   # 需要回灌给模型的错误/警告
    is_final: bool = False                  # 是否为不带工具调用的收尾回答
    source: str = "native"

    @property
    def ok(self) -> bool:
        return bool(self.calls) and not self.issues

    def issue_text(self) -> str:
        return "\n".join(f"- {i}" for i in self.issues)


# ----------------------------------------------------------------------------
# 参数校验
# ----------------------------------------------------------------------------
def validate_arguments(schema: Dict[str, Any], args: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """按 JSON Schema 校验并补齐参数。

    属于"宽容但可控"的校验：
      - 缺失但有默认值 → 自动补默认值（不打扰模型）
      - 缺失且必填     → 记为错误
      - 类型不匹配但可强制转换（如 "42" → 42）→ 转换并给警告
      - 类型不匹配且无法转换 → 记为错误
      - 多余参数       → 丢弃并给警告（防止模型把正文塞进参数）

    Returns:
        (清洗后的参数, 问题列表)
    """
    issues: List[str] = []
    props: Dict[str, Any] = (schema or {}).get("properties", {}) or {}
    required: Sequence[str] = (schema or {}).get("required", []) or []
    clean: Dict[str, Any] = {}

    for key in required:
        if key not in args or args[key] in (None, ""):
            issues.append(f"缺少必填参数 `{key}`")

    for key, value in (args or {}).items():
        spec = props.get(key)
        if spec is None:
            issues.append(f"出现未知参数 `{key}`（已忽略）；可用参数：{', '.join(sorted(props)) or '无'}")
            continue

        expected = spec.get("type")
        if isinstance(expected, list):
            expected = next((e for e in expected if e != "null"), None)

        if expected in _JSON_TYPES and value is not None:
            py_type = _JSON_TYPES[expected]
            if not isinstance(value, py_type):
                coerced = _try_coerce(value, expected)
                if coerced is _NO_MATCH:
                    issues.append(f"参数 `{key}` 类型错误：期望 {expected}，实际 {type(value).__name__}")
                    continue
                issues.append(f"参数 `{key}` 已从 {type(value).__name__} 自动转换为 {expected}")
                value = coerced

        if "enum" in spec and value not in spec["enum"]:
            issues.append(f"参数 `{key}` 取值 {value!r} 不合法，可选：{spec['enum']}")
            continue

        clean[key] = value

    for key, spec in props.items():
        if key not in clean and "default" in spec:
            clean[key] = spec["default"]

    return clean, issues


_NO_MATCH = object()


def _try_coerce(value: Any, expected: Optional[str]):
    """尽力做无损的常见类型转换。"""
    try:
        if expected in ("integer", "int"):
            return int(str(value).strip())
        if expected in ("number",):
            return float(str(value).strip())
        if expected in ("boolean", "bool"):
            if isinstance(value, str):
                low = value.strip().lower()
                if low in ("true", "1", "yes", "y", "on"):
                    return True
                if low in ("false", "0", "no", "n", "off"):
                    return False
            return _NO_MATCH
        if expected in ("string", "str"):
            if isinstance(value, (int, float, bool)):
                return str(value)
            return _NO_MATCH
        if expected in ("array",):
            if isinstance(value, str):
                return [v.strip() for v in value.split(",") if v.strip()]
            return _NO_MATCH
    except (ValueError, TypeError):
        return _NO_MATCH
    return _NO_MATCH


# ----------------------------------------------------------------------------
# 文本协议解析
# ----------------------------------------------------------------------------
def extract_text_calls(text: str) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    """从自然语言中抽取工具调用块。

    Returns:
        (去掉代码块后的正文, 原始调用字典列表, 抽取阶段的问题列表)
    """
    if not text:
        return "", [], []

    candidates: List[str] = []
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group("body"))
    for m in _TAG_RE.finditer(text):
        candidates.append(m.group("body"))

    narration = _FENCE_RE.sub("", text)
    narration = _TAG_RE.sub("", narration).strip()

    raw_calls: List[Dict[str, Any]] = []
    issues: List[str] = []

    for body in candidates:
        body = body.strip()
        if not body:
            continue
        # 兼容一个代码块里写多个调用：[{...}, {...}]
        parsed = _loads_lenient(body)
        if parsed is None:
            issues.append(f"以下代码块不是合法 JSON，已跳过：\n{body[:300]}")
            continue
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            if not isinstance(item, dict):
                issues.append(f"工具调用必须是 JSON 对象，收到：{type(item).__name__}")
                continue
            name = _pick_first(item, _NAME_KEYS)
            if not name:
                issues.append(f"工具调用缺少工具名（需要 {'/'.join(_NAME_KEYS)} 之一）：{json.dumps(item, ensure_ascii=False)[:200]}")
                continue
            args = _pick_first(item, _ARGS_KEYS)
            if args is None:
                # 兼容扁平写法：{"tool": "x", "path": "y"} —— 除名字键外全是参数
                args = {k: v for k, v in item.items() if k not in _NAME_KEYS}
            if not isinstance(args, dict):
                issues.append(f"工具 `{name}` 的参数必须是 JSON 对象，收到 {type(args).__name__}")
                continue
            raw_calls.append({"name": str(name), "arguments": args, "raw": body})

    return narration, raw_calls, issues


def _loads_lenient(body: str) -> Optional[Any]:
    """宽松 JSON 解析：先严格解析，失败后尝试截取最外层 {...} 或 [...] 再解析。"""
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = body.find(opener), body.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(body[start: end + 1])
            except json.JSONDecodeError:
                continue
    # 最后尝试：去掉尾随逗号
    try:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", body))
    except json.JSONDecodeError:
        return None


def _pick_first(item: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in item:
            return item[k]
    return None


# ----------------------------------------------------------------------------
# 解析器主体
# ----------------------------------------------------------------------------
class ToolCallParser:
    """把 AssistantMessage 解析为规范化的 ToolCall 列表。

    Args:
        registry: 工具注册表，提供 schema 用于校验（见 tools/base.py）。
        use_native: 是否优先走原生 tool_calls 通道。
        max_calls_per_turn: 单轮最多接受多少个调用，防止模型一次性刷屏。
    """

    def __init__(self, registry, use_native: bool = True, max_calls_per_turn: int = 8):
        self.registry = registry
        self.use_native = use_native
        self.max_calls_per_turn = max_calls_per_turn

    # ---------------- 对外主入口 ----------------
    def parse(self, msg: AssistantMessage) -> ParseOutcome:
        """解析一条 assistant 消息。任何异常都不外抛，全部转成 issues。"""
        if self.use_native and msg.tool_calls:
            return self._parse_native(msg)
        return self._parse_text(msg)

    # ---------------- 通道 A ----------------
    def _parse_native(self, msg: AssistantMessage) -> ParseOutcome:
        outcome = ParseOutcome(narration=(msg.content or "").strip(), source="native")
        for tc in msg.tool_calls:
            if tc.malformed:
                outcome.issues.append(
                    f"工具 `{tc.name}` 的 arguments 不是合法 JSON，原始内容：{tc.raw_arguments[:300]}"
                )
                continue
            self._build_call(outcome, tc.name, tc.arguments, tc.id, "native")
        self._post_process(outcome, msg)
        return outcome

    # ---------------- 通道 B ----------------
    def _parse_text(self, msg: AssistantMessage) -> ParseOutcome:
        text = msg.content or ""
        narration, raw_calls, issues = extract_text_calls(text)
        outcome = ParseOutcome(narration=narration.strip(), source="text", issues=issues)
        for idx, raw in enumerate(raw_calls):
            self._build_call(outcome, raw["name"], raw["arguments"], f"text_{idx}_{raw['name']}", "text")
        # 文本通道下，"没有调用块"通常意味着模型认为任务已完成
        self._post_process(outcome, msg)
        if not outcome.calls and not outcome.issues:
            outcome.is_final = True
        return outcome

    # ---------------- 公共处理 ----------------
    def _build_call(self, outcome: ParseOutcome, name: str, args: Dict[str, Any], call_id: str, source: str) -> None:
        if len(outcome.calls) >= self.max_calls_per_turn:
            outcome.issues.append(f"单轮调用数量超过上限 {self.max_calls_per_turn}，多余的调用已丢弃")
            return

        spec = self.registry.get(name)
        if spec is None:
            outcome.issues.append(
                f"未知工具 `{name}`。可用工具：{', '.join(self.registry.names())}"
            )
            return

        clean, issues = validate_arguments(spec.parameters, args)
        if issues:
            for i in issues:
                outcome.issues.append(f"{name}: {i}")
            # 若只是缺必填参数，不再生成调用；模型下一轮会补上
            if any("缺少必填参数" in i for i in issues):
                return

        outcome.calls.append(
            ToolCall(
                id=call_id or f"{source}_{name}",
                name=name,
                arguments=clean,
                raw_arguments=json.dumps(args, ensure_ascii=False),
                source=source,
            )
        )

    def _post_process(self, outcome: ParseOutcome, msg: AssistantMessage) -> None:
        """收尾判定与问题归一。"""
        if not outcome.calls and not outcome.issues:
            content = (msg.content or "").strip()
            outcome.is_final = bool(content) and (
                not self.use_native or bool(_FINAL_RE.match(content)) or not msg.tool_calls
            )
        if msg.finish_reason == "length":
            outcome.issues.append("上一次回复因达到 max_tokens 被截断，请缩短输出、分多次完成。")

    # ---------------- 供主循环生成"纠错提示" ----------------
    @staticmethod
    def build_correction_prompt(outcome: ParseOutcome, use_native: bool) -> str:
        """把解析问题组织成给模型的纠错指令。"""
        lines = [
            "你上一次的输出存在问题，无法执行：",
            outcome.issue_text(),
        ]
        if use_native:
            lines.append("请重新输出，使用结构化的 tool_calls，参数必须是合法 JSON，且只使用已提供的工具。")
        else:
            lines.append(
                "请重新输出，严格遵守文本协议：每个调用单独写在一个 ```json 代码块中，"
                '格式为 {"tool": "工具名", "args": {参数}}，不要在一个块里混入其它文字。'
            )
        return "\n".join(lines)
