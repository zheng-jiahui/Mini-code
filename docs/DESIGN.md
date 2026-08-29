# MiniCode 设计文档

> 目标：不依赖任何 Agent 框架/SDK，只用 OpenAI 兼容的**聊天补全客户端**，
> 自行实现对话历史、工具定义与本地执行、输出解析、循环终止、错误处理、上下文管理。
>
> 依赖白名单：`openai`（API 客户端库）/ `requests` / `PyYAML` / `tiktoken`（可选）。
> 未使用：LangChain、LlamaIndex、OpenAI Agents SDK、Claude Agent SDK、AutoGen、CrewAI。

---

## 1. 项目文件结构

```
.
├── run.py                    # 入口：python run.py [-t "任务"] [--mock] [--profile x]
├── config.example.yaml       # 配置模板（入库）
├── config.yaml               # 真实配置，含 API 凭据（.gitignore，不入库）
├── agent/
│   ├── config.py             # 配置加载：YAML/JSON → 环境变量覆盖 → CLI 覆盖 → 校验
│   ├── errors.py             # 错误分层：Config / LLM / Parse / Tool / Security / Budget / Abort
│   ├── llm.py                # LLM 后端（openai SDK / requests 原生 / Mock）+ 重试与退避
│   ├── parser.py             # 输出解析：原生 tool_calls 通道 + 文本协议通道 + 参数校验
│   ├── history.py            # 对话历史、token 估算、回执截断、自动摘要压缩
│   ├── prompts.py            # System Prompt 与注入式提示词（纠错/预算/停滞）
│   ├── security.py           # 路径沙箱、危险命令识别、输出脱敏与截断
│   ├── ui.py                 # 终端渲染（事件流、彩色、确认交互、spinner）
│   ├── loop.py               # ★ 主循环：编排、终止判定、预算控制
│   ├── cli.py                # 参数解析 + REPL（/tools /compact /undo /save …）
│   └── tools/
│       ├── base.py           # ToolSpec / ToolRegistry / ToolResult（自建工具系统）
│       ├── filesystem.py     # read_file / write_file / list_dir
│       ├── shell.py          # run_command
│       ├── search.py         # grep_search / find_files
│       └── meta.py           # finish / ask_user（控制类工具）
└── tests/
    ├── test_smoke.py         # 10 个用例：双通道、沙箱、危险命令、压缩、停滞检测…
    └── test_fake_server.py   # 本地假服务端，验证真实 HTTP 协议链路
```

**职责划分原则**：`loop.py` 只做编排，不认识任何具体工具，也不认识任何厂商；
工具由注册表注入，模型由后端抽象隔离。新增一个工具只需改 `tools/` 下的一个文件，主循环零改动。

---

## 2. 主循环：伪代码与流程图

### 2.1 流程图

```
用户输入 ──▶ history.add_user(task)
                │
                ▼
        ┌───────────────────────┐
        │ 预算检查 / 上下文压缩  │  tokens ≥ 75% 预算 → 摘要压缩
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐   失败且可重试（限流/超时/5xx）
        │   backend.chat()      │◀───────────┐  指数退避 + 抖动
        └───────────┬───────────┘            │
                    ▼                        │
        ┌───────────────────────┐            │
        │   parser.parse()      │──有 issues──┘  把问题原文回灌给模型自纠
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │ 存在工具调用？        │──否──▶ 判定为收尾回答 ──▶ 结束(model_final)
        └───────────┬───────────┘
                    ▼ 是
        ┌───────────────────────┐
        │ registry.execute()    │  异常绝不外泄 → 转成 ok=False 回执
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │ 回执写回 history      │  原生→role=tool；文本协议→role=user
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────────────────────┐
        │ 终止条件检查                           │
        │  ·调用了 finish                        │──▶ 结束(finish)
        │  ·连续失败 ≥ max_consecutive_errors    │──▶ 结束(too_many_errors)
        │  ·最近 3 次调用指纹完全相同（≥2 轮）    │──▶ 结束(no_progress)
        │  ·步数 ≥ max_steps                     │──▶ 结束(max_steps)
        │  ·Ctrl-C                               │──▶ 结束(aborted)
        └───────────┬───────────────────────────┘
                    └── 均未触发 ──▶ 回到顶部
```

### 2.2 伪代码

```python
def run(task):
    history.add_user(task)
    for step in 1..max_steps:
        maybe_compact()                       # 上下文预算
        if step >= 0.8 * max_steps:
            history.add_note(BUDGET_WARNING)

        msg = backend.chat(history.payload(), tools=registry.schemas())
        history.add_assistant(msg)

        outcome, _ = parse_with_retry(msg)    # 解析失败 → 注入纠错提示，最多 max_parse_retries 次
        if not outcome.calls:
            return RunResult(answer=msg.content, reason="model_final")

        for call in outcome.calls:
            if call.name == "finish":
                return RunResult(answer=call.args["summary"], reason="finish")
            result = registry.execute(call.name, call.args, ctx)   # 永不抛异常
            history.add_tool_result(call.id, call.name, result.render(), style=msg_style)
            update_error_streak(result)       # 连续失败 → too_many_errors
            fingerprints.append(call.fingerprint())   # 重复调用检测

        if stagnation_detected():
            return RunResult(reason="no_progress")
    return RunResult(reason="max_steps")
```

### 2.3 为什么这样设计（答辩要点）

- **终止条件必须"多路冗余"**。只靠"模型不再调工具"判断结束是不可靠的：模型可能中途停手、
  可能陷入死循环。本项目同时用「显式 `finish` 工具」「配额上限」「指纹去重」三重保险。
- **回执必须回灌而不是丢弃**。工具报错是给模型最有用的信号，比崩溃有价值得多。

---

## 3. System Prompt（完整内容）

由 `agent/prompts.py: build_system_prompt()` 动态生成；工具清单来自 `registry.describe()`，
避免出现"提示词写了工具、代码里没注册"的不一致。

```text
# 角色
你是一个运行在用户本地机器上的自主编程智能体（Coding Agent）。
你通过调用本地工具来读写文件、执行命令，独立、完整地完成用户交给你的编程任务。

# 运行环境
- 操作系统：{platform.system()} {platform.release()}
- Shell：{PowerShell / cmd | sh}
- Python：{sys.version}
- 工作目录（workspace）：{绝对路径}
- 当前时间：{YYYY-MM-DD HH:MM:SS}
- 路径规则：所有相对路径都相对工作目录解析；绝对路径若位于工作区之外会被拒绝，
  所有文件操作必须留在工作区内。

# 可用工具
【文件】
- read_file(path: string, offset?: integer, limit?: integer) —— 读取文本文件内容，带行号返回…
- write_file(path: string, content: string, append?: boolean) —— 写入文件（整体覆盖语义）…
- list_dir(path?: string, depth?: integer, max_entries?: integer) —— 列出目录结构…
【检索】
- grep_search(pattern: string, …)     - find_files(pattern?: string, …)
【执行】
- run_command(command: string, cwd?: string, timeout?: integer) —— …
【控制】
- finish(summary: string) —— 结束任务并给出总结…
- ask_user(question: string) —— 需求存在歧义时向用户提问…

# 工具调用协议
（native_tools=true 时注入 PROTOCOL_NATIVE，否则注入 PROTOCOL_TEXT，见第 4 节）

# 工作方式（务必遵守）
1. 先看清再动手：先用 list_dir / find_files / grep_search / read_file 弄清项目结构与现状，再修改。
2. 修改任何已存在的文件之前，必须先 read_file 读取原文。write_file 是整体覆盖语义，禁止凭猜测重写。
3. 小步验证：每完成一处改动，就用 run_command 跑一次（测试/脚本/编译），根据真实输出决定下一步。
   禁止凭"我觉得应该没问题"就宣布完成。
4. 命令必须非交互式、可重复执行（用 `pytest -q` 而不是会等待输入的 `pytest`）。
5. 报错处理：完整阅读错误信息，定位根因再修复。同一条命令连续失败 2 次后必须换思路，禁止原地重试。
6. 依赖最小化：不要引入与任务无关的新依赖；确实需要时，先说明原因再安装。
7. 不要为了"显得完整"而编造结果：没跑过的命令不许说跑通了，没读过的文件不许下结论。

# 安全边界
- 禁止破坏性操作：递归删除、格式化磁盘、强制推送、硬重置 Git、关停系统等。
- 禁止读取、打印或写入密钥/凭据内容（config.yaml、.env、*.pem、id_rsa 等）。
- 不要改写 Git 历史，不要向远端推送任何提交。
- 若指令与安全边界冲突，停下来向用户说明，而不是绕过。

# 结束任务
当任务已完成、或因信息不足确实无法继续时：
1. 调用 `finish` 工具，并在 summary 中说明：① 改动了哪些文件；② 用什么命令验证、结果如何；③ 遗留问题。
2. 若你确实无法调用工具，则在回复开头另起一行写 `FINAL:`，再给出同样的总结。
不要在还没验证过的情况下调用 finish。

# 表达风格
- 用简体中文交流；代码与注释沿用目标项目的既有语言风格。
- 简洁务实：每次调用工具前用一句话说明目的，不复述用户的问题，不写客套话。
- 面向结果汇报：说清"做了什么、验证结果是什么"，而不是"我将会做什么"。
```

**设计取舍**：只写模型"猜不到"的信息（协议、边界、终止条件）。
"你是一个乐于助人的 AI"这类话没有信息量，还白占 token。

---

## 4. 工具调用的输出格式规范

### 通道 A：原生 function calling（首选）

发送：

```jsonc
{
  "model": "gpt-4o-mini",
  "messages": [{"role": "system", "content": "…"}, {"role": "user", "content": "任务"}],
  "tools": [{"type": "function", "function": {"name": "read_file", "description": "…", "parameters": {…}}}],
  "tool_choice": "auto",
  "temperature": 0.2
}
```

模型返回 → 解析为 `ToolCall(id, name, arguments)`；回执以 `role=tool` + `tool_call_id` 回灌。

### 通道 B：文本协议（模型不支持 function calling 时自动启用）

约定模型输出（解析器识别 ```` ```json ```` / ```tool``` / `<tool_call>` 三种包裹）：

```json
{"tool": "read_file", "args": {"path": "src/main.py", "limit": 40}}
```

规则：
- 一个代码块只写一个 JSON 对象，块内不得混入其它文字；
- 参数必须是合法 JSON（`true/false/null`，不是 Python 的 `True/None`）；
- 兼容键名：`tool|name|tool_name|function` 与 `args|arguments|parameters|params`；
- 兼容扁平写法 `{"tool": "x", "path": "y"}` —— 除名字键外一律视为参数；
- 回执以 `role=user` 的 `<tool_result name="…" status="ok|error">…</tool_result>` 回灌
  （因为 OpenAI 协议要求 `role=tool` 必须前置一条带 `tool_calls` 的 assistant 消息）。

### 解析器的三条硬规则

1. **绝不猜测**：无法确定就转成 issue，而不是编一个调用；
2. **绝不静默**：每个被拒绝/修正的调用都生成一条给模型看的错误回执；
3. **绝不越权**：只接受注册表中的工具名与参数类型（按 JSON Schema 校验，缺失默认值自动补齐，
   类型可无损转换时自动转换并给警告，多余参数丢弃并给警告）。

---

## 5. 核心模块接口定义

### 5.1 配置 `agent/config.py`

| 函数 / 类 | 签名 | 输入 → 输出 |
|---|---|---|
| `load_config` | `(explicit: str\|None, profile: str\|None, cli_overrides: dict\|None, require_api_key: bool) -> Config` | 配置文件路径 → 校验后的 `Config`；缺 key/路径非法抛 `ConfigError` |
| `LLMProfile` | dataclass | `base_url / api_key / model / temperature / top_p / max_tokens / timeout / max_retries / native_tools / extra_headers / extra_body` |
| `AgentConfig` | dataclass | `workspace / max_steps / command_timeout / max_tool_output_chars / max_context_tokens / compact_* / restrict_to_workspace / command_policy / backup_on_write …` |
| `LLMProfile.masked` | `() -> dict` | 返回密钥打码后的可打印视图 |

### 5.2 LLM 后端 `agent/llm.py`

```python
class LLMBackend(ABC):
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_choice=None, on_delta=None) -> AssistantMessage
    # 内部：_send() 由子类实现；chat() 统一做指数退避重试与错误归一化

@dataclass
class AssistantMessage:
    content: str
    tool_calls: list[ToolCall]      # ToolCall(id, name, arguments, raw_arguments, malformed, source)
    finish_reason: str
    usage: dict[str, int]

build_backend(profile, kind: "auto|sdk|http|mock", script=None) -> LLMBackend
```

### 5.3 解析器 `agent/parser.py`

```python
class ToolCallParser:
    def __init__(self, registry, use_native: bool = True, max_calls_per_turn: int = 8)
    def parse(self, msg: AssistantMessage) -> ParseOutcome
    # ParseOutcome(narration, calls, issues, is_final, source)

extract_text_calls(text: str) -> tuple[str, list[dict], list[str]]
validate_arguments(schema: dict, args: dict) -> tuple[dict, list[str]]
ToolCallParser.build_correction_prompt(outcome, use_native) -> str
```

### 5.4 工具系统 `agent/tools/base.py`

```python
@dataclass
class ToolSpec:
    name, description, parameters(JSON Schema), handler, dangerous, hidden, category
    openai_schema() -> dict          # 转成 function calling schema
    signature() -> str               # read_file(path: string, limit?: integer)
    doc() -> str                     # 给 System Prompt 用的一行说明

class ToolRegistry:
    register(spec) / register_many(specs)
    get(name) -> ToolSpec | None
    schemas() -> list[dict]          # 发给模型的工具列表
    describe() -> str                # 给 System Prompt 的工具清单
    execute(name, args, ctx, call_id="") -> ToolResult    # 保证不抛异常

@dataclass
class ToolResult:
    ok: bool; output: str; error: str; data; meta: dict; elapsed: float
    success(output, **kw) / failure(error, hint, **kw) / from_exception(exc, tool)
    render(max_chars=0) -> str                       # 给模型看（截断+脱敏）
    to_message(call_id, name, style="native|text")   # 给历史用
```

工具处理函数统一签名：`handler(args: dict, ctx: ToolContext) -> ToolResult`。
`ToolContext` 注入 `workspace / guard / config / console / session`，替代全局变量，便于单测。

### 5.5 历史 `agent/history.py`

```python
class History:
    add_user(content) / add_assistant(msg) / add_tool_result(call_id, name, content, style) / add_note(text)
    payload() -> list[dict]                    # 发给模型
    tokens -> int                              # token 估算（tiktoken 优先，否则启发式）
    needs_compaction(budget, threshold) -> bool
    compact(llm, keep_recent=8, budget=None) -> bool   # 摘要失败则硬截断兜底
    to_jsonl() -> str

estimate_tokens(text) -> int; count_messages_tokens(messages) -> int
```

### 5.6 主循环 `agent/loop.py`

```python
class AgentLoop:
    def __init__(self, config, profile, backend, registry, console=None, *, system_prompt=None, on_event=None)
    def run(self, task: str, extra_context: dict | None = None) -> RunResult

@dataclass
class RunResult:
    answer, finish_reason, steps, tool_calls, errors, changes, usage, elapsed, compacted, error_message
    succeeded -> bool     # finish_reason ∈ {finish, model_final}
    stats_line() -> str
```

---

## 6. 错误处理策略

按"错误发生在哪一层"分层处理，不同层策略完全不同：

| 层 | 错误 | 检测方式 | 处理策略 |
|---|---|---|---|
| 配置 | 缺 API key、workspace 不存在、档位不存在 | `load_config` 校验 | **立即终止**并给出可操作提示（这类错误模型救不回来） |
| 模型 | 限流 429 / 超时 / 连接失败 / 5xx | `LLMError.retryable` | **指数退避 + 抖动重试**（默认 3 次，上限 30s/次）；401/400 立即失败，不浪费时间 |
| 解析 | JSON 非法、工具名不存在、参数缺失/类型错、单轮调用过多 | `ParseError` / `ParseOutcome.issues` | **把错误原文注入为 system note，让模型下一轮自纠**；超过 `max_parse_retries` 才放弃 |
| 工具 | 文件不存在、越界、命令非零退出、超时 | `ToolResult.ok=False` | **绝不抛出**；转成带 `[TOOL ERROR]` 前缀的回执回灌模型；连续失败 ≥ `max_consecutive_errors` 停循环 |
| 安全 | 路径越界、危险命令 | `SecurityError` | 拒绝执行并回执"原因 + 替代建议"；`command_policy=confirm` 时先问用户 |
| 预算 | 上下文超限、步数用尽 | token 估算 / 步数计数 | 先**压缩上下文**，仍超则优雅终止并汇报已完成部分 |
| 停滞 | 连续 3 次相同调用指纹 | `deque(maxlen=6)` | 注入"换思路"提示；再犯则终止（`no_progress`） |
| 中断 | Ctrl-C | `KeyboardInterrupt` | 杀掉子进程树、保留历史，允许继续追问 |

三条贯穿始终的原则：

1. **错误要变成信息，而不是崩溃**——工具层的任何异常都转成模型可读的回执；
2. **可恢复的错误给模型一次自纠机会，不可恢复的错误立刻停**——避免"重试 10 次 401"这种浪费；
3. **兜底永远存在**——摘要失败退化为硬截断，tiktoken 缺失退化为启发式估算，SDK 缺失退化为 requests。

---

## 7. 配置管理

### 7.1 三层优先级

```
命令行参数  >  环境变量  >  config.yaml / config.json  >  内置默认值
```

### 7.2 凭据处理（安全红线）

- `config.yaml` 存放端点与模型信息，**已在 `.gitignore` 中，绝不入库**；
- API key 推荐用占位符从环境变量读取，文件里不出现明文：

```yaml
profiles:
  default:
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"     # 也支持 $VAR / env:VAR
    model: "gpt-4o-mini"
```

- 仓库中只提交 `config.example.yaml`（无真实凭据）；
- 环境变量支持 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`，
  以及优先级更高的 `AGENT_API_KEY` / `AGENT_BASE_URL` / `AGENT_MODEL` / `AGENT_TEMPERATURE` /
  `AGENT_PROFILE` / `AGENT_WORKSPACE` / `AGENT_CONFIG`；
- 输出回执会做**密钥脱敏**（`sk-***`、`api_key=***`、`Bearer ***`），
  工具读取 `config.yaml` / `.env` / `*.pem` 一类文件会被路径策略标记；
- 打印配置时统一走 `LLMProfile.masked()`，密钥只显示前后各几位。

### 7.3 多档位切换

```bash
python run.py --list-profiles
python run.py --profile deepseek -t "重构这个函数"
AGENT_MODEL=qwen2.5-coder:7b python run.py -t "…"      # 临时覆盖
```

---

## 8. 关键参数默认值

| 参数 | 默认值 | 含义 |
|---|---|---|
| `max_steps` | 60 | 单任务最大工具轮次 |
| `max_consecutive_errors` | 4 | 连续工具失败上限 |
| `max_parse_retries` | 2 | 单次响应的解析纠错次数 |
| `command_timeout` | 120s | 单条命令超时（超时杀进程树并返回部分输出） |
| `max_tool_output_chars` | 12,000 | 回执上限，head+tail 双端截断 |
| `max_context_tokens` | 96,000 | 上下文预算 |
| `compact_threshold` | 0.75 | 占用预算比例超此值触发压缩 |
| `compact_keep_recent` | 8 | 压缩时保留的最近消息数 |
| `command_policy` | confirm | allow / confirm / deny |
| `restrict_to_workspace` | true | 路径沙箱开关 |

---

## 9. 验证方式

```bash
python tests/test_smoke.py         # 10 个用例：双通道、沙箱、危险命令、压缩、停滞、配置
python tests/test_fake_server.py   # 本地假服务端，校验真实 HTTP 协议链路
python run.py --mock -w ./demo -t "演示"     # 离线跑通完整循环，不联网
```

---

## 10. 面试可能问到的问题（准备要点）

1. **为什么不用框架？** 框架的循环、解析、上下文策略都是黑盒；本题要求自行实现。
   自己写之后，每个决策点（何时压缩、何时终止、错误怎么回灌）都可解释、可调。
2. **工具结果为什么不直接抛异常？** "工具报错"是给模型最有价值的信号。
   抛异常会终止整个任务，回执则给模型一次自我修正的机会。
3. **怎么防止无限循环？** 四重：显式 `finish`、步数上限、连续失败计数、调用指纹去重。
4. **上下文超限怎么办？** 回执先截断（head+tail 保留报错头与结论尾），
   仍超阈值则让模型把中段历史压成结构化摘要；摘要失败退化为硬截断，保证永不超窗报错。
5. **模型不支持 function calling 怎么办？** 自动切文本协议；解析器宽容但严格校验，
   解析失败时把错误原文回灌让模型自纠，而不是崩掉。
6. **安全性如何保证？** 路径沙箱（所有路径 resolve 后必须在 workspace 内）、
   危险命令黑名单（deny/confirm 两档）、命令超时杀进程树、写文件前自动备份、密钥脱敏。
7. **哪里最想继续改进？** ① 更长任务的跨会话记忆（把摘要落盘）；
   ② 并行工具调用（当前是顺序执行，对独立调用可以并发）；
   ③ 结构化 diff 编辑（用 apply_patch 替代整文件覆盖，省 token）；
   ④ 沙箱升级（命令在容器/受限用户下执行）。
