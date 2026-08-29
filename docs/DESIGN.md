# MiniCode 设计文档（答辩小抄）

> 一份"为什么这么做"的设计依据。代码本身讲清楚「做了什么」，这份文档讲清楚「为什么」——
> 评委的追问几乎都落在后者。

---

## 1. 项目定位

MiniCode 是一个**不依赖任何 Agent 框架**的编程智能体（coding agent）。

- 只用 OpenAI 兼容的聊天补全接口（一行 `chat.completions.create` 级别的客户端），
  其余全部自己写：对话历史与上下文管理、工具定义与本地执行、模型输出解析、循环终止条件、
  错误处理、上下文压缩、自修复闭环。
- 约束：仅调用现有 LLM API + Prompt 工程，不做任何训练/微调。
- 实测端点：国家超算长沙中心 MaaS 的 **Qwen3.5**（OpenAI 兼容，支持原生 function calling）。

**为什么强调"零框架"**：框架把"循环、工具、记忆"封装成了黑盒，本项目要证明的是
*这些能力自己也能可靠地实现*——这才是考核的含金量所在，也最容易在答辩里讲出深度。

---

## 2. 整体架构

```
run.py / cli.py        入口：参数解析 + 交互式 REPL（/stats、/undo、/compact…）
   │
   ▼
agent/loop.py          主循环（编排者）：调模型→解析→执行工具→回灌→判定终止
   │
   ├── agent/llm.py      LLM 接入层（OpenAI SDK / 原生 HTTP / Mock 三种后端）
   ├── agent/history.py  对话历史（OpenAI 格式）+ token 计数 + 自动压缩
   ├── agent/parser.py   模型输出解析（原生 tool_calls / 文本协议 ```json）
   ├── agent/tools/      工具系统：base（注册表/回执）+ filesystem/search/shell/meta/repair/review
   ├── agent/profile.py    项目画像（识别语言/框架/构建与测试命令，注入 system prompt）
   ├── agent/selfrepair.py  自修复「感知」层（纯函数，可单测）
   ├── agent/security.py 安全层：路径沙箱 / 危险命令 / 输出净化 / 智能压缩
   └── agent/config.py   配置（三层优先级 + 多档位 + 密钥轮换）
```

**主循环状态机**（termination 有四类触发）：
1. 模型主动 `finish`（最理想）；
2. 无工具调用的收尾回答（`FINAL:` / 文本协议）；
3. 配额触发（步数 / 上下文 / 连续失败 / 重复调用）；
4. 用户 Ctrl-C 中断（保留历史）。

### 2.1 架构总览

<svg viewBox="0 0 680 380" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">
  <rect width="680" height="380" fill="#F5F7FA"/>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1A1A1A">架构总览：主循环是中枢，其余皆为可替换 / 可单测的模块</text>
  <rect x="250" y="40" width="180" height="40" rx="6" fill="#FFFFFF" stroke="#2F6FED" stroke-width="1.5"/>
  <text x="340" y="65" font-size="13" text-anchor="middle" fill="#1A1A1A">run.py / cli.py（入口 + REPL）</text>
  <rect x="250" y="168" width="180" height="56" rx="8" fill="#2F6FED" stroke="#1E4FAE" stroke-width="1.5"/>
  <text x="340" y="192" font-size="14" font-weight="700" text-anchor="middle" fill="#FFFFFF">AgentLoop</text>
  <text x="340" y="210" font-size="11" text-anchor="middle" fill="#EAF1FF">主循环 · 编排者</text>
  <g font-size="11" fill="#1A1A1A" text-anchor="middle">
    <rect x="20" y="100" width="172" height="40" rx="6" fill="#FFFFFF" stroke="#9AA5B1"/>
    <text x="106" y="118">agent/llm.py</text><text x="106" y="133">LLM 接入层</text>
    <rect x="20" y="232" width="172" height="40" rx="6" fill="#FFFFFF" stroke="#9AA5B1"/>
    <text x="106" y="250">agent/history.py</text><text x="106" y="265">历史 · token · 压缩</text>
    <rect x="488" y="100" width="172" height="40" rx="6" fill="#FFFFFF" stroke="#9AA5B1"/>
    <text x="574" y="118">agent/parser.py</text><text x="574" y="133">输出解析</text>
    <rect x="488" y="232" width="172" height="40" rx="6" fill="#FFFFFF" stroke="#9AA5B1"/>
    <text x="574" y="250">agent/tools/</text><text x="574" y="265">工具系统</text>
    <rect x="254" y="300" width="172" height="40" rx="6" fill="#FFFFFF" stroke="#9AA5B1"/>
    <text x="340" y="318">agent/profile.py</text><text x="340" y="333">项目画像</text>
    <rect x="20" y="168" width="172" height="40" rx="6" fill="#FFFFFF" stroke="#9AA5B1"/>
    <text x="106" y="186">agent/security.py</text><text x="106" y="201">安全层</text>
    <rect x="488" y="168" width="172" height="40" rx="6" fill="#FFFFFF" stroke="#9AA5B1"/>
    <text x="574" y="186">agent/selfrepair.py</text><text x="574" y="201">自修复感知</text>
  </g>
  <g stroke="#6B7280" stroke-width="1.2" fill="none">
    <line x1="340" y1="80" x2="340" y2="168"/>
    <line x1="192" y1="120" x2="250" y2="178"/>
    <line x1="192" y1="252" x2="250" y2="212"/>
    <line x1="488" y1="120" x2="430" y2="178"/>
    <line x1="488" y1="252" x2="430" y2="212"/>
    <line x1="340" y1="224" x2="340" y2="300"/>
    <line x1="192" y1="188" x2="250" y2="188"/>
    <line x1="488" y1="188" x2="430" y2="188"/>
  </g>
</svg>

### 2.2 主循环时序（一轮的核心流转）

<svg viewBox="0 0 680 440" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">
  <rect width="680" height="440" fill="#F5F7FA"/>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1A1A1A">主循环时序：一轮的核心流转与四类终止</text>
  <g font-size="12" fill="#1A1A1A" text-anchor="middle">
    <rect x="220" y="44" width="240" height="44" rx="8" fill="#FFFFFF" stroke="#2F6FED" stroke-width="1.5"/>
    <text x="340" y="71">预算检查 / 上下文压缩</text>
    <rect x="220" y="118" width="240" height="44" rx="8" fill="#FFFFFF" stroke="#2F6FED" stroke-width="1.5"/>
    <text x="340" y="145">调模型 backend.chat（重试 / 流式）</text>
    <rect x="220" y="192" width="240" height="44" rx="8" fill="#FFFFFF" stroke="#2F6FED" stroke-width="1.5"/>
    <text x="340" y="219">解析 parser.parse（双通道）</text>
    <rect x="220" y="384" width="240" height="44" rx="8" fill="#FFFFFF" stroke="#2F6FED" stroke-width="1.5"/>
    <text x="340" y="405">执行工具 registry.execute</text>
    <text x="340" y="424" font-size="10" fill="#6B7280">（只读并行 · 写 / 命令顺序）</text>
    <rect x="520" y="278" width="140" height="50" rx="8" fill="#FCE8E6" stroke="#C0392B" stroke-width="1.5"/>
    <text x="590" y="298">收尾 / 结束</text>
    <text x="590" y="316" font-size="10" fill="#C0392B">finish·无调用·配额·Ctrl-C</text>
  </g>
  <polygon points="340,256 460,300 340,344 220,300" fill="#FFFFFF" stroke="#9AA5B1" stroke-width="1.5"/>
  <text x="340" y="305" font-size="12" text-anchor="middle" fill="#1A1A1A">有工具</text>
  <text x="340" y="322" font-size="12" text-anchor="middle" fill="#1A1A1A">调用?</text>
  <g stroke="#6B7280" stroke-width="1.2" fill="none">
    <line x1="340" y1="88" x2="340" y2="118"/>
    <line x1="340" y1="162" x2="340" y2="192"/>
    <line x1="340" y1="236" x2="340" y2="256"/>
    <line x1="340" y1="344" x2="340" y2="384"/>
    <line x1="460" y1="300" x2="520" y2="300"/>
    <line x1="220" y1="406" x2="200" y2="406" stroke-dasharray="4 3"/>
    <line x1="200" y1="406" x2="200" y2="66" stroke-dasharray="4 3"/>
    <line x1="200" y1="66" x2="220" y2="66"/>
  </g>
  <text x="472" y="292" font-size="10" fill="#6B7280">否</text>
  <text x="150" y="240" font-size="10" fill="#6B7280" text-anchor="middle">是 → 下一轮</text>
</svg>

### 2.3 上下文治理（三层逐级降档）

<svg viewBox="0 0 680 280" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">
  <rect width="680" height="280" fill="#F5F7FA"/>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1A1A1A">上下文治理：三层逐级降档，超预算才往下走</text>
  <g font-size="12" fill="#1A1A1A">
    <rect x="40" y="56" width="540" height="46" rx="8" fill="#FFFFFF" stroke="#2F6FED" stroke-width="1.5"/>
    <text x="60" y="78">① 估算 token（tiktoken 精确）—— 发请求前判断预算，早压缩优于超窗</text>
    <text x="60" y="95" font-size="10" fill="#6B7280">count_messages_tokens 遍历消息体；真实消耗另取 API usage</text>
    <rect x="40" y="126" width="540" height="46" rx="8" fill="#FFFFFF" stroke="#2F6FED" stroke-width="1.5"/>
    <text x="60" y="148">② 回执压缩（信号行优先）—— 工具输出写回前截断，保留 traceback / 错误</text>
    <text x="60" y="165" font-size="10" fill="#6B7280">smart_compress：信号行 &gt; 头 &gt; 尾，预算不够也不丢关键错误</text>
    <rect x="40" y="196" width="540" height="46" rx="8" fill="#FFFFFF" stroke="#2F6FED" stroke-width="1.5"/>
    <text x="60" y="218">③ 历史摘要压缩 —— 超阈值时中段交模型摘要；摘要失败硬截断兜底</text>
    <text x="60" y="235" font-size="10" fill="#6B7280">system 始终 index 0 不动；宁可丢信息也不让请求超窗报错</text>
  </g>
  <g stroke="#6B7280" stroke-width="1.2" fill="none">
    <line x1="300" y1="102" x2="300" y2="126"/>
    <line x1="300" y1="172" x2="300" y2="196"/>
  </g>
  <text x="312" y="118" font-size="10" fill="#6B7280">超预算</text>
  <text x="312" y="188" font-size="10" fill="#6B7280">仍超</text>
</svg>

---

## 3. 关键设计决策

### 3.1 双通道工具调用
优先用模型**原生 function calling**；当模型或网关不支持时，自动切换到
` ```json ` 文本协议，由自写解析器完成抽取、参数校验与类型纠错。
- 价值：同一套工具定义（`ToolSpec` 的 JSON Schema）两种通道通吃，模型兼容性最大化。
- 风险点：文本协议下模型常把行号、注释一起抄进参数 → 解析层专门做容错（见 edit_block 剥离行号）。

### 3.2 工具注册表模式（ToolSpec + ToolRegistry + ToolResult）
- 工具以 JSON Schema 声明参数；新增工具**不改主循环**，只在模块里 `@tool_spec` 再 `register`。
- `ToolResult` 统一封装：正常输出 + 错误，且**渲染时做截断/脱敏/压缩**。
- 铁律：**工具执行永不向主循环抛异常**。`registry.execute` 捕获一切，转成 `ok=False` 的回执——
  因为"工具报错"本身是对模型极有价值的信号，比崩溃有用得多。

### 3.3 上下文治理（三层）
1. **token 计数**：`tiktoken(cl100k_base)` 优先（精确），未装回落偏保守启发式；
2. **工具回执压缩**：`smart_compress` 信号行（error/traceback/exception…）优先，预算兜底时信号行仍存活；
3. **历史自动压缩**：超阈值时把中间历史交给模型做摘要（失败则硬截断兜底），system 提示词永远不动。

关键认知：**"发送前"的预算用估算即可，"成本对账"用 API 返回的 `usage`**。两者分开，既不会超窗，也能报出可信数字（见 `/stats`）。

### 3.4 安全与可回滚
- 路径沙箱：所有文件操作先 `PathGuard.resolve` 校验落在 workspace 内；
- 危险命令黑名单：deny / confirm 两档；
- 命令执行：中文 GBK 正确解码（OEM 代码页，非 `locale`）、交互式挂死检测（os.read 流式读 + 超时杀进程树）、输出密钥脱敏；
- 覆盖写前自动备份到 `.agent_backups/.overwrites/`，供 `/undo` 回滚；每次任务结束把整目录快照归档为「第 N 次」。

### 3.5 假完成拦截（V1）
模型改过文件却没跑过任何验证命令就 `finish` → 回灌提示逼它先验证（最多拦 2 次）。
根因：录演示视频最怕"写完即收尾"的半成品。

### 3.6 把模型输出当不可信输入
模型可能传出一个越界路径、一条危险命令、或一个荒谬的 timeout（如 99999 秒）。
三者统一按"边界防御"处理：路径走 `PathGuard` 沙箱、命令走黑白名单、数值走区间夹取。
**不要因为"模型通常是对的"就省掉校验**——一次越界/挂死就会毁掉整场演示。

### 3.7 流式输出的三个坑
1. **tool_calls 是按 index 分片的**：一个函数调用的 `id`/`name`/`arguments` 会跨多个 chunk
   到达，必须按 index 累积后再解析，否则 `arguments` 是残缺 JSON。
   流式与非流式共用同一段解析（`_tool_calls_from_raw`），保证两条路径行为一致。
2. **流式默认不带 usage**，而 `/stats` 的成本对账依赖 API 返回的 usage——
   不报错、数字只是悄悄变成 0，是最阴的一类故障。所以要显式传
   `stream_options={"include_usage": True}`。
3. **体验优化不能变成可用性风险**：网关不支持流式时自动退回整包，并记下不再重复试错。
   同理，UI 上流式与 spinner 只能二选一（spinner 靠 `\r` 覆盖同一行，会和正文互相擦除）。

### 3.8 工具反向文档与"双通道一致性"
给每个工具写清「什么时候不该用它」。关键在**一致性**：任何注入给模型的信息，
都必须同时覆盖 native function calling 与文本协议两条通道，否则默认配置下等于没写
（见 `ToolSpec.effective_description()`）。这也是本项目里反复出现的一类坑：
**新能力只加在一条通道上，另一条悄悄失效**。

---

## 4. 逐版本设计依据

> 完整改动/实测/测试记录见 `docs/ITERATION-PLAN.md`，这里只讲"为什么"和"坑在哪"。

### V0 — edit_block 精确编辑
`write_file` 是整体覆盖语义，改 473 行 HTML 里一处也得重写整个文件，极易因输出过长被截断甚至丢参数。
**结论**：局部修改必须有独立工具（`edit_block`），一个工具只保一种写语义，不要两副面孔。

### V0.5 — 多密钥自动轮换
长任务单 key 额度耗尽即中断。修了一处**致命 bug**：`OpenAIBackend._rebuild_client()` 注释说要重建客户端却空实现，
导致轮换后 SDK 仍用旧 key。教训：注释说了该做什么、代码却没做，单元测试必须覆盖"内部状态"（断言 `api_key` 跟着变）。

### V1 — 可靠性加固（Windows 必现三坑）
1. **中文乱码**：子进程默认 GBK，不能假设 UTF-8；用 OEM 代码页（`GetOEMCP`）而非 `locale`（开了 UTF-8 beta 的机器会误判）。
2. **交互式挂死**：`BufferedReader.read(n)` 是"凑满 n 或 EOF 才返回"，进程活着时读不到部分输出 → 改用 `os.read` 返回可用字节。
3. **假完成拦截**（见上）。

### V2 — 自修复闭环
从"能写代码"升级到"能交付正确的代码"。关键设计：
- **感知与执行分离**：`selfrepair` 只做纯函数分析（测试命令识别 / traceback 定位并读出附近源码），
  真正的"修"仍交给模型，避免把修复逻辑硬编码成脆弱规则。
- **回滚是免费的**：既然每次生成都归档完整快照，`rollback` 只是把快照拷回，从既有规范自然长出。

### V3 — 上下文与成本治理
评委必问"长任务怎么不超窗、花了多少"。交付：精确 token（tiktoken）+ 智能压缩（信号行优先）+ `/stats` 真实成本面板。
修了一处真实 bug：`ToolResult.render` 调用 `smart_compress` 却没 import，任何超长回执都 `NameError`。

### V4 — 可审阅性
`record_change` 采集改动前后内容 → `build_diff` 渲染 unified diff → 既是用户的 `/diff` 命令，
也是模型可调用的 `diff` 工具（finish 前自查）；`rollback(files=[...])` 做单文件级回退。

**实跑才暴露的两个缺陷**（单测全绿，但演示会翻车——说明"实跑验证"这步不能省）：
1. 所有**新建文件**的 diff 全丢了。根因：`record_change` 用 `before is not None and after is not None`
   反推"能否生成 diff"，但新建文件的 `before` 本来就该是 None（文件原本不存在）。
2. `run_command`/`rollback` 这些**操作**被当成**文件改动**渲染，在 diff 正文里刷噪声。

**教训（面试可讲）**：`None` 同时承担了"文件原本不存在"和"内容没采集"两种含义，
一个 `is not None` 判断不了两件事。修法是引入显式的 `captured` 标记把语义分开。

### V5 — 自主性增强
- **项目画像**（`profile.py`）：切入任务目录时扫描语言/框架/构建与测试命令，注入 system prompt 的
  「# 项目画像」段落——让模型"先看清项目再动手"，而不是盲猜该跑什么命令。
- **plan 工具**：复杂任务先列分步计划写入 session，供用户审阅、也帮模型对齐目标。
  触发用启发式（描述够长或含"实现/重构/搭建"等词），不是所有任务都强制。
- **只读工具并行**：一轮里的 `read_file`/`list_dir`/`grep_search`/`find_files`/`diff` 用线程池并发，
  回执按原顺序写回；有副作用的写与命令仍严格顺序执行。

**设计要点**：并行只放"只读"工具。写操作与命令一旦并行，顺序不确定会让结果不可复现，
也会让"回滚到哪一步"变得说不清——并发的收益远不抵确定性丢失。

### 追加工作（V4/V5 之后的打磨）

- **工具反向文档** `when_not_to_use`：工具描述普遍只写"能干什么"，但 agent 出错往往出在
  "该用 A 却用了 B"。关键决策是**必须让模型在两条通道上都看到**——默认走 native function calling，
  模型读的是 `openai_schema` 的 description，只写进系统提示词等于没写，故统一走
  `ToolSpec.effective_description()`。代价明算：系统提示词 +1259 字符（约 +1k tokens），
  换少一次工具误用（一次误用 = 一整轮重试 + 一次 API 往返），划算；并保留 `with_guardrail=False` 退路。
- **命令 timeout 上限夹取**：原先直接用模型传的值、无上限，传 99999 会把会话挂死几小时。
  现在夹到 [1, 300s]。本质是"把模型的输出当不可信输入处理"，与路径沙箱同类。
- **明确不做命令自动重试**：shell 命令可能有副作用（装依赖/写文件），重试 ≠ 重放，
  重复执行会造成重复副作用。所以只把诊断与建议讲清楚，是否重试交给模型。
  ——"没做的东西"也是设计，写进注释免得以后有人来加。
- **流式输出**：原先 `chat()` 的 `on_delta` 只是最后回调一次整段、config 里的 `stream`
  也从未被读取——两处"声明了没实现"。实现要点见 3.7。
- **测试命令识别多语言**：补 Java（Maven/Gradle），wrapper 调用分平台
  （Windows 走 cmd.exe，`./mvnw` 跑不通，必须用 `mvnw.cmd`）；
  并修了探测链在缺 `[tool.pytest]` 的 `pyproject.toml` 上 `return None`（应为 `continue`）的提前终止 bug。

---

## 5. 面试追问应答（备用）

- **"为什么不用 LangChain？"** 框架封装了循环/工具/记忆，本项目要证明这些能力自己也能可靠实现；
  且自写让我们对每次模型调用、每条工具回执、每轮压缩都有完全可控的可见性，便于排错与答辩。
- **"上下文超限怎么办？"** 三层治理（见 3.3）：估算→回执压缩→历史摘要压缩，且摘要失败有硬截断兜底，宁可丢信息也不让请求超窗报错。
- **"工具报错会崩吗？"** 不会。`registry.execute` 兜底成 `ok=False` 回执，错误本身就是给模型的信号。
- **"长任务中途 key 额度耗尽？"** V0.5 多密钥轮换，按 `401/403/429/quota` 自动切下一把，用完即止不绕回。
- **"模型改完不验证就收尾？"** V1 假完成拦截，回灌提示逼它先 run_command 验证。
- **"跑挂了怎么定位？"** V2 自修复：识别测试命令、解析 traceback 并把出错行附近源码直接回灌，省掉模型多一轮 read_file；修不好可 rollback。
- **"成本如何？"** V3 `/stats`：真实 token（API usage）、耗时、各工具耗时占比一目了然。
- **"你怎么知道它改了什么？"** V4 可审阅性：`/diff` 出 unified diff，模型自己也能调 `diff`
  在 finish 前自查；改坏了可以 `rollback(files=[...])` 只回退那一个文件。
- **"模型乱用工具怎么办？"** 三层：① 每个工具写清"不该用它"的场景（且两条通道都注入）；
  ② 工具回执里给出可行动的下一步，而不只是报错（如超时时会建议缩小范围并给出 timeout 上限）；
  ③ 边界防御（3.6）。
- **"换个语言的项目它还行吗？"** 会先做项目画像（`profile.py`）识别语言/框架/测试命令，
  测试命令识别覆盖 Python/Go/Rust/Node/Java(Maven/Gradle)，构建工具优先用仓库自带 wrapper
  并区分平台（Windows 用 `mvnw.cmd`）。
- **"工具调用会不会互相打架？"** 有副作用的（写文件、跑命令）严格顺序执行；
  只有只读工具（read/grep/list/diff）才并发，且回执按原顺序写回，保证结果可复现。
- **"这么多次迭代，最有价值的教训是什么？"** 三个：① 单测全绿 ≠ 能演示——V4 的两个缺陷
  都是实跑才暴露的；② 一个值承担两种含义迟早出事（`None` 既是"文件不存在"又是"没采集"、
  `subprocess` 的 `read(n)` 既是"凑满"又是"读到即返回"）；
  ③ "声明了但没实现"是本项目反复出现的坑——`_rebuild_client` 空实现、
  `on_delta` 假流式、`stream` 配置无人读取、Makefile 目标不校验就建议。
  共同点是"写了名字和注释，没写行为"，靠**读注释之外的代码**才能发现。
- **"流式输出会影响成本统计吗？"** 会，如果不处理的话。默认流式响应不带 usage，
  `/stats` 的 token 会静默变 0；所以显式要 `stream_options={"include_usage": True}`（见 3.7）。

---

## 6. Prompt 工程要点（核心差异点）

本项目硬约束是「仅调用现有 LLM API + Prompt 工程，不做训练 / 微调」，因此 system prompt
就是产品本身——它直接决定模型会不会用对工具、会不会乱来、能不能自纠偏。设计上遵循四条原则：

1. **短而硬**：只写模型猜不到的信息（工具协议、边界、终止条件、项目画像）。
   「你是一个乐于助人的 AI」之类无信息量的话不写，既占 token 也无行为约束。
2. **可执行**：每条都是祈使句、带明确触发条件（"先读现状再改""改前必须 read_file"），
   而不是价值观宣导。
3. **协议与工具清单分离**：工具清单由 registry 动态生成（`describe()`），
   避免"提示词写了工具、代码没注册"的不一致；新增工具不改 prompt 也自动出现。
4. **双通道一致**：原生 function calling 与文本协议共用同一段正文，
   只替换"如何输出调用"那一节——同一份心智模型，两种落地方式。

本轮在原有基础上强化了三处（均有回归测试守护，见 `tests/test_smoke.py` 的
`test_system_prompt_mentions_clarify_parallel_and_decision_order`）：

- **任务含糊先澄清**：需求有歧义或缺关键约束（语言 / 框架 / 输入输出格式）时，
  先 `ask_user` 确认，不替用户拍板；但能用合理默认推进的，不无意义打断。
- **可并行的事一轮发完**：互相独立的多个只读操作（list_dir / read_file /
  grep_search / find_files）在同一轮一起发起，缩短等待（见 V5 并行执行）。
- **决策顺序**：拿不准时按"先读现状 → 最小改动 → 小步验证 → 卡住就求助
  （ask_user / rollback）"走，把"凭感觉宣布完成"提前堵死。

反模式（我们刻意避开）：
- 把系统提示词写成角色扮演剧本；
- 在提示词里硬编码工具名但代码未注册（一致性靠 registry 动态注入保证）；
- 一条规则同时承担两种含义（本项目反复踩坑，如 `None` 既是"文件不存在"又是"没采集"）。

---

## 7. 真实端点验证（v0.1.0 · NSCC MaaS Qwen3.5）

极小任务端到端跑通（REPL 单次任务，`--max-steps 5`，无需人工干预）：
`write_file(hello.py)` → `run_command(python hello.py 输出 hello)` → `finish`。
结果：**步数=3 · 工具调用=3 · 失败=0 · 耗时=4.8s · 结束原因=finish**。

完整轨迹（节选，已隐去密钥）：

```
[1/5] Qwen3.5 ▸ 我来创建 hello.py 文件并验证。
      ▶ write_file(path=hello.py, content=print("hello"))
      ✔ write_file · 0.00s
[2/5] … Qwen3.5
      ▶ run_command(command=python hello.py)
      ✔ run_command · 0.42s  →  $ python hello.py  →  hello  (exit 0)
[3/5] … Qwen3.5
      ▶ finish(summary=① 创建 hello.py … ② 运行验证输出 hello … ③ 无遗留问题)
```

说明：流式输出（正文逐字打印）、原生 function calling、假完成拦截（先验证再 finish）、
上下文预算提示均按设计生效；离线 68 测试全绿，真实端点单步任务亦通过。
