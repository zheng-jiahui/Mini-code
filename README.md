# MiniCode —— 从零实现的编程智能体（Coding Agent）

不依赖 **任何 Agent 框架 / SDK**（LangChain、LlamaIndex、OpenAI Agents SDK、Claude Agent SDK、AutoGen、CrewAI 一律未使用）。
仅使用 OpenAI 兼容的 **聊天补全 API 客户端**；对话历史管理、工具定义与本地执行、模型输出解析、循环终止、错误处理、上下文压缩，全部自行实现。

---

## 1. 快速开始

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 配置 API：复制模板，填入自己的凭据（config.yaml 已在 .gitignore 中，不会入库）
cp config.example.yaml config.yaml
#   推荐做法：把密钥留在环境变量里，config.yaml 只写占位符 ${OPENAI_API_KEY}
export OPENAI_API_KEY="sk-xxxxxx"

# 3) 运行
python run.py                                  # 交互式 REPL
python run.py -t "写一个快排并跑通测试"          # 单次任务
python run.py --workspace ./demo -t "..."      # 指定工作目录（沙箱根）
python run.py --mock -t "演示一次工具调用"        # 离线演示：不发请求，用脚本化模型
python run.py --list-tools                     # 查看全部工具
```

**切换模型档位**（档位在 `config.yaml` 的 `profiles` 下配置，可同时配多个端点）：

```bash
python run.py --list-profiles                  # 列出全部档位（* 为当前生效）
python run.py --profile deepseek -t "..."      # 用指定档位跑一次
AGENT_MODEL=Qwen3.5 AGENT_BASE_URL=... python run.py   # 环境变量临时覆盖，优先级高于配置文件
```

> `AGENT_*` 系列环境变量优先级最高：`AGENT_API_KEY` / `AGENT_BASE_URL` / `AGENT_MODEL` /
> `AGENT_TEMPERATURE` / `AGENT_PROFILE` / `AGENT_WORKSPACE` / `AGENT_CONFIG`。

---

## 2. 文件结构

```
.
├── run.py                    # 命令行入口（REPL / 单次任务 / 工具自检）
├── config.example.yaml       # 配置模板（入库）
├── config.yaml               # 你的真实配置（不入库）
├── agent/
│   ├── config.py             # 配置加载：YAML/JSON + 环境变量覆盖 + 校验
│   ├── errors.py             # 错误类型分层
│   ├── llm.py                # LLM 后端：openai SDK / 原生 requests / 离线 Mock + 重试
│   ├── parser.py             # 模型输出解析（原生 tool_calls + 文本协议 fallback）
│   ├── history.py            # 对话历史、token 估算、工具输出截断、自动压缩
│   ├── prompts.py            # System Prompt 与各类提示词模板
│   ├── security.py           # 路径沙箱、危险命令识别、输出脱敏/截断
│   ├── ui.py                 # 终端渲染（事件流、彩色、确认交互）
│   ├── loop.py               # ★ 主循环：解析 → 执行 → 回灌 → 终止判定
│   ├── cli.py                # 参数解析与 REPL 命令
│   └── tools/
│       ├── base.py           # ToolSpec / ToolRegistry / ToolResult（自建工具系统）
│       ├── filesystem.py     # read_file / write_file / list_dir
│       ├── shell.py          # run_command
│       ├── search.py         # grep_search / find_files
│       └── meta.py           # finish / ask_user
├── docs/DESIGN.md            # 完整设计说明（主循环流程图、接口定义、错误策略等）
└── tests/                    # 冒烟测试（Mock 后端，无需 API key）
```

---

## 3. 已实现能力

| 能力 | 说明 |
|---|---|
| 工具 | `read_file` `write_file` `list_dir` `run_command` `grep_search` `find_files` `finish` `ask_user` |
| 双通道调用 | 优先原生 `tool_calls`；模型不支持时自动切到 ```json 文本协议 |
| 上下文管理 | token 估算 → 工具输出 head/tail 截断 → 超阈值 LLM 摘要压缩 |
| 循环控制 | 步数上限、token 预算、连续错误上限、重复调用指纹去重、用户 Ctrl-C 中断 |
| 安全 | 工作区路径沙箱、危险命令拦截/二次确认、命令超时、写前自动备份 |
| 可观测 | 每步打印「思考 / 工具调用 / 结果 / 耗时」，会话可存 JSONL 复盘 |

---

## 4. 设计文档

完整的 7 项设计（文件结构、主循环伪代码与流程图、System Prompt、工具调用输出格式规范、模块接口定义、错误处理策略、配置管理）见 [`docs/DESIGN.md`](docs/DESIGN.md)。
