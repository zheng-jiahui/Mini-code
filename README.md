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
python run.py --resume -t "接着上次继续"         # 从上次检查点恢复后继续
python run.py --list-tools                     # 查看全部工具
```

**跑一次能力评测**（真实端点，约 40 秒）：

```bash
python -m agent.eval                           # 10 个标准任务，以产物验证为准
python -m agent.eval --task calc,dedup --json  # 只跑指定任务 / 落盘 JSON
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
│   ├── profile.py            # 项目画像（语言/框架/测试命令）+ 工作区状态扫描
│   ├── selfrepair.py         # 自修复感知层：测试命令识别、traceback 定位（纯函数）
│   ├── metrics.py            # 质量指标：结局分布 / 自修复回合 / 返工
│   ├── checkpoint.py         # 会话检查点：跨进程续跑
│   ├── eval.py               # 评测台：10 个标准任务，验证产物
│   └── tools/
│       ├── base.py           # ToolSpec / ToolRegistry / ToolResult（自建工具系统）
│       ├── filesystem.py     # read_file / write_file / edit_block / list_dir
│       ├── shell.py          # run_command
│       ├── search.py         # grep_search（支持 context 上下文）/ find_files
│       ├── recall.py         # recall：按相关度检索最相关文件与片段（模糊意图定位）
│       ├── patch.py          # apply_patch（自实现 unified diff 应用器，不依赖外部 patch/git）
│       ├── repair.py         # rollback（整目录 / 单文件级回退到历史快照）
│       ├── review.py         # build_diff：本次会话改动的 unified diff
│       ├── meta.py           # finish / ask_user / plan
│       ├── todo.py           # todo：可追踪状态的任务清单
│       ├── extra.py          # read_many_files / replace_in_file / web_fetch / think
│       ├── git_tool.py       # git：受控版本控制（白名单只读 + add + 受控 commit）
│       ├── fsops.py          # move_file / copy_file / delete（删前备份）
│       ├── memory.py         # memory：跨会话持久的项目记忆（.minicode/memory.md）
│       ├── replace_files.py  # replace_in_files：跨文件安全查找替换（仓库级重命名，默认 dry_run）
│       ├── lint.py           # lint：代码检查（零配置 Python 语法体检 + 结构化解析 linter 输出）
│       ├── summary.py        # summary：会话改动概览（可粘贴进 PR / 提交说明 / 交接）
│       └── agent.py          # delegate：派发受控子智能体（子任务编排，防递归、独立步数预算）
├── docs/DESIGN.md            # 完整设计说明（主循环流程图、接口定义、错误策略等）
└── tests/                    # 冒烟测试（Mock 后端，无需 API key）
```

---

## 3. 已实现能力

| 能力 | 说明 |
|---|---|
| 工具（共 28 个） | 文件：`read_file` `write_file` `edit_block` `list_dir` `read_many_files` `replace_in_file` `move_file` `copy_file` `delete` `replace_in_files`（跨文件替换）；检索：`grep_search`（支持 context 上下文）`find_files` `web_fetch` `recall`（相关文件检索）；执行：`run_command`；检查：`lint`（代码检查）；版本控制：`git`（受控提交 + 只读 + add）；编排：`delegate`（受控子智能体）；控制：`finish` `ask_user` `plan` `todo`（任务清单）`think`（推理便签）`memory`（项目记忆）；汇报：`summary`（会话改动概览） |
| 双通道调用 | 优先原生 `tool_calls`；模型不支持时自动切到 ```json 文本协议 |
| 上下文管理 | token 估算 → 工具回执智能压缩（信号行优先）→ 超阈值摘要压缩 → 硬截断兜底 |
| 长程记忆 | **常驻事实层**：硬约束/技术选型/已失败方案常驻且不参与再压缩，避免"摘要的摘要"式衰减；压缩后重建工作区真实清单 |
| 会话续跑 | 每个任务（含中断/报错）自动保存检查点，`--resume` 可跨进程接着上次继续；只存"发生过什么"，system 提示词等"当前状态"恢复时重建 |
| 自修复 | 运行失败自动定位 traceback 并附上出错位置源码、测试命令自动识别、修复预算、一键回滚 |
| 循环控制 | 步数上限、token 预算、连续错误上限、重复调用指纹去重、假完成拦截、用户 Ctrl-C 中断 |
| 安全 | 工作区路径沙箱、危险命令拦截/二次确认、命令超时（带上限夹取）、写前自动备份 |
| 权限系统 | 写/破坏性操作（write_file/edit_block/apply_patch/delete/move_file/copy_file/replace_in_files/run_command）按 `permission_mode` 放行：`auto` 直接执行（默认，无头/脚本不动）、`ask` 执行前交互确认、`read_only` 只放行只读工具（安全审查"只看不动"）；REPL 内 `/mode` 实时切换，拒绝不计入连续失败 |
| 可观测 | 每步打印「思考 / 工具调用 / 结果 / 耗时」，流式输出，`/diff` 审阅改动、`/stats` 成本与瓶颈面板，会话可存 JSONL 复盘 |
| 质量指标 | `/stats` 也回答"做对了没有"：结局分布、失败率、自修复回合数、返工 |
| 评测台 | `python -m agent.eval`：10 个标准任务，**以产物能否跑通为准**（不采信模型自述），并标出假完成 / 悲观失败 |
| 任务清单 | `todo` 维护可勾选的待办清单，长任务里防漏做 / 防过早收尾 |
| 批量读取 | `read_many_files` 一次看清多个相关文件，省掉多轮往返 |
| 全局替换 | `replace_in_file` 把所有同名旧符号统一换成新的（区别于 `edit_block` 的唯一匹配） |
| 网页抓取 | `web_fetch` 用标准库自实现抓取 http/https 页面并剥离 HTML，让 agent 能自己读文档 / RFC / API 说明 |
| 受控 git | `git` 白名单只读 + 暂存 + **受控提交**：commit 必须有提交信息、禁止改写历史选项（--amend / --no-verify / --allow-empty / --date / -a / --all）、禁止空暂存区提交；机制上拦截 push / reset --hard 等高危操作，避免误破坏仓库 |
| 推理便签 | `think` 把推理 / 计划固定进上下文，长任务保持思路、便利用户审阅思考过程 |
| 项目记忆 | `memory` 跨会话持久沉淀「项目约定 / 踩过的坑 / 已定选型」（落盘 `.minicode/memory.md`），启动时自动注入 system 提示词，下次会话免从零探索 |
| 文件增删 | `move_file` / `copy_file` / `delete` 补足文件的移动 / 复制 / 删除；`delete` 删除前先备份到 `.agent_backups` 可恢复，删目录需 `recursive=true` |
| 相关文件检索 | `recall` 给定一段模糊意图（如「登录失败重试」「导出 CSV」），按相关度排序找回最相关的若干文件并给出最相关的几行片段；模型不必知道确切函数名也能先定位（类比 Cursor/Claude Code 的 find relevant code），一轮内可与其他只读工具并行 |
| 跨文件替换 | `replace_in_files` 按 glob 圈定一批文件，把某段文本（或正则）全部替换（仓库级符号重命名）；默认 `dry_run=true` 仅预览会动哪些文件/几处，确认后再落盘；写前逐文件备份可 /undo 回滚，二进制文件自动跳过 |
| 子任务编排 | `delegate` 派发受控子智能体：大任务拆成可独立验证的小块、降低父任务上下文压力（类比 Claude Code 的 Task / Codex 子代理）。子智能体复用同一工作区与模型端点、拥有独立上下文与步数预算；**机制上杜绝递归**（剔除 `delegate`）、**不能反问用户**（剔除 `ask_user`）、关闭检查点/会话落盘写入避免污染父任务；其失败不拖垮父任务 |
| 代码检查 | `lint` 写完代码、finish 前先自查：不传 command 时对 Python 文件做零配置语法体检（内置 compile，纯标准库），传 command 时运行你给的检查器（如 `ruff check .`）并结构化解析 `文件:行: 信息`；只读护栏禁止 `--fix` 等改写选项，可据此逐条修复 |
| 会话概览 | `summary` 把本次会话的改动整理成可粘贴的概览：哪些文件新建/修改/删除、各加减多少行、本次任务是什么，适合直接贴进 PR 描述 / 提交说明 / 交接；与 /diff（逐行 diff）、/stats（质量指标）三者互补 |

---

## 4. 设计文档

完整的 7 项设计（文件结构、主循环伪代码与流程图、System Prompt、工具调用输出格式规范、模块接口定义、错误处理策略、配置管理）见 [`docs/DESIGN.md`](docs/DESIGN.md)。
