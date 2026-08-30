# MiniCode —— 从零实现的编程智能体

不依赖任何 Agent 框架 / SDK（LangChain、LlamaIndex、OpenAI Agents SDK、Claude Agent SDK、AutoGen、CrewAI 均未使用）。仅用 OpenAI 兼容的聊天补全 API；对话历史、工具定义与本地执行、输出解析、循环终止、错误处理、上下文压缩全部自写。类似一个简化的 Claude Code / Codex。

## 快速开始

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml      # 填入凭据（config.yaml 已在 .gitignore，不入库）
export NSCC_API_KEY="sk-xxxx"            # hy3 端点密钥
python run.py -t "写一个快排并跑通测试"   # 单次任务
python run.py --mock -t "演示工具调用"    # 离线演示（不发请求）
python -m agent.eval                     # 能力评测（11 个任务，以产物验证为准）
```

切换模型：`python run.py --list-profiles` / `--profile deepseek`；`AGENT_MODEL` 等环境变量可临时覆盖。

## 已实现能力

- 文件：`read_file`（带行号）`write_file`（覆盖前备份）`edit_block`（精确替换）`list_dir`
- 执行：`run_command`（超时夹取、交互挂死检测、GBK 不乱码）
- 检索：`grep_search` `find_files` `fetch_url`（联网查文档，自写 urllib，仅 http/https）
- 版本控制：`git_init` `git_status` `git_diff` `git_log` `git_commit`（仅本地提交，绝不 push / 改写历史）
- 规划与闭环：`plan` `todo`（带 pending/in_progress/completed 进度）`finish` `ask_user` `rollback`（回退快照）`diff` `apply_patch` 自修复
- 工程化：原生 `tool_calls` 与文本协议双通道；token 预算与自动压缩；检查点续跑；危险命令拦截/二次确认；质量指标（结局/工具失败率/自修复回合）

## 设计要点

评测以「产物能否跑通」为准，标出假完成 / 悲观失败；`/stats` 给出真实成本与瓶颈。主循环、工具系统、上下文压缩、错误处理等全部自行实现，详见 `docs/DESIGN.md`。
