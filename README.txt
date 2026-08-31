姓名：郑佳慧
仓库：https://github.com/______/minicode-coding-agent
（公开仓库，保留完整提交历史；密钥仅走环境变量或未入库的 api_keys.yaml，仓库不含任何真实密钥）

一、如何运行
1) pip install -r requirements.txt
2) 填密钥（二选一，推荐环境变量）：cp api_keys.example.yaml api_keys.yaml 后填入；
   或 export NSCC_API_KEY=sk-...（PowerShell: $env:NSCC_API_KEY="sk-..."）。
   api_keys.yaml 与 config.yaml 已在 .gitignore，不会入库。
3) 常用命令：
   python run.py                              交互式，直接输入任务
   python run.py -t "创建用户登录功能"        单次任务
   python run.py --mock -t "演示"             离线演示，不联网也能看完整流程
   python run.py --list-tools / --print-prompt  查看工具清单与系统提示词
4) 产物目录（workplace 与 .agent_backups 同级，均不入库）：
   workplace/任务名/                          该任务的最新代码
   .agent_backups/任务名_时间戳_第N次/        每次生成的完整归档，序号自动递增
   .overwrites/                              覆盖写单文件备份，供 /undo 回滚
5) 测试：python tests/test_smoke.py（123 个用例，无需 API key）
        python tests/test_fake_server.py（本地假服务端，验证 HTTP 协议链路）

二、特色功能
1. 零框架：未用任何 Agent 框架（LangChain/LlamaIndex/OpenAI Agents SDK/AutoGen/CrewAI），
   仅用 OpenAI 兼容聊天补全客户端；对话历史、工具、解析、循环、压缩、错误全部自写。
2. 双通道工具调用：优先原生 function calling；不支持时自动切换 ```json 文本协议 + 自写解析校验纠错。
3. 自建工具系统 ToolSpec+ToolRegistry+ToolResult，新工具不改主循环（共 31 个）：
   read_file / write_file / edit_block / apply_patch / read_many_files / replace_in_file /
   move_file / copy_file / delete(删前备份) / replace_in_files(跨文件) / list_dir / run_command(支持后台) / lint(检查) /
   grep_search(支持上下文) / find_files / web_fetch / diff / recall(相关文件检索) /
   git(受控提交/只读) / delegate(受控子智能体) / check_command(读后台输出) / kill_command(终止后台) /
   plan / finish / ask_user / rollback / todo(任务清单) /
   think(推理便签) / memory(项目记忆) / self_improve(自我改进) / summary(会话概览)。
4. 上下文治理：token 计数 + 回执智能压缩 + 超阈值摘要压缩兜底。
5. 循环控制：步数/连续失败上限、重复调用指纹去重、解析纠错回灌、Ctrl-C 保留历史。
6. 自修复闭环：识别测试命令、定位 traceback 回灌源码、连续失败预算提醒、一键回滚。
7. 可靠性：中文 GBK 正确解码、交互挂死检测、假完成拦截（改文件须先验证）。
8. 安全可回滚：路径沙箱、危险命令黑名单(deny/confirm)、超时杀进程树、覆盖写自动备份。
9. 可观测：流式输出、每步打印思考/调用/回执/耗时；收尾报告卡汇总步数/工具调用/自修复/Token/耗时与记忆状态，首屏横幅亮出零框架合规与已加载经验条数；会话存 JSONL。
10. 可审阅性：/diff 看本次会话 unified diff；rollback 支持单文件回退。
11. 自主性增强：自动识别项目画像注入提示词；plan 复杂任务先列计划再执行；一轮内只读调用并行。
12. 改动落地更全：apply_patch 把标准 unified diff 落到已存在文件（多 hunk、对不上即整体不写盘），不依赖外部 patch/git；diff 配合先看改动再打补丁。
13. 贴近商业 code agent：权限系统/可干预/子任务编排等常规能力均已具备。
14. 跨会话项目记忆：memory 落盘 .minicode/memory.md 约定/踩坑/选型，启动注入提示词（类比 CLAUDE.md）。
15. 文件增删齐备：move_file/copy_file/delete 补足移动/复制/删除；delete 删前备份到
    .agent_backups 可恢复，删目录需 recursive=true。
16. 权限系统（贴近 Claude Code）：写/破坏性操作按 permission_mode 放行——
    auto 直接执行、ask 确认、read_only 只放行只读；REPL 内 /mode 切换。
17. 可干预：ask_user 信息不足时反问；finish 前改文件未验证会被拦下先验证；
    危险命令黑名单(deny/confirm) 与 permission_mode 双保险。
18. 子任务编排（贴近 Claude Code Task / Codex 子代理）：delegate 派发受控子智能体拆大任务为
    可验证小块、降父上下文压力；子智能体复用工作区与模型、独立步数预算，
    禁递归、不能反问、不污染父检查点，失败不拖垮父任务。
19. 后台命令：run_command 传 background=true 立即返回 job_id，用 check_command 读输出、
    kill_command 终止，可后台起服务/跑长任务（类比 Claude Code 后台 shell）。
20. 自我改进：任务结束把失败/修复信号沉淀成经验，落盘 .minicode/memory.md，形成"失败→记忆→更聪明"闭环（类比 Claude Code 记忆自更新）；agent 也可中途 digest/list/forget。

三、其它
- 架构图、System Prompt、工具规范、接口与错误分层见 docs/DESIGN.md。
- 环境 Python 3.10+，Windows/macOS/Linux 均可；实测端点国家超算长沙中心 MaaS 的 Qwen3.5（OpenAI 兼容，支持原生 function calling）。
- 演示任务：空目录建脚本→运行报错→自行定位修复→finish 总结。
