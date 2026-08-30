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
5) 测试：python tests/test_smoke.py（113 个用例，无需 API key）
        python tests/test_fake_server.py（本地假服务端，验证 HTTP 协议链路）

二、特色功能
1. 零框架：未使用 LangChain / LlamaIndex / OpenAI Agents SDK / AutoGen / CrewAI 等任何 Agent 框架，
   仅用 OpenAI 兼容聊天补全客户端；对话历史、工具、解析、循环、压缩、错误全部自写。
2. 双通道工具调用：优先原生 function calling；模型/网关不支持时自动切换 ```json 文本协议 + 自写解析校验纠错。
3. 自建工具系统 ToolSpec+ToolRegistry+ToolResult，新增工具不改主循环（共 25 个）：
   read_file / write_file / edit_block / apply_patch / read_many_files / replace_in_file /
   move_file / copy_file / delete(删前备份) / replace_in_files(跨文件替换) / list_dir / run_command /
   grep_search(支持上下文) / find_files / web_fetch / diff / recall(相关文件检索) /
   git(受控提交/只读) / plan / finish / ask_user / rollback / todo(任务清单) / think(推理便签) /
   memory(项目记忆)。
4. 上下文治理：tiktoken 精确 token 计数 + 工具回执智能压缩（信号行优先）+ 超阈值摘要压缩兜底；
   /stats 面板报真实 token、各工具耗时占比与时间去向（实测等模型约九成，瓶颈一目了然）。
5. 循环控制：步数/连续失败上限、重复调用指纹去重、解析纠错回灌、Ctrl-C 保留历史。
6. 自修复闭环：识别测试命令、定位 traceback 并回灌源码上下文、连续失败预算提醒、rollback 一键回滚。
7. 可靠性：中文 GBK 正确解码、交互式挂死检测、假完成拦截（改文件须先验证再 finish）。
8. 安全可回滚：路径沙箱、危险命令黑名单(deny/confirm)、命令超时杀进程树、覆盖写自动备份、输出密钥脱敏。
9. 可观测：流式输出（正文逐字实时打印，网关不支持时自动退回整包）、每步打印思考/调用/回执/耗时；
   会话落盘 JSONL。
10. 可审阅性：/diff 查看本次会话的 unified diff；rollback 支持单文件级回退（只恢复指定文件）。
11. 自主性增强：自动识别项目画像（语言/框架/构建/测试命令）注入提示词，让模型先看清新项目再动手；
   plan 工具支持复杂任务"先列计划再执行"；一轮内的多个只读调用（read/grep/list/diff）并行发出，缩短等待。
12. 改动落地更全：apply_patch 工具把标准 unified diff 落到已存在文件（多 hunk、上下文匹配容错、
   任一 hunk 对不上即整体失败不写盘），不依赖外部 patch/git 二进制；diff 工具与之配合可"先看改动再打补丁"。
13. 更贴近商业 code agent：todo 任务清单防漏做、read_many_files 批量读、replace_in_file/replace_in_files 单文件与跨文件替换、
   web_fetch 自读在线文档/RFC/API 说明、recall 按相关度定位文件（模糊意图不必知道函数名）、安全受控 git（可常规提交、禁改写历史选项）、think 推理便签；一轮内多个只读调用并行发出。
14. 跨会话项目记忆：memory 工具把「约定/踩坑/选型」落盘 .minicode/memory.md，启动时自动注入
    system 提示词（类比 CLAUDE.md/AGENTS.md），下次会话免从零探索，且可 append 持续积累。
15. 文件增删齐备：move_file/copy_file/delete 补足文件的移动/复制/删除；delete 删前先备份到
    .agent_backups 可恢复，删目录需 recursive=true，越界路径一律拒绝。

三、其它
- 架构图、System Prompt、工具规范、接口与错误分层见 docs/DESIGN.md。
- 环境 Python 3.10+，Windows/macOS/Linux 均可；实测端点国家超算长沙中心 MaaS 的 Qwen3.5（OpenAI 兼容，支持原生 function calling）。
- 演示任务：空目录建脚本→运行报错→自行定位修复→finish 总结。
