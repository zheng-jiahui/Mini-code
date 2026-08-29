姓名：郑佳慧
提交物：README.txt + 演示视频

一、Git 仓库地址
https://github.com/______/minicode-coding-agent
（公开仓库，保留完整提交历史；API 凭据一律经环境变量或未入库的 api_keys.yaml 提供，
 仓库内不含任何真实密钥）

二、如何运行
1) pip install -r requirements.txt
2) 填入密钥（二选一，推荐第二种）：
   · 配置文件：cp api_keys.example.yaml api_keys.yaml，把密钥填进去
   · 环境变量：export NSCC_API_KEY=sk-...（PowerShell: $env:NSCC_API_KEY="sk-..."）
   （api_keys.yaml 与 config.yaml 均已加入 .gitignore，不会入库）
3) 常用命令：
   python run.py                                 交互式，直接输入任务
   python run.py -p nscc -t "写一个快排并跑通测试"   单次任务
   python run.py --mock -t "演示"                 离线演示，不联网也能看完整流程
   python run.py --list-tools / --print-prompt    查看工具清单与系统提示词
4) 测试：python tests/test_smoke.py（10 个用例，无需 API key）
        python tests/test_fake_server.py（本地假服务端，验证 HTTP 协议链路）

三、特色功能
1. 零框架：未使用 LangChain / LlamaIndex / OpenAI Agents SDK / AutoGen / CrewAI 等
   任何 Agent 框架，仅用 OpenAI 兼容的聊天补全客户端；对话历史与上下文管理、工具
   定义与本地执行、模型输出解析、循环终止条件、错误处理、上下文压缩全部自行实现。
2. 双通道工具调用：优先用模型原生 function calling；当模型或网关不支持时，自动切换
   到 ```json 文本协议，由自写解析器完成抽取、校验与参数纠错。
3. 自建工具系统：ToolSpec + ToolRegistry + ToolResult，工具以 JSON Schema 声明，
   新增工具不改主循环。已实现 read_file / write_file / list_dir / run_command /
   grep_search / find_files / finish / ask_user。
4. 上下文管理：token 估算 → 工具回执 head+tail 截断 → 超阈值自动摘要压缩
   （摘要失败则硬截断兜底），保证长任务不超窗。
5. 循环控制：步数上限、连续失败上限、重复调用指纹去重、解析纠错回灌、
   Ctrl-C 中断保留历史。
6. 安全与可回滚：工作区路径沙箱、危险命令黑名单（deny/confirm 两档）、命令超时并
   杀进程树、覆盖写前自动备份、输出密钥脱敏。
7. 可观测可复盘：每步打印思考/调用/回执/耗时，会话自动落盘为 JSONL。
8. 任务隔离：每个任务在 workspace 下自动建立独立子目录（时间戳+任务摘要），
   该任务的代码、.agent_backups 备份、.agent_sessions 会话日志都归拢在同一目录内，
   任务之间互不干扰；同会话追问沿用目录，/new 才开新目录。

四、其它说明
- 项目结构、主循环流程图与伪代码、System Prompt 全文、工具输出格式规范、各模块
  接口定义、错误处理分层策略，均见仓库 docs/DESIGN.md。
- 运行环境：Python 3.10+，Windows / macOS / Linux 均可。实测端点为国家超算长沙中心
  MaaS 的 Qwen3.5（OpenAI 兼容接口，支持原生 function calling）。
- 视频演示任务：在空目录下创建 Python 脚本、运行验证，发现报错后自行定位修复，
  最后调用 finish 给出总结。
