姓名：______
提交物：README.txt + 演示视频

一、Git 仓库地址
https://github.com/______/minicode-coding-agent
（公开仓库，保留完整提交历史；API 凭据通过环境变量提供，config.yaml 已加入 .gitignore，未入库）

二、如何运行
1) pip install -r requirements.txt
2) cp config.example.yaml config.yaml，填入自己的 base_url / model / api_key
   （推荐把密钥留在环境变量：export OPENAI_API_KEY=xxx，config.yaml 里写 ${OPENAI_API_KEY}）
3) python run.py                     进入交互式，直接输入任务
   python run.py -t "写一个快排并跑通单元测试"      单次任务
   python run.py --profile deepseek -t "..."        切换模型档位
   python run.py --mock -t "演示"                   离线演示，不联网也能看到完整流程
   python run.py --list-tools / --print-prompt      查看工具清单与系统提示词
4) 测试：python tests/test_smoke.py（10 个用例，无需 API key）
        python tests/test_fake_server.py（本地假服务端，验证 HTTP 协议链路）

三、特色功能
1. 零框架：未使用 LangChain / LlamaIndex / OpenAI Agents SDK / AutoGen / CrewAI 等任何 Agent 框架，
   仅用 OpenAI 兼容的聊天补全客户端；对话历史、工具定义与本地执行、输出解析、循环终止、
   错误处理、上下文压缩全部自行实现。
2. 双通道工具调用：优先使用模型原生 function calling；当模型或网关不支持时，
   自动切换到 ```json 文本协议，并由自写解析器完成抽取、校验与参数纠错。
3. 自建工具系统：ToolSpec + ToolRegistry + ToolResult，工具以 JSON Schema 声明，
   新增工具不改主循环。已实现 read_file / write_file / list_dir / run_command /
   grep_search / find_files / finish / ask_user。
4. 上下文管理：token 估算 → 工具回执 head+tail 截断 → 超阈值自动摘要压缩（摘要失败则硬截断兜底），
   保证长任务不超窗。
5. 循环控制：步数上限、连续失败上限、重复调用指纹去重、解析纠错回灌、Ctrl-C 中断保留历史。
6. 安全与可回滚：工作区路径沙箱、危险命令黑名单（deny/confirm 两档）、命令超时并杀进程树、
   覆盖写前自动备份（/undo 可回滚）、输出密钥脱敏。
7. 可观测与可复盘：每步打印思考/调用/回执/耗时，会话自动落盘为 JSONL。

四、其它说明
- 项目结构、主循环流程图与伪代码、System Prompt 全文、工具输出格式规范、
  各模块接口定义、错误处理分层策略、配置管理方式，均见仓库 docs/DESIGN.md。
- 运行环境：Python 3.10+，Windows / macOS / Linux 均可。
- 视频中演示的任务：让智能体在空目录下创建一个 Python 脚本、运行验证，
  并在发现报错后自行定位修复，最后调用 finish 给出总结。
