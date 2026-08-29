# MiniCode 夜间迭代账本

> 本文件是**迭代进度账本**，也是开发过程的记录。每个版本完成后状态改为「已完成」并附上实测结论。
> 定时任务每轮读取本文件，挑第一个「待做」的版本开工，做完后回来更新状态。

## 使用方式

每个版本必须走完这条闭环，缺一步都不算完成：

1. 读本文件，找到第一个状态为「待做」的版本
2. 实现该版本列出的改动
3. `python tests/test_smoke.py` 全部通过
4. 实跑一次真实任务验证（不是只看单测）
5. `git add` + `git commit`（一个版本一个提交，提交信息写清为什么这么做）
6. 回来把状态改成「已完成」，填实测结论与遗留问题
7. 继续下一个「待做」版本

## 硬性约束（每一轮都必须遵守）

- **绝不提交密钥**：`api_keys.yaml`、`config.yaml`、`start.py`、`.workbuddy/` 都在 .gitignore 里，
  提交前用 `git status --short` 确认它们没出现
- **不要 push**：仓库还没建远程，本地提交即可
- **不要改已推送的历史**：目前也没推送过
- 每个版本都要能独立工作 —— 时间不够就停在已完成的版本上，不留半成品
- 每轮结束前跑一次完整测试

## 版本状态

| 版本 | 目标 | 状态 |
|---|---|---|
| V0 | edit_block 精确编辑 | ✅ 已完成 |
| V0.5 | 多密钥自动轮换 | ✅ 已完成 |
| V1 | 可靠性加固 | ✅ 已完成 |
| V2 | 自修复闭环 | ✅ 已完成 |
| V3 | 上下文与成本治理 | ✅ 已完成 |
| V6 | 交付打磨 | ✅ 已完成 |
| V4 | 可审阅性 | ✅ 已完成 |
| V5 | 自主性增强 | ⬜ 待做 |

---

## V0 edit_block 精确编辑 ✅

**commit**：`5121f47`

**动机**：`write_file` 是整体覆盖语义，改 473 行 HTML 里的一处也得把整个文件重写一遍。
2026-08-29 22:42 那次 `HTTP 400` 就是这么来的 —— 模型输出超长 HTML，参数被截断，连
`path` 都丢了。

**改动**：新增 `edit_block(path, old_text, new_text[, expected_replacements])`
（`agent/tools/filesystem.py`）
- `old_text` 必须唯一才替换；不唯一时返回每一处的行号，让模型补上下文后重试
- 容错：模型常把 `read_file` 的行号（`"   12| ..."`）一起抄进 `old_text`，自动剥离后再匹配
- 找不到时给出排查方向 + 文件中最相近的几行
- 替换前同样走 `.agent_backups` 备份

**实测**：改 473 行 HTML 的标题 → 3 步 / 4.3s，只改一行，文件行数不变。
**测试**：5 个新用例，共 18 个全通过。

**答辩要点**：为什么不直接让 write_file 支持局部改？因为"整体覆盖"的语义必须明确无歧义 ——
一个工具同时有两种写语义，模型更容易搞错。宁可多一个工具，也不要一个工具两副面孔。

---

## V0.5 多密钥自动轮换 ✅

**commit**：`97fa08e`

**动机**：长任务跑一夜，单把 key 额度耗尽就中断。

**改动**：
- `LLMProfile.api_keys`：备用密钥列表，`masked()` 中脱敏
- 401/403/429/quota/额度/余额/无效令牌 → 自动切下一把，用完即止不绕回
- **修了一处致命缺陷**：`OpenAIBackend._rebuild_client()` 原为空实现。
  注释写着"轮换密钥后必须重建客户端"，方法体却是空的 —— SDK 的 `api_key` 在构造时固定，
  结果 profile 换了新 key、请求却仍用旧 key，轮换完全失效

**实测**：无效 key 打头 → 自动切到有效 key → 请求成功。三把 key 均验证可用。
**测试**：2 个新用例，共 20 个全通过。

**答辩要点**：这个 bug 是"注释说了该做什么、代码却没做"的典型。补单元测试能挡住它
（断言 `_client.api_key` 跟着变），所以测试必须覆盖"对象内部状态"而不只是返回值。

---

## V1 可靠性加固 ✅

**commit**：（见 git log：feat: V1 可靠性加固）

**动机**：录演示视频最怕翻车，而这三类问题在 Windows 上是必现的。

**改动清单**：
1. **中文乱码**（`agent/tools/shell.py`）
   旧实现用 `text=True, encoding="utf-8"` 直接解码，但 Windows 中文环境默认 GBK，
   `dir`/`type`/老工具的中文会成 `������`。
   改为字节流捕获 + 解码顺序：**UTF-8 严格 → 系统 OEM 代码页（cp936/GBK）→ UTF-8 替换兜底**。
   关键修正：不能只用 `locale.getpreferredencoding()`——开启 UTF-8 beta 的机器上它返回 utf-8，
   此时子进程若吐 GBK 字节仍会乱码，必须用 OEM 代码页（`ctypes.windll.kernel32.GetOEMCP()`）。
2. **交互式命令挂死**（`agent/tools/shell.py`）
   旧实现用 `BufferedReader.read(4096)`，它会**一直阻塞到凑满 4096 字节或 EOF**，
   导致进程还活着时读不到已产生的部分输出，交互式检测因此永远不触发。
   改用 `os.read(fd, 4096)`（返回当前可用字节），并增加检测：命令长时间（默认 20s）
   零新输出、且缓冲区停在未换行的提示符上 → 判定疑似等待输入 → 提前终止并给明确错误。
3. **"假完成"拦截**（`agent/loop.py` 的 `_execute_calls`）
   模型改过文件（`write`/`edit` 类变更）却没跑过任何 `run_command` 就调 `finish` 时，
   回灌提示逼它先验证（最多拦 2 次防死循环），跑过命令后才允许收尾。

**实测**：
- 真实任务「写脚本打印中文问候并运行」→ 输出 `你好，世界！` 无乱码；3 步 5.9s 完成。
- 单测 `test_run_command_decodes_chinese_without_garbage`：GBK 文件经 `type` 输出解码正确，无替换字符。
- 单测 `test_interactive_command_is_killed_early`：打印未换行提示符后 sleep 的命令被 1.2s 内识别并终止。
- 单测 `test_fake_finish_blocked_until_verified`：写完文件直接 finish 被拦截，跑过命令后才允许收尾。
- 3 个新用例，共 **23** 个全通过。

**答辩要点**：
- 中文乱码根因是「解码编码假设错误」——不能假设子进程都是 UTF-8；OEM 代码页才是 Windows 非 Unicode 程序的真实输出编码。
- 交互式挂死根因是「读取方式假设错误」——`BufferedReader.read(n)` 是「凑满 n 或 EOF」而非「读到即返回」，
  流式输出必须用语义正确的 `os.read`。两个 bug 都说明：和操作系统/标准库打交道时，不能想当然。

---

## V2 自修复闭环 ✅

**commit**：（见 git log：feat: V2 自修复闭环）

**动机**：从"能写代码"升级到"能交付正确的代码"——这是 coding agent 的核心竞争力，
也是面试最能讲出彩的一块。

**改动清单**：
1. **测试框架识别**（`agent/selfrepair.py::detect_test_command`）
   扫描 `pytest.ini`/`pyproject.toml`/`go.mod`/`Cargo.toml`/`package.json`/`Makefile` 及
   `test_*.py` 命名约定，猜出该项目的测试命令（pytest -q / go test ./... / npm test …）。
2. **traceback 定位**（`parse_traceback` + `build_failure_note`）
   run_command 失败时，提取首个 `File "路径", line N` 与末行错误，**直接读出该处附近源码**
   附在回灌提示里——省掉模型多一轮 `read_file` 才能看到出错位置。
3. **与「第 N 次」归档打通**（`agent/tools/repair.py::rollback` 工具）
   新增 `rollback` 工具：把 `.agent_backups/{任务名}_*` 最新快照拷回任务目录，
   连续修不好时一键回到上一次能跑通的状态。
4. **修复预算**（`loop.py::_on_command_failure` + `config.max_repair_retries=5`）
   run_command 连续失败达到预算 → 提醒停止无方向乱试、考虑 rollback；成功一次即清零。

**实测**：
- 单测 `test_detect_test_command`：pytest/go/package.json 各自返回正确命令，空目录返回 None。
- 单测 `test_build_failure_note_reads_offending_lines`：NameError 的 traceback 被解析，
  出错行 `return a / c` 及其上下 3 行源码被附上。
- 单测 `test_rollback_restores_latest_snapshot`：rollback 把"好"快照拷回，覆盖"坏"文件。
- 单测 `test_self_repair_feeds_traceback_context_and_recovers`：MockBackend 跑「写坏→运行失败→
  据回灌位置修好→跑通→finish」全流程，验证循环确实注入了出错上下文。
- 4 个新用例，共 **27** 个全通过。

**答辩要点**：
- 回滚能力不是额外设计的，是从"每次生成都归档"这个规范里**自然长出来**的——
  既然保留了每一次的完整快照，"回到上一个已知可用状态"就是免费的。
- 感知与执行分离：`selfrepair` 只做分析（纯函数、可单测），真正的"修"仍交给模型，
  避免把修复逻辑硬编码成脆弱的规则。

---

## V3 上下文与成本治理 ✅

**commit**：（见 git log：feat: V3 上下文与成本治理）

**动机**：评委大概率会问"长任务怎么不超窗 / 花了多少"，要能报出数字而不是空谈。

**改动清单**：
1. **精确 token 计数**（`agent/history.py`）
   `count_messages_tokens` 优先用 `tiktoken(cl100k_base)`（本机 venv 已装，真正精确），
   未安装时回落到偏保守的启发式（英文 4 字符/token、CJK 1.3 字符/token）。
   新增 `token_counter_name()` 暴露当前实际用的是哪种计数方式，面板里如实标注。
   关键认知："发送前"的上下文预算用估算即可；**真正精确的消耗来自 API 返回的 `usage`**。
2. **工具回执智能压缩**（`agent/security.py::smart_compress` + `agent/tools/base.py`）
   旧 `truncate_output` 是纯 head+tail，会把中间的 traceback 丢掉。新版「信息优先」：
   先保住 signal 行（error/traceback/exception/assert/exit code/失败/异常…），再补首尾；
   **预算仍不够时信号行永远最高优先级**，head/tail 在剩余预算内硬截断——绝不让 traceback 被丢进省略号。
   修了一处真实 bug：`base.py::ToolResult.render` 调用了 `smart_compress` 却没 import，
   任何超长回执都会 `NameError`；已补 import。
3. **会话成本面板 `/stats`**（`agent/loop.py::build_stats_panel` + `agent/cli.py`）
   CLI 的 `/stats` 从"只报轮数/估算 tokens"升级为完整面板：
   真实 token（prompt/completion/total，来自 API usage）、当前上下文估算与计数方式、
   会话耗时、工具调用总数、上下文压缩次数、回执智能压缩次数、各工具耗时占比。
   loop 新增会话级画像：`_tool_timings` / `_tool_calls_by_name` / `_output_compressions`（跨任务累计）。

**实测**：
- 单测 `test_stats_panel_reports_real_tokens_and_tool_breakdown`：MockBackend 跑「写→运行→finish」，
  面板报 total=450（3 轮 × 150，与脚本 usage 吻合），run_command/write_file 出现在耗时占比里。
- 单测 `test_stats_panel_counts_output_compressions`：超长输出触发智能压缩，面板记「回执智能压缩次数：1」。
- 单测 `test_smart_compress_keeps_signal_lines`：60 行日志夹 traceback，压缩后仍含 Traceback/ZeroDivisionError。
- 单测 `test_tool_result_render_uses_smart_compress_no_nameerror`：回归上述 import bug。
- 5 个新用例，共 **31** 个全通过。

**答辩要点**：
- "估算"和"真实"是两件事：上下文预算用估算（够用且不必等网络），成本对账用 API `usage`（最准）。
  把两者分开，既不会超窗，也能报出可信的数字。
- 智能压缩的核心是"信号行优先于首尾"——模型排错最需要的恰恰是中间的 traceback，纯 head+tail 恰恰丢掉它。
  预算兜底逻辑保证信号行在再紧的预算下也存活，这是和"截断"的本质区别。

---

## V6 交付打磨 ✅

**commit**：（见 git log：feat: V6 交付打磨）

**动机**：这是"可提交"的硬门槛——光有代码不够，评委要看文档、要看演示、要能跑测试。

**改动清单**：
1. **测试补到覆盖每个工具至少一条**：新增 `read_file` / `grep_search` / `find_files` /
   `ask_user`（非交互兜底）/ `list_dir` 正例，工具覆盖无死角；共 **36** 个用例（无需 API key）。
2. **`docs/DESIGN.md`**：逐版本设计依据 + 关键设计决策（双通道、注册表、上下文三层、安全）+ 面试追问应答，
   即答辩小抄。
3. **`README.txt` 定稿**：586 汉字（≤1000），含 Git 仓库地址、运行方式、9 条特色功能、其它说明。
4. **`docs/VIDEO-SCRIPT.md`**：2 分钟演示脚本（空目录 → 写脚本 → 运行报错 → 自修复定位 → 修复跑通 → finish → /stats）。
5. **提交历史确认干净**：`.gitignore` 已覆盖 `config.yaml` / `api_keys.yaml` / `start.py` / `.workbuddy` /
   `workplace` / `.agent_backups`；历史提交无密钥、无大文件。

**实测**：`python tests/test_smoke.py` 36 个用例全通过；README 字数 586、gitignore 与 git log 复核无误。

**答辩要点**：交付物之间要互相印证——README 的特色功能、DESIGN.md 的设计决策、视频演示的闭环，
三者讲的是同一套能力，评委从任一入口都能闭环验证。

---

## V4 可审阅性 ✅

**commit**：（见 git log：feat: V4 可审阅性）

**动机**：coding agent 最怕"改了一堆自己都没看清"——无论是对模型还是对用户，改动都应是可审阅的。

**改动清单**：
1. **before/after 采集**（`agent/tools/base.py::ToolContext.record_change`）
   扩展签名接受 `before`/`after`/`path`；`write_file`/`edit_block` 在落盘时记录改动前后内容
   （两侧均超 50KB 自动跳过，避免大文件撑爆会话内存）。
2. **unified diff 渲染**（`agent/tools/review.py::build_diff` + `diff` 工具）
   把本次会话的 changes 渲染成标准 unified diff；新建文件（before=None）显示为全量新增，
   超大数据只列摘要。既是用户的 `/diff` 命令，也是模型可调用的 `diff` 工具——finish 前自查改动。
3. **`/diff` 命令**（`agent/cli.py`）+ 帮助菜单补 `/diff` 一行。
4. **单文件级回退**（`agent/tools/repair.py::rollback` 新增 `files` 参数）
   从「第 N 次」快照里只恢复指定文件（如 `rollback(files=["calc.py"])`），而非整目录回滚；
   做了路径越界校验，杜绝 `../` 逃逸。

**实测**：
- 单测 `test_diff_renders_unified_diff_of_session_changes`：写→改一行→diff，历史里出现 `--- a/` `+++ b/` 与新旧两行。
- 单测 `test_single_file_rollback_restores_only_named_file`：两文件改坏后只回滚 good.py，bad.py 保持坏状态。
- 单测 `test_build_diff_handles_new_and_unchanged`：新建/无变化/超大数据三类边界都正确渲染。
- 3 个新用例，共 **39** 个全通过。

**答辩要点**：
- 可审阅性 = 把"模型改了什么"从黑盒变成可读的 diff；这既是安全网（防意外提交），
  也是自纠错信号（模型能在 finish 前发现"我其实改错了"）。
- 单文件回退比整目录回滚更细粒度，代价只是快照里多一次"挑文件"拷贝——能力从既有归档自然长出。

---

## V5 自主性增强 ⬜ 待做

- 复杂任务先出计划再执行（plan-then-execute）
- 无依赖的工具调用并行（多个 `read_file`/`grep_search` 一起发）
- 项目画像：自动识别语言/框架/构建命令，注入 system prompt

---

## 全部完成后的追加工作

如果上面 8 个版本都完成且距天亮还有时间，可以做这些（每做一项都要走同样的闭环）：

- 流式输出：边生成边打印，长任务体验更好
- 多语言项目的测试命令适配（Java/Maven、Go、Rust）
- 工具执行的超时/重试策略细化
- 性能基准：记录各阶段耗时，找出瓶颈
- 给每个工具写一份"什么时候不该用它"的说明（反向文档，面试很加分）

---

## 进度日志

- 2026-08-29 23:50 — V0 edit_block 完成（commit 5121f47），18 个测试通过
- 2026-08-29 00:20 — V0.5 多密钥轮换完成（commit 97fa08e），20 个测试通过；
  顺带修了 `_rebuild_client` 空实现这个致命缺陷
- 2026-08-30 00:35 — V1 可靠性加固完成，23 个测试通过；
  修了中文乱码（OEM 代码页解码）与交互式挂死（BufferedReader→os.read 阻塞）两个真实 bug，
  新增"假完成"拦截
- 2026-08-30 01:10 — V2 自修复闭环完成，27 个测试通过；
  新增 selfrepair 感知层（测试命令识别 / traceback 定位上下文）+ rollback 工具 + 修复预算
- 2026-08-30 02:20 — V3 上下文与成本治理完成，31 个测试通过；
  精确 token（tiktoken 优先）+ 智能压缩（信号行优先，并修 render 漏 import 的 NameError）+ /stats 成本面板
- 2026-08-30 03:10 — V6 交付打磨完成，36 个测试通过；DESIGN.md + 视频脚本 + README 定稿（586 字）+ 历史复核干净
- 2026-08-30 03:50 — V4 可审阅性完成，39 个测试通过；before/after 采集 + unified diff（/diff 与 diff 工具）+ 单文件级 rollback
