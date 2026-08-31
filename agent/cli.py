"""
命令行入口：参数解析 + 交互式 REPL。

两种用法：
    python run.py                        # 进入 REPL，支持多轮连续对话
    python run.py -t "写一个快排并测试"    # 执行单个任务后退出

REPL 内以 `/` 开头的是控制命令，`!` 开头的是直接跑 shell（绕过模型，方便人肉介入）。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from .config import load_config
from .errors import AgentError, ConfigError
from .llm import build_backend
from .loop import AgentLoop
from .tools import build_default_registry
from .ui import Console

__all__ = ["build_argparser", "main"]

BANNER_HELP = """\
命令：
  /help              显示本帮助
  /exit, /quit       退出
  /clear             清空对话历史（不影响已写入的文件）
  /tools             列出可用工具及其签名
  /config            显示当前生效配置（密钥自动打码）
  /stats             显示本次会话的统计信息
  /diff              查看本次会话改了哪些文件（unified diff）
  /compact           手动压缩上下文
  /undo              撤销最近一次文件写入（覆盖写前自动备份）
  /new [任务名]      切换到另一个任务；同名任务沿用 workplace 下已有目录
  /dir               显示当前任务名与目录
  /save [路径]       把当前对话导出为 JSONL
  /checkpoint        手动保存会话检查点（下次可用 --resume 续跑）
  /resume [路径]     从检查点恢复会话（默认取最近一次）
  /mode [auto|ask|read_only]   切换写/破坏性操作的权限模式（默认 auto）
  !<命令>            直接执行 shell 命令（不经过模型）
其余内容作为自然语言任务发送给智能体。
"""

DEMO_SCRIPT: List[Dict[str, Any]] = [
    {"content": "我先看一下工作区里有什么。", "tool_calls": [{"name": "list_dir", "arguments": {"path": ".", "depth": 1}}]},
    {
        "content": "写一个文件并跑一下，验证环境可用。",
        "tool_calls": [
            {
                "name": "write_file",
                "arguments": {
                    "path": "hello_agent.py",
                    "content": 'def greet(name: str) -> str:\n    return f"Hello, {name}!"\n\n\nif __name__ == "__main__":\n    print(greet("MiniCode"))\n',
                },
            }
        ],
    },
    {"content": "运行脚本验证输出。", "tool_calls": [{"name": "run_command", "arguments": {"command": "python hello_agent.py"}}]},
    {"content": "完成。", "tool_calls": [{"name": "finish", "arguments": {"summary": "已创建 hello_agent.py 并运行验证，输出 Hello, MiniCode!。"}}]},
]


# ----------------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="minicode",
        description="MiniCode —— 不依赖任何 Agent 框架的编程智能体",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               "  python run.py\n"
               "  python run.py -t \"给 utils.py 里的函数补单元测试并跑通\"\n"
               "  python run.py --profile deepseek -t \"重构这段代码\"\n"
               "  python run.py --mock -t \"离线演示\"\n",
    )
    p.add_argument("-t", "--task", help="单次任务：执行完后退出")
    p.add_argument("-n", "--name", help="任务名：决定 workplace/ 下的文件夹名，"
                                        "同名任务复用同一目录（不指定则由任务描述自动生成）")
    p.add_argument("-c", "--config", help="配置文件路径（默认依次查找 config.yaml/json）")
    p.add_argument("-p", "--profile", help="选择 config.yaml 中的模型档位")
    p.add_argument("-w", "--workspace", help="工作区根目录（沙箱边界）")
    p.add_argument("-m", "--model", help="覆盖模型名")
    p.add_argument("--base-url", help="覆盖 API 端点")
    p.add_argument("--temperature", type=float, help="覆盖采样温度")
    p.add_argument("--max-steps", type=int, help="覆盖单任务最大步数")
    p.add_argument("--permission-mode", choices=["auto", "ask", "read_only"],
                   help="写/破坏性操作的权限模式：auto(默认,直接执行) / ask(执行前确认) / read_only(只放行只读工具)")
    p.add_argument("--backend", choices=["auto", "sdk", "http", "mock"], default="auto", help="LLM 后端实现")
    p.add_argument("--mock", action="store_true", help="离线演示模式（不发网络请求）")
    p.add_argument("--list-tools", action="store_true", help="列出所有工具后退出")
    p.add_argument("--list-profiles", action="store_true", help="列出配置中的模型档位后退出")
    p.add_argument("--print-prompt", action="store_true", help="打印完整 System Prompt 后退出")
    p.add_argument("--resume", action="store_true",
                   help="从上次保存的会话检查点恢复后再开始（默认取最近一次）")
    p.add_argument("-q", "--quiet", action="store_true", help="精简输出")
    p.add_argument("--no-color", action="store_true", help="关闭彩色输出")
    p.add_argument("-v", "--version", action="version", version=f"MiniCode {__version__}")
    return p


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    console = Console(verbose=not args.quiet, color=False if args.no_color else None)

    # ---- 列出工具（无需配置即可查看）----
    if args.list_tools:
        reg = build_default_registry()
        console.echo(reg.describe())
        return 0

    # ---- 配置 ----
    overrides: Dict[str, Any] = {
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "workspace": args.workspace,
        "max_steps": args.max_steps,
        "permission_mode": args.permission_mode,
    }
    # 离线演示模式不需要真实凭据
    is_mock = args.mock or args.backend == "mock"
    try:
        cfg = load_config(
            explicit=args.config,
            profile=args.profile,
            cli_overrides=overrides,
            require_api_key=not is_mock,
        )
    except ConfigError as exc:
        console.error(str(exc))
        console.echo("\n提示：复制 config.example.yaml 为 config.yaml 并填入凭据，或设置环境变量 OPENAI_API_KEY。")
        return 2

    if args.list_profiles:
        for name, prof in cfg.profiles.items():
            mark = "*" if name == cfg.llm.name else " "
            console.echo(f" {mark} {name}: {prof.model} @ {prof.base_url}")
        return 0

    backend_kind = "mock" if is_mock else args.backend
    try:
        backend = build_backend(cfg.llm, kind=backend_kind, script=DEMO_SCRIPT if backend_kind == "mock" else None)
    except ConfigError as exc:
        console.error(str(exc))
        return 2

    registry = build_default_registry()
    if args.print_prompt:
        from .prompts import build_system_prompt
        console.echo(build_system_prompt(
            tool_list=registry.describe(),
            workspace=str(cfg.agent.resolved_workspace()),
            native_tools=cfg.llm.native_tools,
        ))
        return 0

    loop = AgentLoop(cfg.agent, cfg.llm, backend, registry, console=console)

    console.banner(cfg.describe(), version=f"v{__version__}")
    if cfg.source:
        console.echo(console._c(f"  配置来源：{cfg.source}", "\033[90m"))
    if backend_kind == "mock":
        console.warn("离线演示模式：不会发送任何网络请求，模型回复来自内置脚本。")

    # ---- 恢复上次会话（仅显式要求时）----
    if args.resume:
        n = _do_resume(loop, console, None)
        if n:
            console.info(f"已从检查点恢复 {n} 条历史，当前任务「{loop.task_name}」→ {loop.task_dir}")
        else:
            console.warn("没有找到可恢复的检查点，将作为新会话开始。")
    elif not args.task:
        _hint_checkpoint(loop, console)

    # ---- 单次任务 ----
    if args.task:
        # 先定好目录再跑，用户等待时就知道产物会落在哪
        task_dir = loop.prepare_task_dir(args.task, task_name=args.name)
        console.info(f"任务「{loop.task_name}」→ {task_dir}")
        result = loop.run(args.task)
        console.final(result.answer)
        if result.backup_dir:
            console.info(f"已归档备份：{result.backup_dir}")
        if result.checkpoint:
            console.info(f"会话检查点：{result.checkpoint}（下次 --resume 可续跑）")
        console.stats({"步数": result.steps, "工具调用": result.tool_calls,
                       "失败": result.errors, "耗时": f"{result.elapsed:.1f}s",
                       "结束原因": result.finish_reason})
        return 0 if result.succeeded else 1

    # ---- REPL ----
    return _repl(loop, console)


def _do_resume(loop: AgentLoop, console: Console, path=None) -> int:
    """从检查点恢复，返回恢复的历史条数；没有可用检查点时返回 0。

    检查点**损坏**时报 error 而不是静默失败：用户必须能分清
    "没有可恢复的会话"与"有、但坏了"——前者正常开新任务，后者要删掉坏文件。
    """
    from . import checkpoint as _ck

    target = Path(str(path)) if path else _ck.latest(loop)
    if target is None or not Path(target).exists():
        return 0
    try:
        state = _ck.load(Path(target))
    except ValueError as exc:
        console.error(f"恢复失败（检查点已损坏）：{exc}")
        console.echo(f"  位置：{target}\n  删除该文件即可恢复正常启动。")
        return 0
    console.info(f"检查点：{_ck.describe(state)}")
    return loop.resume_checkpoint(target)


def _hint_checkpoint(loop: AgentLoop, console: Console) -> None:
    """启动时提示存在可恢复的会话，但**不自动恢复**。

    为什么不自动恢复：恢复会改变模型看到的上下文，而用户此刻可能只是想开个新任务。
    自动做的事必须没有歧义，"接着上次继续"显然有歧义——所以只提示，不代劳。
    """
    from . import checkpoint as _ck

    try:
        p = _ck.latest(loop)
        if p is None:
            return
        state = _ck.load(p)
    except (OSError, ValueError):
        return                      # 启动提示不该因为一个坏检查点而崩掉
    text = f"  发现上次的会话检查点：{_ck.describe(state)}；加 --resume 可继续。"
    console.echo(console._c(text, "\033[90m") if console.color else text)


def _repl(loop: AgentLoop, console: Console) -> int:
    console.echo("直接输入任务开始工作；/help 查看命令。")
    while True:
        try:
            raw = input(console._c("\n你 ▸ ", "\033[1m\033[36m") if console.color else "\n你 ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            console.echo("\n再见。")
            return 0

        if not raw:
            continue

        # ---- 控制命令 ----
        if raw.startswith("/"):
            cmd = raw.split()[0].lower()
            if cmd in ("/exit", "/quit", "/q"):
                console.echo("再见。")
                return 0
            if cmd == "/help":
                console.echo(BANNER_HELP)
            elif cmd == "/clear":
                loop.history.reset()
                console.info("对话历史已清空。")
            elif cmd == "/tools":
                console.echo(loop.registry.describe())
            elif cmd == "/config":
                console.echo(f"profile = {loop.profile.name}")
                for k, v in loop.profile.masked().items():
                    console.echo(f"  {k} = {v}")
                console.echo("agent:")
                for k in ("workspace", "max_steps", "command_timeout", "max_context_tokens",
                          "command_policy", "restrict_to_workspace", "backup_on_write", "auto_compact"):
                    console.echo(f"  {k} = {getattr(loop.config, k)}")
            elif cmd == "/stats":
                console.echo(loop.build_stats_panel())
            elif cmd == "/mode":
                parts = raw.split(maxsplit=1)
                cur = getattr(loop.config, "permission_mode", "auto")
                if len(parts) > 1 and parts[1].strip() in ("auto", "ask", "read_only"):
                    loop.config.permission_mode = parts[1].strip()
                    console.info(
                        f"权限模式已切换为：{loop.config.permission_mode}"
                        "（auto=直接执行 / ask=执行前确认 / read_only=只放行只读工具）"
                    )
                else:
                    console.echo(
                        f"用法：/mode [auto|ask|read_only]    当前模式：{cur}\n"
                        "  auto      —— 写/破坏性操作直接执行（默认）\n"
                        "  ask       —— 执行前交互确认（无交互环境自动放行）\n"
                        "  read_only —— 只放行只读工具，拒绝一切写/破坏性操作"
                    )
            elif cmd == "/diff":
                from .tools.review import build_diff
                console.echo(build_diff(loop.ctx.session.get("changes", [])))
            elif cmd == "/compact":
                # 走 loop.compact_context()：与自动压缩同一条路径，
                # 压缩后同样会补上「常驻事实 + 工作区当前状态」
                if loop.compact_context():
                    console.info(f"压缩完成，当前 ≈{loop.history.tokens} tokens")
                else:
                    console.info("无需压缩（历史较短）。")
            elif cmd == "/undo":
                console.echo(_undo(loop.console, loop.config, loop.task_name))
            elif cmd == "/new":
                parts = raw.split(maxsplit=1)
                name = parts[1].strip() if len(parts) > 1 else ""
                d = loop.prepare_task_dir("新任务", task_name=name, force_new=True)
                console.info(f"已开启新任务「{loop.task_name}」→ {d}")
            elif cmd == "/dir":
                console.info(f"当前任务「{loop.task_name or '尚未开始'}」→ {loop.task_dir or '—'}")
            elif cmd == "/checkpoint":
                p = loop.save_checkpoint()
                console.info(f"已保存检查点：{p}" if p else "没有可保存的内容（会话为空或未启用检查点）。")
            elif cmd == "/resume":
                parts = raw.split(maxsplit=1)
                n = _do_resume(loop, console, parts[1] if len(parts) > 1 else None)
                if n:
                    console.info(f"已恢复 {n} 条历史，当前任务「{loop.task_name}」→ {loop.task_dir}")
                else:
                    console.warn("没有可用的检查点。")
            elif cmd == "/save":
                parts = raw.split(maxsplit=1)
                path = parts[1] if len(parts) > 1 else "session.jsonl"
                Path(path).write_text(loop.history.to_jsonl(), encoding="utf-8")
                console.info(f"已导出到 {path}")
            else:
                console.warn(f"未知命令 {cmd}，输入 /help 查看可用命令。")
            continue

        # ---- 直接执行 shell ----
        if raw.startswith("!"):
            command = raw[1:].strip()
            if command:
                res = loop.registry.execute(
                    "run_command", {"command": command}, loop.ctx, call_id="manual"
                )
                console.echo(res.render(max_chars=loop.config.max_tool_output_chars))
            continue

        # ---- 正常任务 ----
        prev_dir = loop.task_dir
        try:
            result = loop.run(raw)
        except AgentError as exc:
            console.error(str(exc))
            continue
        # 只在新建目录时提示，追问沿用同一目录时不打扰
        if loop.task_dir != prev_dir:
            console.info(f"任务「{loop.task_name}」→ {loop.task_dir}")
        if result.backup_dir:
            console.info(f"已归档备份：{result.backup_dir}")
        console.final(result.answer)
        console.echo(console._c("  " + result.stats_line(), "\033[90m") if console.color else "  " + result.stats_line())


def _undo(console: Optional[Console], config, task_name: Optional[str] = None) -> str:
    """把最近一次"覆盖写"备份的文件恢复回去。

    只回滚 .overwrites 下的单文件备份；任务级完整快照（<任务名>_<时间戳>_<第N次>）
    是归档用的，不会被 /undo 改动，需要时直接去 .agent_backups 里拷贝即可。
    """
    home = getattr(config, "workspace_root", None) or config.workspace
    base = Path(str(home)).expanduser().resolve().parent
    root = base / (getattr(config, "backup_dir", None) or ".agent_backups") / ".overwrites"
    if task_name:
        root = root / task_name
    if not root.exists():
        return "没有找到任何覆盖写备份（.agent_backups/.overwrites 不存在）。"

    # 备份目录形如 .overwrites/<任务名>/<时间戳>/，取时间戳最大的那个
    candidates = sorted((p for p in root.rglob("*") if p.is_dir() and any(p.iterdir())),
                        key=lambda x: x.name)
    if not candidates:
        return "没有找到任何覆盖写备份。"
    latest = candidates[-1]
    restored = []
    for src in latest.rglob("*"):
        if src.is_file():
            rel = src.relative_to(latest)
            dest = config.resolved_workspace() / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            restored.append(str(rel))
    return f"已从 {latest.name} 恢复 {len(restored)} 个文件：{', '.join(restored) or '（空）'}"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
