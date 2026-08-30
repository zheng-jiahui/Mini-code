"""
命令执行工具：run_command。

要点：
    · 单进程、同步、可超时：超时后杀掉整个进程树并返回已产生的部分输出
      （很多测试卡住时，前几行输出恰恰是定位问题的关键）；
    · stdout 与 stderr 合并返回，但保留退出码 —— 模型需要 exit code 判断成败；
    · 工作目录默认 workspace，可用 cwd 指定子目录；
    · 危险命令在执行前由 CommandGuard 判定（deny / confirm / allow）；
    · 以字节流捕获输出，按「系统首选编码（cp936/GBK）→ UTF-8」解码，避免 Windows 下中文乱码；
    · 交互式挂死检测：命令长时间零输出且停在未换行的提示符上时，判定为疑似等待输入并提前终止，
      避免 REPL / pager 之类把进程挂到整体超时。
"""

from __future__ import annotations

import locale
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List

from ..errors import SecurityError, ToolError
from .base import ToolContext, ToolResult, tool_spec

__all__ = ["run_command", "register"]


def _resolve_timeout(raw: Any, config: Any) -> tuple:
    """把模型传入的 timeout 收敛到 [1, 上限] 区间，并返回给模型的说明。

    为什么必须夹取上限：模型偶尔会传一个荒谬的值（如 99999），
    没有上限的话整条会话会挂死好几个小时，用户只能杀进程。
    这是"把模型的输出当不可信输入处理"，和路径沙箱是同一类边界。

    为什么不自动重试失败的命令：shell 命令可能有副作用（装依赖、写文件、
    提交），重试不等于重放——重复执行很可能造成重复副作用。
    所以这里只做"把诊断和建议讲清楚"，是否重试交给模型判断。

    ⚠️ 辅助函数必须定义在 @tool_spec 装饰器**之前**：装饰器紧贴它下面的第一个
    def，插在中间会让装饰器挂到辅助函数上，工具就退化成裸函数了。
    """
    default_timeout = int(getattr(config, "command_timeout", 120) or 120)
    cap = int(getattr(config, "max_command_timeout", 300) or 300)
    if cap < 1:
        cap = 300

    try:
        asked = int(raw) if raw is not None else default_timeout
    except (TypeError, ValueError):
        # 模型传了非数字——不要用异常打断工具，退回默认值并告知
        asked = default_timeout

    if asked <= 0:
        asked = default_timeout

    if asked > cap:
        return cap, f"（你请求的 timeout={asked}s 超过上限，已按 {cap}s 执行）"
    return asked, ""


@tool_spec(
    name="run_command",
    description=(
        "在工作区内执行一条 shell 命令（同步，带超时）。"
        "用于运行测试、安装依赖、执行脚本、查看版本等。"
        "命令应当是非交互式的：例如用 `pytest -q` 而不是等待输入的 `pytest`。"
        "stdout 与 stderr 会合并返回，并附带退出码。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的命令，如 `pytest -q`、`python main.py`"},
            "cwd": {"type": "string", "description": "执行目录，相对工作区，默认工作区根目录", "default": "."},
            "timeout": {"type": "integer", "description": "超时秒数，默认取配置中的 command_timeout"},
        },
        "required": ["command"],
    },
    dangerous=True,
    category="执行",
    when_not_to_use=(
        "只是读写文件就用 read_file/write_file/edit_block，不必绕道 shell。"
        "不要跑交互式命令（等待输入的 REPL、需要确认的 `rm -i`）——会卡到超时；"
        "也不要用一条超长复合命令同时做验证和清理，拆开才能定位是哪一步挂了。"
    ),
)
def run_command(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    command = (args.get("command") or "").strip()
    if not command:
        raise ToolError("命令为空", tool="run_command")

    # 1) 安全审查
    decision, reason = ctx.session.get("command_guard").check(command) if ctx.session.get("command_guard") else ("ok", "")
    policy = getattr(ctx.config, "command_policy", "confirm")
    if decision == "deny" or (decision == "confirm" and policy == "deny"):
        raise SecurityError(f"命令被拒绝执行：{reason}", tool="run_command",
                            hint="该命令命中危险命令黑名单，请改用更安全的方式达成目标。")
    if decision == "confirm" and policy == "confirm":
        if not ctx.confirm(f"即将执行危险命令：\n      {command}\n      原因：{reason}\n      是否继续？"):
            raise SecurityError("用户拒绝了这条命令", tool="run_command",
                                hint="请换一种更安全的方式；若必须执行，请让用户手动运行。")

    # 2) 解析 cwd
    cwd_arg = args.get("cwd") or "."
    cwd = ctx.resolve(cwd_arg, must_exist=True)
    if not cwd.is_dir():
        raise ToolError(f"cwd 不是目录：{cwd_arg}", tool="run_command")
    timeout, timeout_note = _resolve_timeout(args.get("timeout"), ctx.config)

    # 3) 构造环境
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env["NO_COLOR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    started_note = f"$ {command}    (cwd={ctx.guard.relpath(cwd)})"
    interactive_timeout = float(getattr(ctx.config, "interactive_timeout", 20))
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
        )
    except OSError as exc:
        raise ToolError(f"无法启动进程：{exc}", tool="run_command") from exc

    # 后台线程读取 stdout（字节流），保证超时/被杀后仍拿到已产生的部分输出。
    # 用字节而非 text 模式：Windows 子进程可能按系统编码（cp936/GBK）输出，
    # 用 utf-8 直接解码会把中文变成乱码（dir index.html → ������）。
    buf: List[bytes] = []
    reader_done = threading.Event()

    def _reader() -> None:
        try:
            fd = proc.stdout.fileno()
            # 用 os.read 而非 BufferedReader.read：后者会一直阻塞到凑满 4096 字节或 EOF，
            # 导致进程还活着时读不到已产生的部分输出（交互式挂死检测因此失效）。
            # os.read 返回「当前可用」的字节，能实时反映流式输出。
            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                buf.append(chunk)
        except (OSError, ValueError):  # noqa: BLE001 —— 管道被关/进程被杀都算正常结束
            pass
        finally:
            reader_done.set()

    threading.Thread(target=_reader, daemon=True).start()

    started = time.time()
    timed_out = False
    interactive_killed = False
    last_output_len = 0
    last_output_time = started
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        now = time.time()
        cur_len = sum(len(c) for c in buf)
        if cur_len > last_output_len:
            last_output_len = cur_len
            last_output_time = now
        # 交互式挂死检测：进程还活着、长时间零新输出，且缓冲区停在未换行的提示符上
        # （如 REPL 的 `>>>`），极可能在等待输入 —— 提前终止，避免挂到整体超时。
        if (now - last_output_time) >= interactive_timeout:
            peek = b"".join(buf)
            if peek and not peek.rstrip().endswith(b"\n"):
                _kill_tree(proc)
                interactive_killed = True
                break
        if (now - started) >= timeout:
            _kill_tree(proc)
            timed_out = True
            break
        time.sleep(0.2)
    reader_done.wait(timeout=5)
    out_bytes = b"".join(buf)

    # 解码：优先 UTF-8（现代工具与 PYTHONUTF8=1 的输出都是 UTF-8），失败再试系统 OEM 编码
    # （中文 Windows 为 cp936/GBK，cmd 内建命令、老工具的中文都按它输出），仍失败才用
    # UTF-8 替换兜底。这样无论子进程吐 UTF-8 还是 GBK 都不会乱码。
    # 注意：不能只用 locale.getpreferredencoding()——在开启 UTF-8 beta 的机器上它会返回 utf-8，
    # 此时若子进程输出的是 GBK 字节，用 utf-8 解码就会变成乱码。
    output = _decode_output(out_bytes)

    exit_code = proc.returncode if proc.returncode is not None else -1

    ctx.record_change("command", command[:120])

    parts = [started_note]
    if timeout_note:
        parts.append(timeout_note)
    if output.strip():
        parts.append(output.rstrip())
    else:
        parts.append("(无输出)")
    if interactive_killed:
        parts.append(
            f"[疑似等待输入] 命令运行约 {interactive_timeout:.0f}s 无任何新输出，且停在未换行的提示符上，"
            "疑似在等待交互输入；已提前终止（避免挂到整体超时）。"
            "请改用非交互式命令（例如用 `pytest -q` 而非会等待输入的 `pytest`）。"
        )
    if timed_out:
        # 只说"超时了"没用——模型会原样重试一次，再白耗同样多的时间。
        # 超时往往意味着范围太大（跑整个测试套件、全量构建），
        # 所以把"怎么缩小"和"确实需要更久时怎么办"一起讲清楚。
        cap = int(getattr(ctx.config, "max_command_timeout", 300) or 300)
        if timeout < cap:
            next_step = (f"② 若确实需要更久，显式传 timeout（上限 {cap}s）："
                         f'run_command(command="…", timeout={cap})；')
        else:
            next_step = f"② 本次已是上限 {cap}s，只能靠缩小范围，不能再加时间；"
        parts.append(
            f"[超时] 命令执行超过 {timeout}s 已被终止（以上为终止前的部分输出）。\n"
            f"下一步建议：① 缩小范围再跑（如 `pytest -q tests/test_x.py` 而非整个套件）；\n"
            f"{next_step}\n"
            f"③ 不要原样重试同一条命令，大概率还是超时。"
        )
    parts.append(f"[exit code: {exit_code}]")

    # 工具是否"正常执行完"：只要命令被成功拉起、拿到了输出，就算工具成功——
    # 哪怕命令自己以非 0 退出（pytest 失败、assert 触发）也是一次**有效的验证结果**，
    # 不是工具出错。真正的工具错误只有：超时、疑似等待输入被杀、进程根本起不来
    # （后者在前面以 ToolError 抛出，经 registry 转成 ok=False 回执）。
    # command_failed 显式标记"命令自身失败（非 0 退出）"，供主循环区分
    # "验证没过"（应计入自修复素材、但不算工具失败率）与"工具真出错"。
    command_failed = (exit_code != 0)
    ok = not timed_out and not interactive_killed
    result = ToolResult.success(
        "\n".join(parts),
        meta={
            "exit_code": exit_code,
            "timed_out": timed_out,
            "interactive_killed": interactive_killed,
            "command_failed": command_failed,
            "cwd": str(cwd),
        },
    )
    result.ok = ok
    # 仅真正的工具错误才把内容挪到 error 并清空 output；
    # 命令非 0 退出时保留 output（测试日志是模型修复的依据）。
    if not ok:
        result.error = "\n".join(parts)
        result.output = ""
    return result


def _kill_tree(proc: subprocess.Popen) -> None:
    """尽可能杀掉进程及其子进程（shell=True 时子进程是 shell 的孩子）。"""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
            )
        else:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001 —— 尽力而为
        try:
            proc.kill()
        except Exception:
            pass


def _oem_encoding() -> str:
    """Windows 控制台（非 Unicode 程序）输出用的 OEM 代码页，如中文 Windows 的 cp936/GBK。"""
    if os.name != "nt":
        return locale.getpreferredencoding(False) or "utf-8"
    try:
        import ctypes  # noqa: WPS433 —— 仅在 Windows 上用，延迟导入避免跨平台问题
        cp = ctypes.windll.kernel32.GetOEMCP()  # type: ignore[attr-defined]
        if cp:
            return f"cp{cp}"
    except Exception:  # noqa: BLE001 —— 取不到就用 gbk 兜底
        pass
    return "gbk"


def _decode_output(raw: bytes) -> str:
    """把子进程字节流解码成字符串，兼顾 UTF-8 与 GBK 两种常见中文输出。"""
    if not raw:
        return ""
    for enc in ("utf-8", _oem_encoding()):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def register(registry) -> None:
    registry.register(run_command)
