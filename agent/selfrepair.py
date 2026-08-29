"""
自修复闭环的「感知」层：把一次失败的执行结果翻译成模型能直接修的上下文。

设计边界：
    · 本模块只做分析，不调用模型、不写文件、不执行命令。
    · 它从 run_command 的失败输出里提取两件事：
        ① 该项目「该怎么跑测试」（detect_test_command）；
        ② traceback 指向的「文件:行」附近的源码（build_failure_note），
          省掉模型多一轮 read_file 才能看到出错位置。
    · 真正的「修」由主循环把这些信息回灌给模型完成；回滚由 rollback 工具完成。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

# 各类项目「应该怎么跑测试」的特征：按出现顺序，先命中先返回
_TEST_HINTS = [
    ("pytest.ini", "pytest -q"),
    ("pyproject.toml", "pytest -q"),      # 含 [tool.pytest] 时也成立
    ("setup.py", "pytest -q"),
    ("conftest.py", "pytest -q"),
    ("go.mod", "go test ./..."),
    ("Cargo.toml", "cargo test"),
    ("package.json", "npm test"),
    ("Makefile", "make test"),
]

_TRACEBACK_RE = re.compile(r'File "([^"]+)", line (\d+)')


def detect_test_command(workspace) -> Optional[str]:
    """扫描工作区根目录，猜出该项目的测试命令。"""
    ws = Path(workspace)
    for marker, cmd in _TEST_HINTS:
        p = ws / marker
        if not p.exists():
            continue
        if marker == "pyproject.toml":
            try:
                return cmd if "tool.pytest" in p.read_text(encoding="utf-8", errors="replace") else None
            except OSError:
                return cmd
        return cmd
    # 退而求其次：目录里有测试文件命名约定
    if any(ws.rglob("test_*.py")) or any(ws.rglob("*_test.py")):
        return "pytest -q"
    return None


def parse_traceback(text: str) -> Optional[Dict[str, Any]]:
    """从输出里提取第一个 traceback 的 File/line 与末行错误信息。"""
    m = _TRACEBACK_RE.search(text or "")
    if not m:
        return None
    last = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return {"path": m.group(1), "line": int(m.group(2)), "error": last[-1] if last else ""}


def build_failure_note(output: str, ctx) -> Optional[str]:
    """失败执行后，生成一个带「出错位置源码上下文」的提示，直接喂给模型修。

    返回 None 表示没有可提取的失败上下文（无需干预）。
    只读取工作区内的文件（沙箱外文件在 traceback 里会被忽略，避免泄露 site-packages 等）。
    """
    tb = parse_traceback(output)
    if not tb:
        return None
    try:
        target = ctx.resolve(tb["path"], must_exist=True)
    except Exception:  # noqa: BLE001 —— 解析不出/不在沙箱内都不干预
        target = None

    head = f"运行失败，错误出现在 `{tb['path']}` 第 {tb['line']} 行附近：{tb['error']}。"
    if target and target.is_file():
        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        if lines:
            lo = max(1, tb["line"] - 3)
            hi = min(len(lines), tb["line"] + 3)
            snippet = "\n".join(f"{i:>4} | {lines[i - 1]}" for i in range(lo, hi + 1))
            head += f"\n该处附近的源码（请据此直接修复，不必再 read_file）：\n{snippet}"
    return head + "\n请定位根因后定向修改，再 run_command 验证，不要凭猜测乱改其他文件。"
