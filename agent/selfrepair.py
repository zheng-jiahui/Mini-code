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

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

_IS_WINDOWS = os.name == "nt"


# ----------------------------------------------------------------------------
# 各档特征的解析函数
# ----------------------------------------------------------------------------
def _read(ws: Path, name: str) -> str:
    """读一个标记文件的内容；读不到就当空串，绝不让 IO 问题中断探测。"""
    try:
        return (ws / name).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _maven_cmd(ws: Path) -> str:
    """Maven：优先用仓库自带的 wrapper（版本可控），没有才退回全局 mvn。

    wrapper 的调用方式分平台——Windows 下是 cmd.exe 执行，`./mvnw` 这种
    POSIX 写法跑不通，必须用 `mvnw.cmd`。
    """
    if _IS_WINDOWS and (ws / "mvnw.cmd").exists():
        return "mvnw.cmd -q test"
    if (ws / "mvnw").exists():
        return "mvnw -q test" if _IS_WINDOWS else "./mvnw -q test"
    return "mvn -q test"


def _gradle_cmd(ws: Path) -> str:
    """Gradle：同 Maven，优先 wrapper，且 wrapper 调用方式分平台。"""
    if _IS_WINDOWS and (ws / "gradlew.bat").exists():
        return "gradlew test"
    if (ws / "gradlew").exists():
        return "gradlew test" if _IS_WINDOWS else "./gradlew test"
    return "gradle test"


def _pytest_if_declared(ws: Path) -> Optional[str]:
    """pyproject.toml / setup.cfg：只在明确声明了 pytest 配置时才算数。

    大量 Python 项目有 pyproject.toml 却没写 [tool.pytest]（用默认配置跑 pytest），
    此时该档不成立，交给后面的 setup.py / 命名约定去兜，而不是让探测到此为止。
    """
    for name in ("pyproject.toml", "setup.cfg"):
        if (ws / name).exists() and "tool.pytest" in _read(ws, name):
            return "pytest -q"
    return None


def _npm_or_yarn(ws: Path) -> str:
    """Node 项目：有 yarn.lock 就用 yarn，否则 npm。"""
    return "yarn test" if (ws / "yarn.lock").exists() else "npm test"


def _has_target(text: str, name: str) -> bool:
    """判断 Makefile / justfile 里是否定义了某个目标。

    只认行首（`^test:`），避免把注释里的 "test:" 或别的目标的依赖行误判成目标定义。
    """
    return re.search(rf"^{re.escape(name)}\s*:", text, re.MULTILINE) is not None


def _make_test_if_target_exists(ws: Path) -> Optional[str]:
    """Makefile 只在真的有 test 目标时才建议 `make test`。

    否则 `make test` 必然报 "No rule to make target 'test'"——
    回灌一条注定失败的命令比不给建议更糟，会白耗一轮自修复预算。
    """
    return "make test" if _has_target(_read(ws, "Makefile"), "test") else None


def _just_test_if_recipe_exists(ws: Path) -> Optional[str]:
    return "just test" if _has_target(_read(ws, "justfile"), "test") else None

# 各类项目「应该怎么跑测试」的特征，按**可信度**从高到低排列，先命中先返回：
#   ① 专为测试而生的配置文件（pytest.ini / tox.ini / phpunit.xml…）——作者意图最明确
#   ② 构建文件所暗示的语言约定（go.mod / Cargo.toml / pom.xml / build.gradle…）
#   ③ 构建脚本里的 test 目标（Makefile / package.json）——必须先确认目标真的存在
#   ④ 文件命名约定（test_*.py…）——最弱，只在前面都没命中时兜底
# 值可以是固定字符串，也可以是 (workspace) -> Optional[str] 的解析函数：
# 后者用于「命令取决于仓库里还有什么」（如 Maven/Gradle wrapper）或
# 「要先确认目标存在」（如 Makefile 的 test 目标），返回 None 表示这一档不成立、继续往下探。
_Resolver = Callable[[Path], Optional[str]]
_TEST_HINTS: list = [
    # ---- ① 测试专用配置 ----
    ("pytest.ini", "pytest -q"),
    ("tox.ini", "tox"),
    ("noxfile.py", "nox"),
    ("phpunit.xml", "phpunit"),
    ("phpunit.xml.dist", "phpunit"),
    # ---- ② 构建文件 → 语言约定 ----
    ("Cargo.toml", "cargo test"),
    ("go.mod", "go test ./..."),
    ("pom.xml", _maven_cmd),
    ("build.gradle", _gradle_cmd),
    ("build.gradle.kts", _gradle_cmd),
    ("pyproject.toml", _pytest_if_declared),
    ("setup.cfg", _pytest_if_declared),
    ("setup.py", "pytest -q"),
    ("conftest.py", "pytest -q"),
    ("package.json", _npm_or_yarn),
    # ---- ③ 构建脚本里的 test 目标（目标不存在则不成立）----
    ("Makefile", _make_test_if_target_exists),
    ("justfile", _just_test_if_recipe_exists),
]

_TRACEBACK_RE = re.compile(r'File "([^"]+)", line (\d+)')


def detect_test_command(workspace) -> Optional[str]:
    """扫描工作区根目录，猜出该项目的测试命令；猜不出来返回 None。

    注意：返回的是**推断**结果而非事实，调用方（自修复回灌、项目画像）据此给模型
    提示时应当说明这一点——猜错时代价很小（一条失败的命令），猜不出来代价是大
    （模型只能盲改），所以宁可猜也不要不猜。
    """
    ws = Path(workspace)
    for marker, cmd in _TEST_HINTS:
        if not (ws / marker).exists():
            continue
        resolved = cmd(ws) if callable(cmd) else cmd
        # 关键：解析不成立时必须继续往下探，不能 return None 提前终止整条链。
        # 早期版本在 pyproject.toml 缺少 [tool.pytest] 时直接 `return None`，
        # 而现代 Python 项目几乎都有 pyproject.toml —— 探测链在最常见的文件上就断了。
        if resolved:
            return resolved

    # ④ 兜底：目录里有测试文件命名约定
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
