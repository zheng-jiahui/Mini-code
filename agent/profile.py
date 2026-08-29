"""
项目画像：扫描工作区，自动识别语言 / 框架 / 构建与测试命令，注入 system prompt。

为什么做：让模型"先看清项目再动手"不止靠 prompt 里的一句空话——
真正扫一遍目录，把"这是什么项目、怎么跑测试、怎么装依赖"直接喂给它，
比让它自己去猜（并可能猜错语言/框架）可靠得多。这是"自主性"的第一块基石。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from .selfrepair import detect_test_command

__all__ = ["detect_project_profile", "format_project_profile", "looks_like_complex_task",
           "list_workspace_files", "format_workspace_state", "WORKSPACE_STATE_MARKER"]

_LANG_MARKERS = [
    ("requirements.txt", "Python"),
    ("pyproject.toml", "Python"),
    ("setup.py", "Python"),
    ("setup.cfg", "Python"),
    ("Pipfile", "Python"),
    ("package.json", "JavaScript/TypeScript"),
    ("tsconfig.json", "JavaScript/TypeScript"),
    ("go.mod", "Go"),
    ("Cargo.toml", "Rust"),
    ("pom.xml", "Java"),
    ("build.gradle", "Java"),
    ("build.gradle.kts", "Java"),
    ("Gemfile", "Ruby"),
    ("composer.json", "PHP"),
    ("pubspec.yaml", "Dart"),
]

# 构建命令：命中对应标记就提示对应的"装依赖/构建"命令
_BUILD_CMDS = {
    "Python": "pip install -r requirements.txt",
    "JavaScript/TypeScript": "npm install",
    "Go": "go mod tidy",
    "Rust": "cargo build",
    "Java": "mvn -q package",
    "Ruby": "bundle install",
    "PHP": "composer install",
    "Dart": "dart pub get",
}

# 复杂度启发词：任务描述里出现这些，倾向"先出计划再动手"
_COMPLEX_HINTS = (
    "实现", "重构", "搭建", "系统", "项目", "架构", "框架", "模块", "服务", "应用",
    "端到端", "pipeline", "爬虫", "网站", "后端", "前端", "sdk", "cli", "api",
)


def detect_project_profile(workspace) -> Dict[str, Any]:
    """扫描工作区，返回结构化的项目画像。"""
    ws = Path(workspace)
    languages: List[str] = []
    build_cmds: List[str] = []
    frameworks: List[str] = []

    for marker, lang in _LANG_MARKERS:
        if (ws / marker).exists() and lang not in languages:
            languages.append(lang)
            if lang in _BUILD_CMDS and _BUILD_CMDS[lang] not in build_cmds:
                build_cmds.append(_BUILD_CMDS[lang])

    # 从 package.json 抽取依赖名（最有信息量的"框架"线索）
    pkg = ws / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8", errors="replace") or "{}")
            deps = {**((data.get("dependencies") or {})), **((data.get("devDependencies") or {}))}
            frameworks.extend(list(deps.keys())[:12])
        except Exception:
            pass

    test_cmd = detect_test_command(ws)
    return {
        "languages": languages,
        "frameworks": frameworks,
        "build_cmds": build_cmds,
        "test_cmd": test_cmd,
    }


def format_project_profile(profile: Dict[str, Any]) -> str:
    """把画像渲染成可注入 system prompt 的文本段落。空画像返回空串。"""
    if not profile:
        return ""
    lines = ["# 项目画像（已自动识别，请据此选择语言/框架/命令）"]
    langs = profile.get("languages") or []
    lines.append(f"- 主要语言：{', '.join(langs) if langs else '（未识别到，按任务默认语言处理）'}")
    fw = profile.get("frameworks") or []
    if fw:
        lines.append(f"- 已声明的依赖/框架：{', '.join(fw[:12])}")
    builds = profile.get("build_cmds") or []
    if builds:
        lines.append(f"- 安装/构建命令：{'; '.join(builds)}")
    tc = profile.get("test_cmd")
    if tc:
        lines.append(f"- 测试命令：{tc}（运行验证时优先用它）")
    else:
        lines.append("- 测试命令：未在仓库中发现标准测试配置；如任务需要测试，请先按项目约定建立。")
    return "\n".join(lines)


# ---- 工作区状态扫描（压缩后重建"磁盘上现在有什么"）----
# 与 filesystem._IGNORE_DIRS 保持同一口径，免得"列目录"和"扫状态"漏掉不同的东西。
_IGNORE_FOR_STATE = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea", ".vscode",
    ".agent_backups", ".agent_sessions", "dist", "build", ".next", "coverage",
}
_MAX_STATE_FILES = 30
# 清单的起始标记：压缩后要能认出"上一次注入的那份"并删掉，只留最新。
WORKSPACE_STATE_MARKER = "【工作区当前状态】"
# 数行必须把文件读进来；超过这个体量就只报大小。一个几十 MB 的日志/数据集，
# 不该让"重建工作区状态"这件事本身变成负担。
_MAX_LINE_COUNT_BYTES = 512 * 1024


def list_workspace_files(workspace, max_files: int = _MAX_STATE_FILES) -> List[Dict[str, Any]]:
    """扫一遍工作区，返回 [{"path": 相对路径, "lines": 行数或 None, "size": 字节}]。

    遍历时用 os.walk 并**原地剪掉噪声目录**，而不是 rglob 之后再过滤——
    后者会先把 node_modules 整个走一遍才丢弃，长任务里这是白花的时间。
    """
    ws = Path(workspace)
    if not ws.is_dir():
        return []

    entries: List[Dict[str, Any]] = []
    truncated = False
    for dirpath, dirnames, filenames in os.walk(ws):
        dirnames[:] = [d for d in dirnames if d not in _IGNORE_FOR_STATE]
        for name in filenames:
            p = Path(dirpath) / name
            try:
                rel = p.relative_to(ws).as_posix()
                size = p.stat().st_size
            except (ValueError, OSError):
                continue
            if len(entries) >= max_files:
                truncated = True
                break
            lines = None
            if size <= _MAX_LINE_COUNT_BYTES:
                try:
                    with p.open("rb") as f:
                        lines = sum(1 for _ in f)
                except OSError:
                    lines = None
            entries.append({"path": rel, "lines": lines, "size": size})
        if truncated:
            break

    entries.sort(key=lambda e: e["path"].lower())
    if truncated:
        entries.append({"path": "\u2026", "lines": None, "size": 0, "more": True})
    return entries


def format_workspace_state(entries: List[Dict[str, Any]]) -> str:
    """把工作区状态渲染成注入给模型的提示。空工作区返回空串。

    这里**刻意不含文件内容**：一是体积会失控，二是"列清单"与"读内容"是两件事。
    清单的作用是让模型知道磁盘上现在有什么、别去编辑一个不存在的文件；
    内容仍然要它用 read_file 现读——否则它拿压缩前的旧记忆去写 old_text，必然对不上。
    """
    if not entries:
        return ""
    shown = [e for e in entries if not e.get("more")]
    lines = [f"{WORKSPACE_STATE_MARKER}以下是压缩后**重新扫描磁盘**得到的真实文件清单"
             f"（{len(shown)} 个文件，只列路径与行数，不含内容）："]
    for e in entries:
        if e.get("more"):
            lines.append("  \u2026（文件更多，已省略；需要时用 list_dir / find_files 查看）")
            continue
        n = e.get("lines")
        lines.append(f"  {e['path']}" + (f"\u2014 {n} 行" if n is not None
                                        else f"\u2014 {_human_bytes(e.get('size') or 0)}"))
    lines.append(
        "注意：上面没有文件内容。要用 edit_block 精确修改某个文件时，请先用 read_file 读取"
        "当前内容，不要凭记忆写 old_text\u2014\u2014旧内容可能已在压缩中被丢弃，凭记忆写必然对不上。"
    )
    return "\n".join(lines)


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}B" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}GB"


def looks_like_complex_task(task: str) -> bool:
    """启发式：任务是否"较复杂、值得先出计划"。

    触发条件（任一）：描述较长（≥40 字）或命中复杂度关键词。
    简单任务（"写个除法函数""列目录"）不触发，避免不必要的计划开销。
    """
    t = (task or "").lower()
    if len(t) >= 40:
        return True
    return any(k in t for k in _COMPLEX_HINTS)
