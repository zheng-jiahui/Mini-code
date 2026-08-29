"""
项目画像：扫描工作区，自动识别语言 / 框架 / 构建与测试命令，注入 system prompt。

为什么做：让模型"先看清项目再动手"不止靠 prompt 里的一句空话——
真正扫一遍目录，把"这是什么项目、怎么跑测试、怎么装依赖"直接喂给它，
比让它自己去猜（并可能猜错语言/框架）可靠得多。这是"自主性"的第一块基石。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .selfrepair import detect_test_command

__all__ = ["detect_project_profile", "format_project_profile", "looks_like_complex_task"]

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


def looks_like_complex_task(task: str) -> bool:
    """启发式：任务是否"较复杂、值得先出计划"。

    触发条件（任一）：描述较长（≥40 字）或命中复杂度关键词。
    简单任务（"写个除法函数""列目录"）不触发，避免不必要的计划开销。
    """
    t = (task or "").lower()
    if len(t) >= 40:
        return True
    return any(k in t for k in _COMPLEX_HINTS)
