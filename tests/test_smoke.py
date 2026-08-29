"""
冒烟测试：全部使用 Mock 后端，不需要 API key 也能跑。

    python -m pytest tests -q
    # 或直接运行：
    python tests/test_smoke.py

覆盖点（对应设计文档里的关键承诺）：
    1. 原生 tool_calls 通道端到端跑通并以 finish 结束；
    2. 文本协议通道（模型不支持 function calling）同样跑通；
    3. 解析器：未知工具 / 缺必填参数 / 类型错误都能转成给模型的 issues；
    4. 工具执行永不抛出：内部异常被转成 ok=False 的回执；
    5. 路径沙箱：越界读写被拦截；
    6. 危险命令：deny 策略下被拒绝；
    7. 上下文压缩：超阈值时触发，system 提示词不被丢弃。
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import AgentConfig, LLMProfile, load_config  # noqa: E402
from agent.llm import AssistantMessage, MockBackend  # noqa: E402
from agent.loop import AgentLoop  # noqa: E402
from agent.parser import ToolCallParser, extract_text_calls  # noqa: E402
from agent.tools import build_default_registry, build_tool_context  # noqa: E402
from agent.tools.base import ToolResult  # noqa: E402


# ----------------------------------------------------------------------------
# 辅助
# ----------------------------------------------------------------------------
def _make_env(tmpdir: str, native: bool = True, **agent_overrides):
    # 端到端用例期望产物直接落在 tmp 里，所以默认关闭任务子目录
    opts = {"per_task_dir": False}
    opts.update(agent_overrides)
    cfg = AgentConfig(workspace=tmpdir, session_log=None, **opts)
    profile = LLMProfile(native_tools=native, api_key="test", model="mock")
    registry = build_default_registry()
    ctx = build_tool_context(cfg, console=None, session={})
    return cfg, profile, registry, ctx


def _scripted(script, native: bool = True):
    profile = LLMProfile(native_tools=native, api_key="test", model="mock")
    return MockBackend(profile, script), profile


# ----------------------------------------------------------------------------
# 1) 原生通道
# ----------------------------------------------------------------------------
def test_native_channel_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        script = [
            {"content": "写文件。", "tool_calls": [
                {"name": "write_file", "arguments": {"path": "a.py", "content": "print('hi')\n"}}]},
            {"content": "执行。", "tool_calls": [
                {"name": "run_command", "arguments": {"command": f"python a.py"}}]},
            {"content": "完成。", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "done"}}]},
        ]
        backend, profile = _scripted(script, native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        result = loop.run("创建一个脚本并运行")

        assert (Path(tmp) / "a.py").exists()
        assert (Path(tmp) / "a.py").read_text(encoding="utf-8") == "print('hi')\n"
        assert result.finish_reason == "finish", result.finish_reason
        assert result.answer == "done"
        assert result.tool_calls == 3
        assert result.errors == 0


# ----------------------------------------------------------------------------
# 2) 文本协议通道
# ----------------------------------------------------------------------------
def test_text_protocol_channel():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp, native=False)
        script = [
            {"content": "读取并追加内容。", "tool_calls": [
                {"name": "write_file", "arguments": {"path": "b.txt", "content": "line1\n"}},
                {"name": "write_file", "arguments": {"path": "b.txt", "content": "line2\n", "append": True}},
            ]},
            {"content": "完成。", "tool_calls": [{"name": "finish", "arguments": {"summary": "ok"}}]},
        ]
        backend, profile = _scripted(script, native=False)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        result = loop.run("用文本协议写文件")

        assert (Path(tmp) / "b.txt").read_text(encoding="utf-8") == "line1\nline2\n"
        assert result.finish_reason == "finish", (result.finish_reason, result.answer)
        # 文本协议下，工具回执以 user 角色回灌
        assert any(m["role"] == "user" and "<tool_result" in (m.get("content") or "")
                   for m in loop.history.messages)


# ----------------------------------------------------------------------------
# 3) 解析器
# ----------------------------------------------------------------------------
def test_parser_text_extraction():
    text = (
        "我先看一下结构。\n"
        '```json\n{"tool": "read_file", "args": {"path": "x.py", "limit": 20}}\n```\n'
        "然后搜索。\n"
        '```json\n{"tool": "grep_search", "args": {"pattern": "def test"}}\n```'
    )
    narration, calls, issues = extract_text_calls(text)
    assert not issues, issues
    assert "我先看一下结构" in narration
    assert [c["name"] for c in calls] == ["read_file", "grep_search"]
    assert calls[0]["arguments"]["limit"] == 20


def test_parser_rejects_unknown_tool_and_bad_args():
    registry = build_default_registry()
    parser = ToolCallParser(registry, use_native=False)

    msg = AssistantMessage(
        content='```json\n{"tool": "delete_everything", "args": {}}\n```',
        finish_reason="stop",
    )
    out = parser.parse(msg)
    assert not out.calls
    assert any("未知工具" in i for i in out.issues)

    msg2 = AssistantMessage(content='```json\n{"tool": "read_file", "args": {}}\n```')
    out2 = parser.parse(msg2)
    assert not out2.calls
    assert any("缺少必填参数" in i for i in out2.issues)

    msg3 = AssistantMessage(content='```json\n{"tool": "read_file", "args": {"path": "a.py", "offset": "abc"}}\n```')
    out3 = parser.parse(msg3)
    # "abc" 无法转成 integer → 记为类型错误
    assert any("类型错误" in i for i in out3.issues)


# ----------------------------------------------------------------------------
# 4) 工具执行不抛异常
# ----------------------------------------------------------------------------
def test_registry_never_raises():
    with tempfile.TemporaryDirectory() as tmp:
        _, _, registry, ctx = _make_env(tmp)
        # 未知工具
        r = registry.execute("no_such_tool", {}, ctx)
        assert isinstance(r, ToolResult) and not r.ok
        # 参数非法导致的内部异常（read_file 传 None）
        r2 = registry.execute("read_file", {"path": 12345}, ctx)
        assert isinstance(r2, ToolResult) and not r2.ok


# ----------------------------------------------------------------------------
# 5) 路径沙箱
# ----------------------------------------------------------------------------
def test_path_sandbox_blocks_escape():
    with tempfile.TemporaryDirectory() as tmp:
        outside = Path(tmp).parent / "outside_secret.txt"
        outside.write_text("secret", encoding="utf-8")
        _, _, registry, ctx = _make_env(tmp)

        r = registry.execute("read_file", {"path": str(outside)}, ctx)
        assert not r.ok
        assert "越界" in r.render() or "路径" in r.render()

        r2 = registry.execute("write_file", {"path": "../escape.py", "content": "x"}, ctx)
        assert not r2.ok


# ----------------------------------------------------------------------------
# 6) 危险命令
# ----------------------------------------------------------------------------
def test_dangerous_command_denied():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, _, registry, ctx = _make_env(tmp, command_policy="deny")
        r = registry.execute("run_command", {"command": "rm -rf /"}, ctx)
        assert not r.ok
        assert "拒绝" in r.render() or "拦截" in r.render()


# ----------------------------------------------------------------------------
# 7) 上下文压缩
# ----------------------------------------------------------------------------
def test_compaction_keeps_system_prompt():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        backend, profile = _scripted([], native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)

        for i in range(20):
            loop.history.add_user(f"第 {i} 条用户消息，用于撑大上下文。" * 20)
            loop.history.add_assistant(AssistantMessage(content=f"第 {i} 条回复"))

        before = loop.history.tokens
        ok = loop.history.compact(llm=backend, keep_recent=4)
        assert ok
        assert loop.history.messages[0]["role"] == "system"
        assert loop.history.messages[0]["content"] == loop.system_prompt
        assert loop.history.tokens < before
        assert loop.history.compact_count == 1


# ----------------------------------------------------------------------------
# 8) 无进展停滞检测
# ----------------------------------------------------------------------------
def test_stagnation_detection_stops_loop():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp, max_steps=30)
        repeat = {"name": "list_dir", "arguments": {"path": "."}}
        script = [{"content": "再看一次。", "tool_calls": [repeat]} for _ in range(10)]
        backend, profile = _scripted(script, native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        result = loop.run("反复列目录（用于测试停滞检测）")
        assert result.finish_reason == "no_progress", result.finish_reason


# ----------------------------------------------------------------------------
# 9) 配置加载
# ----------------------------------------------------------------------------
def test_config_env_placeholder_expansion(tmp_path=None):
    os.environ["MINICODE_TEST_KEY"] = "sk-unit-test-0000000000"
    cfg_file = Path(os.environ.get("TMP", "/tmp")) / "minicode_test_config.yaml"
    cfg_file.write_text(
        "profiles:\n"
        "  default:\n"
        "    base_url: https://example.invalid/v1\n"
        "    api_key: ${MINICODE_TEST_KEY}\n"
        "    model: unit-test-model\n"
        "agent:\n"
        "  workspace: .\n",
        encoding="utf-8",
    )
    cfg = load_config(explicit=str(cfg_file))
    assert cfg.llm.api_key == "sk-unit-test-0000000000"
    assert cfg.llm.model == "unit-test-model"
    assert "***" in cfg.llm.masked()["api_key"] and "unit-test" not in cfg.llm.masked()["api_key"]
    cfg_file.unlink(missing_ok=True)


# ----------------------------------------------------------------------------
# 11) 任务级子目录
# ----------------------------------------------------------------------------
def test_per_task_dir_isolates_each_task():
    """开启 per_task_dir 后：每个任务独占一个子目录，代码与备份都归拢在里面。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp, per_task_dir=True, backup_on_write=True)
        script = [
            {"content": "写文件。", "tool_calls": [
                {"name": "write_file", "arguments": {"path": "a.py", "content": "print('hi')\n"}}]},
            {"content": "覆盖写一次，用于触发备份。", "tool_calls": [
                {"name": "write_file", "arguments": {"path": "a.py", "content": "print('hi2')\n"}}]},
            {"content": "完成。", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "done"}}]},
        ]
        backend = MockBackend(profile, script)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)

        result = loop.run("写一个快排脚本")
        assert result.succeeded, result.error_message

        # 产物不在工作区根目录，而在唯一的任务子目录里
        entries = list(Path(tmp).iterdir())
        assert len(entries) == 1, f"工作区根目录应只有 1 个任务子目录，实际：{entries}"
        task_dir = entries[0]
        assert task_dir.is_dir()
        assert re.match(r"^\d{8}-\d{6}", task_dir.name), f"子目录名应带时间戳前缀：{task_dir.name}"
        assert "快排" in task_dir.name, f"子目录名应含任务摘要：{task_dir.name}"

        # 代码与备份都归拢在该任务目录内
        assert (task_dir / "a.py").exists(), "代码应落在任务子目录里"
        assert (task_dir / "a.py").read_text(encoding="utf-8") == "print('hi2')\n"
        backup_root = task_dir / ".agent_backups"
        assert backup_root.exists(), "备份目录应跟着任务走，而不是散在全局"
        assert any(backup_root.rglob("*.py")), "覆盖写应产生备份文件"

        # 同一会话内追问：沿用同一目录（否则模型看不到上一轮刚生成的文件）
        backend2 = MockBackend(profile, [
            {"content": "完成。", "tool_calls": [{"name": "finish", "arguments": {"summary": "ok"}}]}])
        followup = AgentLoop(cfg, profile, backend2, registry, console=None)
        followup._task_dir = task_dir                 # 模拟同一会话继续
        followup.run("改成降序输出")
        assert followup.task_dir == task_dir, "同会话追问应沿用同一任务目录"

        # /new 开启新目录：新目录仍在工作区根下（不套娃），旧任务文件不受影响
        newer = followup.prepare_task_dir("另一个任务", force_new=True)
        assert newer != task_dir, "/new 应创建新目录"
        assert newer.parent == Path(tmp), "新目录应直接建在工作区根下，而不是套在旧任务里"
        assert (task_dir / "a.py").exists(), "旧任务目录里的文件不应被动过"


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'全部通过' if failures == 0 else str(failures) + ' 个用例失败'}")
    sys.exit(1 if failures else 0)
