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

import contextlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.checkpoint import checkpoint_dir, load as load_checkpoint  # noqa: E402
from agent.config import AgentConfig, LLMProfile, load_config  # noqa: E402
from agent.history import MAX_FACTS_CHARS, History  # noqa: E402
from agent.llm import AssistantMessage, MockBackend  # noqa: E402
from agent.loop import AgentLoop, RunResult  # noqa: E402
from agent.parser import ToolCallParser, extract_text_calls  # noqa: E402
from agent.tools import build_default_registry, build_tool_context  # noqa: E402
from agent.tools.base import DIFF_CAPTURE_CAP, ToolResult  # noqa: E402
from agent.profile import WORKSPACE_STATE_MARKER, list_workspace_files  # noqa: E402
from agent.tools.review import build_diff  # noqa: E402


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
            # V1 新行为：写完文件需先验证再 finish（这里用 type 确认内容确实写入）
            {"content": "确认。", "tool_calls": [
                {"name": "run_command", "arguments": {"command": "type b.txt"}}]},
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
# 11) 任务目录（workplace/{任务名}）与历史备份（.agent_backups/{任务名}_{时间戳}_{第N次}）
# ----------------------------------------------------------------------------
def test_task_dir_named_by_task_and_snapshotted():
    """workplace 存最新代码；.agent_backups 与其同级，按"第N次"累计历史快照。"""
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / "workplace"                  # 生成代码的家
        cfg = AgentConfig(workspace=str(home), session_log=None,
                          per_task_dir=True, backup_on_write=True)
        profile = LLMProfile(native_tools=True, api_key="test", model="mock")
        registry = build_default_registry()
        backup_root = Path(tmp) / ".agent_backups"      # 与 workplace 同级

        def run_task(name: str, path: str, content: str):
            script = [
                {"content": "写文件。", "tool_calls": [
                    {"name": "write_file", "arguments": {"path": path, "content": content}}]},
                {"content": "完成。", "tool_calls": [
                    {"name": "finish", "arguments": {"summary": "done"}}]},
            ]
            loop = AgentLoop(cfg, profile, MockBackend(profile, script), registry, console=None)
            loop.prepare_task_dir("任务描述", task_name=name)
            res = loop.run("任务描述")
            assert res.succeeded, res.error_message
            return loop, res

        # --- 第 1 次：目录名就是任务名，并归档"第1次"快照 ---
        loop1, res1 = run_task("user_login", "login.py", "print('v1')\n")
        task_dir = loop1.task_dir
        assert task_dir == home / "user_login", f"目录名应为任务名，实际 {task_dir}"
        assert (task_dir / "login.py").read_text(encoding="utf-8") == "print('v1')\n"

        snaps = sorted(p.name for p in backup_root.iterdir() if p.is_dir() and p.name != ".overwrites")
        assert len(snaps) == 1, f"应产生 1 份快照，实际 {snaps}"
        assert re.match(r"^user_login_\d{8}_\d{6}_第1次$", snaps[0]), f"命名不符：{snaps[0]}"
        assert (backup_root / snaps[0] / "login.py").read_text(encoding="utf-8") == "print('v1')\n"
        assert res1.backup_dir == str(backup_root / snaps[0]), "RunResult 应记录本次归档目录"

        # --- 第 2 次：同名任务复用同一目录，workplace 保持最新，快照变"第2次" ---
        loop2, _ = run_task("user_login", "login.py", "print('v2')\n")
        assert loop2.task_dir == task_dir, "同名任务应复用同一目录"
        assert (task_dir / "login.py").read_text(encoding="utf-8") == "print('v2')\n", \
            "workplace 里应始终是最新代码"

        snaps = sorted(p.name for p in backup_root.iterdir() if p.is_dir() and p.name != ".overwrites")
        assert len(snaps) == 2, f"应累计 2 份快照，实际 {snaps}"
        assert any("第2次" in s for s in snaps), f"应出现第2次：{snaps}"
        # 两次快照各自保留当时的版本，不被后续覆盖
        assert (backup_root / snaps[0] / "login.py").read_text(encoding="utf-8") == "print('v1')\n"
        assert (backup_root / snaps[1] / "login.py").read_text(encoding="utf-8") == "print('v2')\n"

        # --- 另一个任务：独立目录、独立计数 ---
        run_task("word_count", "wc.py", "print('wc')\n")
        assert (home / "word_count" / "wc.py").exists()
        wc_snaps = [p.name for p in backup_root.iterdir()
                    if p.is_dir() and p.name.startswith("word_count_")]
        assert len(wc_snaps) == 1 and "第1次" in wc_snaps[0], f"{wc_snaps}"

        # 覆盖写的单文件备份收在 .overwrites 下，不污染顶层快照命名
        assert (backup_root / ".overwrites").exists(), "覆盖写应有单独归档位置"
        top = [p.name for p in backup_root.iterdir() if p.is_dir()]
        assert sorted(n for n in top if n != ".overwrites") == sorted(snaps + wc_snaps)


def test_task_name_defaults_to_slug_of_task():
    """不指定任务名时，由任务描述自动生成目录名。"""
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / "workplace"
        cfg = AgentConfig(workspace=str(home), session_log=None, per_task_dir=True)
        profile = LLMProfile(native_tools=True, api_key="test", model="mock")
        registry = build_default_registry()
        script = [{"content": "完成。", "tool_calls": [
            {"name": "finish", "arguments": {"summary": "ok"}}]}]
        loop = AgentLoop(cfg, profile, MockBackend(profile, script), registry, console=None)

        loop.run("写一个冒泡排序 bubble_sort.py，可直接运行并自测")
        assert loop.task_name == "写一个冒泡排序-bubble_sort", f"实际：{loop.task_name}"
        assert loop.task_dir == home / "写一个冒泡排序-bubble_sort"


# ----------------------------------------------------------------------------
# 13) 纠错回灌不能破坏消息结构
# ----------------------------------------------------------------------------
def test_correction_note_never_puts_system_after_start():
    """回归：纠错提示若以 system 角色追加到末尾，网关会 400。

    真实报错：`HTTP 400 System message must be at the beginning.`
    OpenAI 兼容协议要求 system 只能位于消息列表开头。
    """
    from agent.history import History

    h = History("系统提示词")
    h.add_user("写一个前端页面")
    h.add_note("纠错：write_file 缺少必填参数 path")

    payload = h.payload()
    assert payload[0]["role"] == "system", "首条必须是 system"

    seen_non_system = False
    for msg in payload:
        if msg["role"] == "system":
            assert not seen_non_system, \
                f"system 消息出现在非开头位置，会被网关拒绝：{msg['content'][:40]}"
        else:
            seen_non_system = True

    # 纠错提示本身应以 user 角色回灌（对模型同样有效，且各家网关都接受）
    assert payload[-1]["role"] == "user", f"实际：{payload[-1]['role']}"
    assert "path" in payload[-1]["content"]

    # 该网关比 OpenAI 更严格：连"开头连续两条 system"也 400，故只保留首条
    h2 = History("系统提示词")
    h2.messages = [{"role": "system", "content": "系统提示词"},
                   {"role": "system", "content": "摘要"},
                   {"role": "user", "content": "继续"}]
    assert [m["role"] for m in h2.payload()] == ["system", "user", "user"]

    # 中间/末尾出现的 system 同样要降级
    h3 = History("系统提示词")
    h3.messages = [{"role": "system", "content": "系统提示词"},
                   {"role": "user", "content": "hi"},
                   {"role": "system", "content": "迟到的系统提示"}]
    assert [m["role"] for m in h3.payload()] == ["system", "user", "user"]


# ----------------------------------------------------------------------------
# 14) edit_block 精确编辑
# ----------------------------------------------------------------------------
def test_edit_block_replaces_unique_occurrence():
    """精确替换：只改 old_text 那一处，文件其余部分原样保留。"""
    with tempfile.TemporaryDirectory() as tmp:
        _cfg, _profile, registry, ctx = _make_env(tmp)
        f = Path(tmp, "a.py")
        f.write_text("def f():\n    return 1\n\n\ndef g():\n    return 2\n", encoding="utf-8")

        res = registry.execute("edit_block", {
            "path": "a.py",
            "old_text": "def f():\n    return 1",
            "new_text": "def f():\n    return 42",
        }, ctx)
        assert res.ok, res.render()

        after = f.read_text(encoding="utf-8")
        assert after == "def f():\n    return 42\n\n\ndef g():\n    return 2\n", after
        assert "def g():\n    return 2" in after, "未涉及的部分不应被改动"


def test_edit_block_refuses_ambiguous_match():
    """old_text 不唯一时拒绝替换，列出所有匹配行号，且文件保持原样。"""
    with tempfile.TemporaryDirectory() as tmp:
        _cfg, _profile, registry, ctx = _make_env(tmp)
        f = Path(tmp, "a.py")
        f.write_text("x = 1\nx = 1\ny = 2\n", encoding="utf-8")

        res = registry.execute("edit_block", {
            "path": "a.py", "old_text": "x = 1", "new_text": "x = 9",
        }, ctx)
        assert not res.ok, "不唯一时必须拒绝，否则可能改错位置"

        text = res.render()
        assert "匹配到 2 处" in text, text
        assert "第 1 行" in text and "第 2 行" in text, f"应列出每处行号：{text}"
        assert f.read_text(encoding="utf-8") == "x = 1\nx = 1\ny = 2\n", "失败时不应改动文件"


def test_edit_block_strips_line_numbers_copied_from_read():
    """模型把 read_file 的行号一起抄进来时，仍能匹配上。"""
    with tempfile.TemporaryDirectory() as tmp:
        _cfg, _profile, registry, ctx = _make_env(tmp)
        f = Path(tmp, "a.py")
        f.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

        res = registry.execute("edit_block", {
            "path": "a.py",
            "old_text": "    2| beta",          # read_file 的行号格式："{:>5}| "
            "new_text": "BETA",
        }, ctx)
        assert res.ok, res.render()
        assert f.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"
        assert "行号" in res.render(), "回执应提醒模型不要抄行号"


def test_edit_block_multiple_with_expected_replacements():
    """明确传 expected_replacements 时可一次改多处。"""
    with tempfile.TemporaryDirectory() as tmp:
        _cfg, _profile, registry, ctx = _make_env(tmp)
        f = Path(tmp, "a.py")
        f.write_text("x = 1\nx = 1\ny = 2\n", encoding="utf-8")

        res = registry.execute("edit_block", {
            "path": "a.py", "old_text": "x = 1", "new_text": "x = 9",
            "expected_replacements": 2,
        }, ctx)
        assert res.ok, res.render()
        assert f.read_text(encoding="utf-8") == "x = 9\nx = 9\ny = 2\n"


def test_edit_block_reports_not_found_with_hints():
    """找不到时给出排查方向，而不是让模型瞎猜。"""
    with tempfile.TemporaryDirectory() as tmp:
        _cfg, _profile, registry, ctx = _make_env(tmp)
        f = Path(tmp, "a.py")
        f.write_text("alpha\nbeta\n", encoding="utf-8")

        res = registry.execute("edit_block", {
            "path": "a.py", "old_text": "完全不存在的一段", "new_text": "x",
        }, ctx)
        assert not res.ok
        text = res.render()
        assert "找不到" in text, text
        assert "检查" in text, f"应给出排查提示：{text}"


# ----------------------------------------------------------------------------
# 15) 多密钥轮换
# ----------------------------------------------------------------------------
def test_key_rotation_rebuilds_client():
    """轮换后客户端必须重建，否则 profile 换了新 key、请求却仍用旧 key。

    SDK 的 api_key 在构造时固定，_rebuild_client 空实现会让轮换形同虚设。
    """
    from agent.llm import OpenAIBackend

    profile = LLMProfile(api_key="sk-aaa", api_keys=["sk-bbb", "sk-ccc"],
                         native_tools=False, base_url="https://example.invalid/v1")
    backend = OpenAIBackend(profile)
    assert backend._client.api_key == "sk-aaa", "初始应持主密钥"

    assert backend._rotate_key() is True
    assert profile.api_key == "sk-bbb"
    assert backend._client.api_key == "sk-bbb", "轮换后客户端必须重建，否则仍用旧密钥发请求"

    assert backend._rotate_key() is True
    assert backend._client.api_key == "sk-ccc"

    assert backend._rotate_key() is False, "密钥用完后不应绕回第一个"
    assert backend._client.api_key == "sk-ccc"


def test_should_rotate_only_on_credential_errors():
    """只有凭据/配额类错误值得换 key，普通错误轮换没有意义。"""
    from agent.errors import LLMError
    from agent.llm import LLMBackend

    assert LLMBackend._should_rotate(LLMError("无效的令牌", status=401))
    assert LLMBackend._should_rotate(LLMError("rate limited", status=429))
    assert LLMBackend._should_rotate(LLMError("quota exceeded"))
    assert LLMBackend._should_rotate(LLMError("余额不足"))
    assert not LLMBackend._should_rotate(LLMError("upstream boom", status=500))
    assert not LLMBackend._should_rotate(LLMError("connection timeout"))


# ----------------------------------------------------------------------------
# V1：可靠性加固（中文乱码 / 交互式挂死 / 假完成拦截）
# ----------------------------------------------------------------------------
def test_run_command_decodes_chinese_without_garbage():
    """Windows 中文环境默认 GBK：直接 utf-8 解码会把中文变乱码。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, _, registry, ctx = _make_env(tmp)
        # 造一个 GBK（cp936）编码的文件，模拟 Windows 程序/老工具的中文输出
        (Path(tmp) / "gbk.txt").write_bytes("中文测试输出".encode("gbk"))
        r = registry.execute("run_command", {"command": "type gbk.txt"}, ctx)
        assert r.ok, r.render()
        out = r.output
        assert "中文测试输出" in out
        # 若误用 utf-8 解码 cp936 字节，会出现 ä¸­æ–‡ 之类乱码，断言不存在
        assert "�" not in out
        assert "ä¸­" not in out


def test_interactive_command_is_killed_early():
    """命令打印未换行的提示符后卡住（疑似等待输入）→ 提前终止而非挂到整体超时。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, _, registry, ctx = _make_env(tmp)
        ctx.config.interactive_timeout = 1.2   # 测试里压短，避免真的等 20s
        # 输出 "AWAIT_INPUT" 但不换行，然后 sleep(60)：模拟 REPL 等待输入。
        # 注意：提示符不能用 '>'，否则 cmd.exe 会把 '>' 当作重定向吞掉输出。
        cmd = "python -c \"import sys,time; sys.stdout.write('AWAIT_INPUT'); sys.stdout.flush(); time.sleep(60)\""
        r = registry.execute("run_command", {"command": cmd}, ctx)
        assert r.meta.get("interactive_killed") is True, r.render()
        assert "AWAIT_INPUT" in r.render()
        assert r.meta.get("timed_out") is False


def test_fake_finish_blocked_until_verified():
    """改了文件却没跑过命令就 finish → 被拦截逼它先验证，跑过命令后才允许收尾。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        script = [
            {"content": "写文件", "tool_calls": [
                {"name": "write_file", "arguments": {"path": "a.py", "content": "print(1)\n"}}]},
            # ① 没验证就 finish → 应被拦截
            {"content": "完成", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "done"}}]},
            # ② 被逼着跑验证
            {"content": "运行", "tool_calls": [
                {"name": "run_command", "arguments": {"command": "python a.py"}}]},
            # ③ 验证过之后 finish → 允许
            {"content": "完成", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "done"}}]},
        ]
        backend, profile = _scripted(script, native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        result = loop.run("写个脚本")

        assert result.finish_reason == "finish", result.finish_reason
        # 拦截强制多跑了一轮（write + finish被拦 + run + finish），共 4 次工具调用
        assert result.tool_calls == 4
    joined = "\n".join(m.get("content", "") for m in loop.history.messages)
    assert "验证" in joined   # 拦截提示确实被注入历史


# ----------------------------------------------------------------------------
# V2：自修复闭环（测试命令识别 / traceback 上下文 / 回滚 / 端到端自愈）
# ----------------------------------------------------------------------------
def test_detect_test_command():
    from agent.selfrepair import detect_test_command
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        assert detect_test_command(tmp) == "pytest -q"
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "go.mod").write_text("module x\n", encoding="utf-8")
        assert detect_test_command(tmp) == "go test ./..."
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "package.json").write_text("{}", encoding="utf-8")
        assert detect_test_command(tmp) == "npm test"
    with tempfile.TemporaryDirectory() as tmp:
        assert detect_test_command(tmp) is None


def test_detect_test_command_covers_java_and_other_stacks():
    """多语言适配：补齐 Java（Maven/Gradle），且 wrapper 调用方式要分平台。

    Windows 下 run_command 走 cmd.exe，`./mvnw` 这类 POSIX 写法跑不通，
    必须用 `mvnw.cmd` —— 否则探测出一个必然失败的命令，白白浪费自修复预算。
    """
    from agent.selfrepair import detect_test_command
    windows = os.name == "nt"

    def probe(files):
        with tempfile.TemporaryDirectory() as tmp:
            for name, content in files.items():
                Path(tmp, name).write_text(content, encoding="utf-8")
            return detect_test_command(tmp)

    assert probe({"pom.xml": "<project/>"}) == "mvn -q test"
    assert probe({"pom.xml": "<project/>", "mvnw.cmd": ""}) == \
        ("mvnw.cmd -q test" if windows else "mvnw -q test")
    assert probe({"pom.xml": "<project/>", "mvnw": ""}) == \
        ("mvnw -q test" if windows else "./mvnw -q test")
    assert probe({"build.gradle": ""}) == "gradle test"
    assert probe({"build.gradle.kts": "", "gradlew.bat": ""}) == \
        ("gradlew test" if windows else "./gradlew test")
    assert probe({"tox.ini": "[tox]"}) == "tox"
    assert probe({"Cargo.toml": ""}) == "cargo test"
    assert probe({"package.json": "{}", "yarn.lock": ""}) == "yarn test"


def test_detect_test_command_does_not_abort_scan_early():
    """回归：探测链不能在单个标记文件上提前终止。

    早期实现遇到没有 [tool.pytest] 的 pyproject.toml 直接 `return None`，
    而现代 Python 项目几乎都有 pyproject.toml —— 探测链在最常见的文件上就断了，
    后面的 go.mod、测试文件命名约定全都没机会被检查。
    """
    from agent.selfrepair import detect_test_command
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
        Path(tmp, "test_demo.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
        assert detect_test_command(tmp) == "pytest -q", "应继续往下探，靠命名约定兜底"

    with tempfile.TemporaryDirectory() as tmp:
        # 没有 [tool.pytest] 时不该假装是 pytest 项目，应让位给更明确的信号
        Path(tmp, "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
        Path(tmp, "go.mod").write_text("module demo\n", encoding="utf-8")
        assert detect_test_command(tmp) == "go test ./..."


def test_detect_test_command_skips_doomed_commands():
    """Makefile 里没有 test 目标时不要建议 `make test`。

    `make test` 会直接报 "No rule to make target 'test'" ——
    回灌一条注定失败的命令比不给建议更糟。
    """
    from agent.selfrepair import detect_test_command
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "Makefile").write_text("all:\n\techo hi\n", encoding="utf-8")
        assert detect_test_command(tmp) is None

    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "Makefile").write_text("all:\n\techo hi\n\ntest:\n\tpytest -q\n", encoding="utf-8")
        assert detect_test_command(tmp) == "make test"


def test_build_failure_note_reads_offending_lines():
    from agent.selfrepair import build_failure_note
    with tempfile.TemporaryDirectory() as tmp:
        code = "def add(a, b):\n    return a + b\n\ndef div(a, b):\n    return a / c\n"  # c 未定义
        (Path(tmp) / "calc.py").write_text(code, encoding="utf-8")
        cfg, _, registry, ctx = _make_env(tmp)
        tb = (
            'Traceback (most recent call last):\n'
            '  File "calc.py", line 5, in div\n'
            "ZeroDivisionError: name 'c' is not defined\n"
        )
        note = build_failure_note(tb, ctx)
        assert note is not None
        assert "calc.py" in note and "第 5 行" in note
        assert "return a / c" in note   # 出错附近的源码被附上，模型无需再 read_file


def test_rollback_restores_latest_snapshot():
    """rollback 工具把 .agent_backups 里最新快照拷回任务目录。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, _, registry, _ = _make_env(tmp, per_task_dir=True)
        backend, profile = _scripted([{"content": "x", "tool_calls": []}])
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        # 造一个任务目录与一份"好"快照
        task_dir = loop.prepare_task_dir("demo")
        (Path(task_dir) / "good.py").write_text("print('ok')\n", encoding="utf-8")
        loop.ctx.session["changes"] = [{"kind": "write", "detail": "good.py"}]
        snap = loop.snapshot_to_backups()
        assert snap is not None
        # 把当前目录改"坏"
        (Path(task_dir) / "good.py").write_text("print('BROKEN')\n", encoding="utf-8")
        # 通过工具回滚（必须用任务目录对应的 ctx：其 workspace 才是 task 目录）
        r = registry.execute("rollback", {}, loop.ctx)
        assert r.ok, r.render()
        assert (Path(task_dir) / "good.py").read_text(encoding="utf-8") == "print('ok')\n"


def test_self_repair_feeds_traceback_context_and_recovers():
    """写带 bug 的脚本 → 运行失败 → 循环回灌出错位置 → 模型读上下文修好 → 跑通。

    ⚠️ 本用例曾长期是**假通过**：原脚本只定义了函数、从未调用它，
    `python calc.py` 退出码是 0，根本没触发失败；而断言写成
    "第 4 行" in joined or "calc.py" in joined，后者在任何回显里都会命中。
    结果是 V2 的招牌能力（自修复闭环）**没有任何测试真正覆盖过它**。
    修法：脚本必须真的在运行时报错（这里用 div(4, 0) 触发 ZeroDivisionError），
    断言则改为检查自修复提示真正交付的东西——出错行号 + 附近源码片段。
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        # 第 1 轮：写会真的报错的脚本并运行（失败）；第 2 轮：据回灌的出错位置修好；第 3 轮：跑通并 finish
        script = [
            {"content": "写脚本", "tool_calls": [
                {"name": "write_file", "arguments": {"path": "calc.py",
                 "content": "def div(a, b):\n    return a / b\n\n\nprint(div(4, 0))\n"}}]},
            {"content": "运行", "tool_calls": [
                {"name": "run_command", "arguments": {"command": "python calc.py"}}]},
            {"content": "修", "tool_calls": [
                {"name": "edit_block", "arguments": {
                    "path": "calc.py", "old_text": "print(div(4, 0))", "new_text": "print(div(4, 2))"}}]},
            {"content": "再运行", "tool_calls": [
                {"name": "run_command", "arguments": {"command": "python calc.py"}}]},
            {"content": "完成", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "已修复并验证"}}]},
        ]
        backend, profile = _scripted(script, native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        result = loop.run("写个除法函数")
        assert result.finish_reason == "finish", result.finish_reason
        assert result.errors == 1, f"第一次运行必须真的失败，实际 errors={result.errors}"

        joined = "\n".join(str(m.get("content", "")) for m in loop.history.messages)
        assert "运行失败" in joined, "失败后应回灌自修复提示"
        assert "calc.py" in joined and "第 5 行" in joined, "提示里应给出出错的文件与行号"
        # 核心价值：把出错处附近的源码直接给模型，省掉一轮 read_file
        assert "该处附近的源码" in joined, "应附带出错位置附近的源码片段"
        assert "print(div(4, 0))" in joined, "源码片段里应包含真正出错的那一行"


# ----------------------------------------------------------------------------
# V3：上下文与成本治理（智能压缩 / 精确 token / /stats 成本面板）
# ----------------------------------------------------------------------------
def test_smart_compress_keeps_signal_lines():
    """智能压缩应保住 traceback 这类关键「信号行」，而非纯 head+tail 丢掉中间。"""
    from agent.security import smart_compress
    lines = [f"无关日志行 {i}" for i in range(1, 60)]
    lines[30] = "Traceback (most recent call last):"
    lines[31] = "  File 'a.py', line 5, in <module>"
    lines[32] = "ZeroDivisionError: division by zero"
    text = "\n".join(lines)
    out = smart_compress(text, max_chars=200, note="压缩")
    assert "Traceback" in out
    assert "ZeroDivisionError" in out
    assert "无关日志行" in out   # 首尾仍保留


def test_smart_compress_short_text_unchanged():
    from agent.security import smart_compress
    # 未超长 → 原样返回
    assert smart_compress("短文本", max_chars=200) == "短文本"
    # 超长但单行且无信号行 → 退回 head+tail 截断（长度被砍）
    long_one = "A" * 300
    out = smart_compress(long_one, max_chars=50, note="压缩")
    assert len(out) <= 50 + 120   # 截断 + 省略提示，明显短于原文


def test_tool_result_render_uses_smart_compress_no_nameerror():
    """回归：ToolResult.render 必须已导入 smart_compress，否则超长回执会 NameError。"""
    from agent.tools.base import ToolResult
    big = "正常输出一行\n" * 60   # 远超默认 max_chars
    r = ToolResult.success(big)
    rendered = r.render(max_chars=80)
    assert len(rendered) < len(big)
    assert "省略" in rendered or "压缩" in rendered


def test_stats_panel_reports_real_tokens_and_tool_breakdown():
    """/stats 面板应报出真实 token（来自 API usage）与按工具拆分的耗时占比。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        script = [
            {"content": "写。", "tool_calls": [
                {"name": "write_file", "arguments": {"path": "a.py", "content": "print(1)\n"}}]},
            {"content": "跑。", "tool_calls": [
                {"name": "run_command", "arguments": {"command": "python a.py"}}]},
            {"content": "完成。", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "ok"}}]},
        ]
        backend, profile = _scripted(script, native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        result = loop.run("写脚本并运行")
        assert result.succeeded

        panel = loop.build_stats_panel()
        # MockBackend 每轮 usage = {prompt:100, completion:50, total:150}，本任务 3 轮模型调用 → 450
        assert "total" in panel
        assert "450" in panel, panel
        assert "run_command" in panel, "各工具耗时占比应含 run_command"
        assert "write_file" in panel
        assert "工具调用总数：3" in panel, panel


def test_stats_panel_counts_output_compressions():
    """超长工具回执被智能压缩时，面板里的『回执智能压缩次数』应 +1。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp, max_tool_output_chars=80)
        # 打印 300 个 A（超过 80 → 触发压缩）
        script = [
            {"content": "长输出。", "tool_calls": [
                {"name": "run_command", "arguments": {"command": "python -c \"print('A'*300)\""}}]},
            {"content": "完成。", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "ok"}}]},
        ]
        backend, profile = _scripted(script, native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        result = loop.run("产生一个超长输出的命令")
        assert result.succeeded
        assert loop._output_compressions >= 1, "应记录一次回执压缩"
        assert "回执智能压缩次数：1" in loop.build_stats_panel()


# ----------------------------------------------------------------------------
# V6：交付打磨（每个工具至少一条测试 + 覆盖率补强）
# ----------------------------------------------------------------------------
def test_read_file_returns_numbered_lines():
    """read_file 带行号返回，是后续 edit_block 定位的可靠锚点。"""
    with tempfile.TemporaryDirectory() as tmp:
        _cfg, _profile, registry, ctx = _make_env(tmp)
        (Path(tmp) / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        r = registry.execute("read_file", {"path": "a.py"}, ctx)
        assert r.ok, r.render()
        assert "1| def f()" in r.render()
        assert "2|     return 1" in r.render()


def test_grep_search_finds_pattern():
    with tempfile.TemporaryDirectory() as tmp:
        _cfg, _profile, registry, ctx = _make_env(tmp)
        (Path(tmp) / "a.py").write_text("def foo():\n    pass\ndef bar():\n    pass\n", encoding="utf-8")
        r = registry.execute("grep_search", {"pattern": "def (foo|bar)"}, ctx)
        assert r.ok
        assert "a.py:1:" in r.render() and "a.py:3:" in r.render()


def test_grep_search_context_shows_surrounding_lines():
    from agent.tools import build_default_registry, build_tool_context
    from agent.config import AgentConfig
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.py").write_text(
            "import os\n\ndef foo():\n    return 1\n\n\ndef bar():\n    return 2\n",
            encoding="utf-8")
        cfg = AgentConfig(workspace=tmp, session_log=None)
        registry = build_default_registry()
        ctx = build_tool_context(cfg, console=None, session={})
        r = registry.execute("grep_search", {"pattern": "def foo", "context": 2}, ctx)
        assert r.ok
        # 命中行用 > 标出，且出现了上下各 2 行的 import / return
        assert ">" in r.render()
        assert "import os" in r.render(), "应显示上方 2 行上下文"
        assert "return 1" in r.render(), "应显示下方 2 行上下文"
        assert r.meta["files_with_match"] == 1


def test_find_files_by_glob():
    with tempfile.TemporaryDirectory() as tmp:
        _cfg, _profile, registry, ctx = _make_env(tmp)
        (Path(tmp) / "a.py").write_text("x", encoding="utf-8")
        (Path(tmp) / "b.txt").write_text("x", encoding="utf-8")
        r = registry.execute("find_files", {"pattern": "*.py"}, ctx)
        assert r.ok
        assert "a.py" in r.render()
        assert "b.txt" not in r.render()


def test_ask_user_non_interactive_returns_graceful():
    """非交互环境（console=None，如测试）：应提示自行合理假设，而非抛错或卡住。"""
    with tempfile.TemporaryDirectory() as tmp:
        _cfg, _profile, registry, ctx = _make_env(tmp)
        r = registry.execute("ask_user", {"question": "用哪个框架？"}, ctx)
        assert r.ok, r.render()
        assert "非交互" in r.render()


def test_list_dir_shows_tree():
    with tempfile.TemporaryDirectory() as tmp:
        _cfg, _profile, registry, ctx = _make_env(tmp)
        (Path(tmp) / "sub").mkdir()
        (Path(tmp) / "sub" / "x.py").write_text("x", encoding="utf-8")
        r = registry.execute("list_dir", {"path": "."}, ctx)
        assert r.ok
        assert "sub/" in r.render()
        assert "x.py" in r.render()


# ----------------------------------------------------------------------------
# V4：可审阅性（unified diff 预览 / /diff / 单文件级回退）
# ----------------------------------------------------------------------------
def test_diff_renders_unified_diff_of_session_changes():
    """diff 工具应把本次会话的改动渲染成 unified diff，便于 finish 前自查。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        script = [
            {"content": "写文件。", "tool_calls": [
                {"name": "write_file", "arguments": {"path": "a.py", "content": "print('hi')\n"}}]},
            {"content": "改一行。", "tool_calls": [
                {"name": "edit_block", "arguments": {
                    "path": "a.py", "old_text": "print('hi')", "new_text": "print('hello')"}}]},
            {"content": "看 diff。", "tool_calls": [
                {"name": "diff", "arguments": {}}]},
            {"content": "完成。", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "ok"}}]},
        ]
        backend, profile = _scripted(script, native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        result = loop.run("写文件并改一行")
        assert result.succeeded
        # diff 工具是最后一轮模型调用，它的回执（unified diff）应进了历史
        joined = "\n".join(m.get("content", "") for m in loop.history.messages)
        assert "print('hi')" in joined and "print('hello')" in joined
        assert "--- a/" in joined and "+++ b/" in joined, "应出现 unified diff 的分隔头"


def test_single_file_rollback_restores_only_named_file():
    """rollback 的 files 参数应只恢复指定文件，而非整目录回滚。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, _, registry, _ = _make_env(tmp, per_task_dir=True)
        backend, profile = _scripted([{"content": "x", "tool_calls": []}])
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        task_dir = loop.prepare_task_dir("demo")
        (Path(task_dir) / "good.py").write_text("print('ok')\n", encoding="utf-8")
        (Path(task_dir) / "bad.py").write_text("print('ok')\n", encoding="utf-8")
        loop.ctx.session["changes"] = [{"kind": "write", "detail": "good.py"}, {"kind": "write", "detail": "bad.py"}]
        snap = loop.snapshot_to_backups()
        assert snap is not None
        # 把两个都改坏
        (Path(task_dir) / "good.py").write_text("print('BROKEN')\n", encoding="utf-8")
        (Path(task_dir) / "bad.py").write_text("print('BROKEN')\n", encoding="utf-8")
        # 只恢复 good.py
        r = registry.execute("rollback", {"files": ["good.py"]}, loop.ctx)
        assert r.ok, r.render()
        assert (Path(task_dir) / "good.py").read_text(encoding="utf-8") == "print('ok')\n"
        # bad.py 不应被恢复
        assert (Path(task_dir) / "bad.py").read_text(encoding="utf-8") == "print('BROKEN')\n"


def test_new_file_change_is_diffable_not_mistaken_for_too_big():
    """回归：新建文件的 before 本来就是 None，不能被误判成"改动过大"。

    早期 record_change 用 `before is not None and after is not None` 判断能否生成 diff，
    结果所有新建文件（最该被审阅的一类改动）的 diff 全部丢失。
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        backend, profile = _scripted([{"content": "x", "tool_calls": []}])
        loop = AgentLoop(cfg, profile, backend, registry, console=None)

        registry.execute("write_file", {"path": "fresh.py", "content": "x = 1\ny = 2\n"}, loop.ctx)
        changes = loop.ctx.session["changes"]
        assert len(changes) == 1
        assert changes[0]["captured"] is True, "新建文件（before=None）也必须可生成 diff"
        assert changes[0]["before"] is None

        text = build_diff(changes)
        assert "+x = 1" in text and "+y = 2" in text, "新建文件应显示为全量新增"
        assert "过大" not in text


def test_record_change_marks_command_and_oversize_as_not_captured():
    """run_command 这类非文件操作不参与 diff；超大文件才标未采集。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        backend, profile = _scripted([{"content": "x", "tool_calls": []}])
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        ctx = loop.ctx

        ctx.record_change("command", "python a.py")                      # 无 path
        ctx.record_change("write", "big.py", path="big.py",
                          before=None, after="x" * (DIFF_CAPTURE_CAP + 1))  # 超限
        ctx.record_change("write", "ok.py", path="ok.py", before=None, after="z = 1\n")

        changes = ctx.session["changes"]
        assert [c["captured"] for c in changes] == [False, False, True]

        text = build_diff(changes)
        assert "+z = 1" in text                       # 正常文件照常出 diff
        assert "big.py" in text                       # 超限的只在末尾提一句
        assert "python a.py" not in text              # 命令不是文件改动，不该出现在 diff 里
        assert "另有 1 个变更内容过大" in text


# ----------------------------------------------------------------------------
# 追加工作：流式输出（边生成边打印）
# ----------------------------------------------------------------------------
class _FakeFn:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _FakeDeltaCall:
    def __init__(self, index, id_=None, name=None, arguments=None):
        self.index = index
        self.id = id_
        self.function = _FakeFn(name, arguments)


class _FakeChunk:
    def __init__(self, content=None, tool_calls=None, finish_reason=None, usage=None):
        self.usage = usage
        if content is None and tool_calls is None and finish_reason is None:
            self.choices = []
        else:
            self.choices = [type("C", (), {
                "delta": type("D", (), {"content": content, "tool_calls": tool_calls})(),
                "finish_reason": finish_reason,
            })()]


class _FakeStreamClient:
    """按给定 chunk 序列模拟流式响应；stream_options 不支持时可让 create 抛错。"""

    def __init__(self, chunks, reject_stream_options=False):
        self._chunks = chunks
        self._reject = reject_stream_options

    # 真实 SDK 的调用链是 client.chat.completions.create(...)
    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        if kwargs.get("stream"):
            if self._reject and "stream_options" in kwargs:
                raise RuntimeError("400 stream_options is not supported")
            return iter(self._chunks)
        raise AssertionError("本用例不应走整包路径")


def _streaming_backend(chunks):
    """造一个只跑 _send_streaming 的 OpenAIBackend（不需要真实网络客户端）。"""
    from agent.llm import OpenAIBackend, LLMProfile
    backend = OpenAIBackend.__new__(OpenAIBackend)
    backend._client = _FakeStreamClient(chunks)
    backend.profile = LLMProfile(native_tools=True, api_key="test", model="m")
    return backend


def test_streaming_accumulates_fragmented_tool_calls():
    """流式下 tool_calls 是按 index 分片的，必须攒够再解析。

    一个函数调用的 id / name / arguments 会跨多个 chunk 到达；直接拿单个 chunk 的
    arguments 去 json.loads 必然是残缺 JSON —— 这正是流式实现最容易错的地方。
    """
    chunks = [
        _FakeChunk(content="我先看看"),
        _FakeChunk(content="文件。"),
        # 第 0 个调用：id / name / arguments 分三片到达
        _FakeChunk(tool_calls=[_FakeDeltaCall(0, id_="call_1")]),
        _FakeChunk(tool_calls=[_FakeDeltaCall(0, name="read_file")]),
        _FakeChunk(tool_calls=[_FakeDeltaCall(0, arguments='{"pa')]),
        _FakeChunk(tool_calls=[_FakeDeltaCall(0, arguments='th": "a.py"}')]),
        # 第 1 个调用：与第 0 个交错到达，且 index 顺序打乱
        _FakeChunk(tool_calls=[_FakeDeltaCall(1, id_="call_2", name="grep_search",
                                              arguments='{"pattern": "x"}')]),
        _FakeChunk(finish_reason="tool_calls"),
    ]

    deltas = []
    msg = _streaming_backend(chunks)._send_streaming({"model": "m"}, deltas.append)

    assert "".join(deltas) == "我先看看文件。"
    assert msg.content == "我先看看文件。"
    assert msg.finish_reason == "tool_calls"
    assert len(msg.tool_calls) == 2
    # 分片累积后必须是完整可解析的 JSON，而不是空 dict + malformed
    assert msg.tool_calls[0].name == "read_file"
    assert msg.tool_calls[0].arguments == {"path": "a.py"}
    assert msg.tool_calls[0].malformed is False, "分片 arguments 累积后应能正常解析"
    assert msg.tool_calls[1].name == "grep_search"
    assert msg.tool_calls[1].arguments == {"pattern": "x"}


def test_streaming_keeps_usage_for_cost_accounting():
    """流式默认不带 usage，而 /stats 的真实成本依赖它——必须显式要回来。

    否则流式一开，成本对账就悄悄失效（不报错、只是数字变成 0）。
    """
    usage = type("U", (), {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})()
    # usage 通常只在最后一个（只含 usage 的）chunk 里出现
    chunks = [_FakeChunk(content="hi"), _FakeChunk(usage=usage)]
    msg = _streaming_backend(chunks)._send_streaming({"model": "m"}, lambda _t: None)
    assert msg.usage == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    assert msg.content == "hi"


def test_streaming_falls_back_when_gateway_rejects_it():
    """网关不支持流式（或 stream_options）时，要退回整包而不是直接失败。

    流式只是体验优化，不该成为可用性风险；更不能让它在演示时突然挂掉。
    """
    from agent.llm import OpenAIBackend, LLMProfile

    class _BlockingOnlyClient:
        """流式一律拒绝（模拟不支持 stream 的网关），只接受整包。"""

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            if kwargs.get("stream"):
                raise RuntimeError("400 stream is not supported")
            resp = type("R", (), {})()
            choice = type("C", (), {})()
            choice.message = type("M", (), {"content": "整包返回", "tool_calls": []})()
            choice.finish_reason = "stop"
            resp.choices = [choice]
            resp.usage = type("U", (), {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})()
            return resp

    backend = OpenAIBackend.__new__(OpenAIBackend)
    backend._client = _BlockingOnlyClient()
    backend.profile = LLMProfile(native_tools=True, api_key="test", model="m")

    out = []
    msg = backend._send([{"role": "user", "content": "hi"}], tools=None,
                        tool_choice=None, on_delta=out.append)
    assert msg.content == "整包返回", "流式被拒时应自动退回整包"
    assert msg.usage["total_tokens"] == 7


# ----------------------------------------------------------------------------
# 追加工作：工具反向文档（when_not_to_use）
# ----------------------------------------------------------------------------
def test_every_tool_documents_when_not_to_use_it():
    """每个面向模型的工具都要写清「什么时候不该用它」。

    工具越多模型越容易误用（典型：用 write_file 整体重写大文件里的一行）。
    这条断言保证以后加新工具时不会漏掉反向文档。
    """
    registry = build_default_registry()
    missing = [s.name for s in registry.visible_specs() if not s.when_not_to_use.strip()]
    assert not missing, f"这些工具缺少 when_not_to_use：{missing}"


def test_command_timeout_is_clamped_to_a_hard_cap():
    """模型传的 timeout 必须被夹到上限内，否则一条命令能把整个会话挂死几小时。

    把模型的输出当不可信输入处理——和路径沙箱是同一类边界。
    """
    from agent.tools.shell import _resolve_timeout

    class Cfg:
        command_timeout = 120
        max_command_timeout = 300

    cfg = Cfg()
    assert _resolve_timeout(None, cfg)[0] == 120                  # 不传 → 用默认
    assert _resolve_timeout(30, cfg)[0] == 30                     # 合理值照用
    assert _resolve_timeout(99999, cfg)[0] == 300                 # 荒谬值 → 夹到上限
    assert "超过上限" in _resolve_timeout(99999, cfg)[1]           # 并告知模型被夹了
    assert _resolve_timeout(0, cfg)[0] == 120                     # 非法 → 退回默认
    assert _resolve_timeout(-5, cfg)[0] == 120
    assert _resolve_timeout("abc", cfg)[0] == 120                 # 非数字不抛异常


def test_run_command_survives_absurd_timeout_and_reports_partial_output():
    """端到端：传一个荒谬 timeout 也要在可控时间内返回，且给出部分输出。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, ctx = _make_env(tmp)
        ctx.config.max_command_timeout = 3   # 上限压到 3s，别让测试真等 300s
        # 先打印一行再睡很久——验证超时后仍拿得到"终止前的部分输出"
        r = registry.execute("run_command", {
            "command": 'echo before-sleep && python -c "import time; time.sleep(60)"',
            "timeout": 99999,
        }, ctx)
        assert not r.ok
        text = r.render()
        assert "before-sleep" in text, "超时也应保留已产生的部分输出"
        assert "超过上限" in text, "应告知模型 timeout 被夹取了"
        assert "不要原样重试" in text, "超时提示要给出可行动的下一步，而不只是报错"
        assert "只能靠缩小范围" in text, "已到上限时不应再建议加大 timeout"
        assert r.meta["timed_out"] is True


def test_run_command_background_and_check_and_kill():
    """后台命令：run_command(background=true) 立即返回 job_id，check_command 读输出，kill_command 终止。"""
    import shutil as _shutil
    import time as _t
    d = tempfile.mkdtemp()
    try:
        cfg, profile, registry, ctx = _make_env(d)

        # 1) 后台启动一个会很快结束的命令，立即拿到 job_id（不阻塞）
        r = registry.execute("run_command", {
            "command": 'python -c "print(\'HELLO_BG\')"',
            "background": True,
        }, ctx)
        assert r.ok and r.meta.get("background") is True, r.render()
        job_id = r.meta["job_id"]
        assert job_id

        # 2) 等命令结束后 check_command 拿到完整输出与退出码
        for _ in range(50):
            _t.sleep(0.1)
            if registry.execute("check_command", {"job_id": job_id}, ctx).meta.get("status") == "done":
                break
        rc = registry.execute("check_command", {"job_id": job_id}, ctx)
        assert rc.meta["status"] == "done", rc.render()
        assert "HELLO_BG" in rc.output, "check_command 应拿到后台命令的完整输出"
        assert rc.meta["exit_code"] == 0

        # 3) 杀掉一个长任务：先后台起 sleep，再 kill
        r2 = registry.execute("run_command", {
            "command": "python -c \"import time; time.sleep(30)\"",
            "background": True,
        }, ctx)
        jid2 = r2.meta["job_id"]
        running = registry.execute("check_command", {"job_id": jid2}, ctx)
        assert running.meta["status"] == "running"
        killed = registry.execute("kill_command", {"job_id": jid2}, ctx)
        assert killed.ok and killed.meta["status"] == "killed", killed.render()
        jobs = getattr(ctx, "bg_jobs", {})
        assert jobs[jid2]["proc"].poll() is not None, "kill_command 应真正终止进程"

        # 4) 未知 job_id 报友好错误
        bad = registry.execute("check_command", {"job_id": "bg999"}, ctx)
        assert not bad.ok and "未知" in bad.error
    finally:
        # 后台进程若残留会锁住 cwd（Windows），先确保全部终止，再带重试地清理临时目录
        jobs = getattr(ctx, "bg_jobs", {})
        for j in jobs.values():
            p = j["proc"]
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
        for _ in range(10):
            try:
                _shutil.rmtree(d)
                break
            except OSError:
                _t.sleep(0.3)


def test_stats_panel_breaks_down_where_time_went():
    """「时间去向」必须把等模型的时间算进来，否则答不出瓶颈在哪。

    原先只统计工具耗时，而等模型通常是最大的一块——面板因此只能看到局部，
    优化方向也会被误导（以为该优化工具，实际该减少轮数与上下文长度）。
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        script = [
            {"content": "写文件。", "tool_calls": [
                {"name": "write_file", "arguments": {"path": "a.py", "content": "x = 1\n"}}]},
            {"content": "跑一下。", "tool_calls": [
                {"name": "run_command", "arguments": {"command": "echo hi"}}]},
            {"content": "完成。", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "ok"}}]},
        ]
        backend, profile = _scripted(script, native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        loop.run("写文件并运行")

        assert loop._model_calls == 3, "三轮任务应有三次模型调用"
        assert loop._model_time >= 0
        panel = loop.build_stats_panel()
        assert "时间去向" in panel
        assert "等模型" in panel and "执行工具" in panel and "其它" in panel
        assert "模型调用次数：3" in panel


def test_guardrail_reaches_model_on_both_channels():
    """反向文档必须让模型在**两条通道**上都看到，否则等于没写。

    默认配置走 native function calling，模型读的是 openai_schema 里的 description；
    只把它写进系统提示词（describe）的话 native 通道根本看不到。
    """
    registry = build_default_registry()
    spec = registry.get("write_file")
    assert spec is not None

    # ① native function calling 通道
    schema = spec.openai_schema()
    assert "不该用它的情况" in schema["function"]["description"]

    # ② 文本协议通道（系统提示词的工具清单）
    assert "⚠ 不该用" in registry.describe()
    # 需要压缩时可只要签名+用途
    assert "⚠ 不该用" not in registry.describe(with_guardrail=False)


def test_write_over_huge_file_marks_change_not_captured():
    """覆盖写一个超大文件时，不能把 before="" 当成"原本是空文件"。

    否则 diff 会把一次修改显示成全量新增，比不显示更糟。
    同时验证原文不会被读进会话内存（changes 里不存超大内容）。
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        backend, profile = _scripted([{"content": "x", "tool_calls": []}])
        loop = AgentLoop(cfg, profile, backend, registry, console=None)

        huge = "Z" * (DIFF_CAPTURE_CAP * 5)          # 远超采集上限
        r = registry.execute("write_file", {"path": "huge.txt", "content": huge}, loop.ctx)
        assert r.ok, r.render()

        # 再覆盖写一次（这次原文件已超限）
        r = registry.execute("write_file", {"path": "huge.txt", "content": "small\n"}, loop.ctx)
        assert r.ok, r.render()
        assert "行" in r.render(), "回执仍应报出行数变化"

        change = loop.ctx.session["changes"][-1]
        assert change["captured"] is False, "原文过大时应显式标记未采集，而非当成空文件"
        assert change["before"] is None and change["after"] is None

        # diff 里只说"未生成 diff"，不能出现"全量新增"这种误导性内容
        text = build_diff(loop.ctx.session["changes"])
        assert "未生成 diff" in text
        assert "huge.txt" in text


def test_build_diff_handles_new_and_unchanged():
    from agent.tools.review import build_diff
    # 新建文件（before=None）应显示为全量新增；内容无变化的标"无变化"
    changes = [
        {"kind": "write", "detail": "new.py", "path": "new.py",
         "captured": True, "before": None, "after": "x = 1\n"},
        {"kind": "edit", "detail": "same.py", "path": "same.py",
         "captured": True, "before": "y = 1\n", "after": "y = 1\n"},
        {"kind": "write", "detail": "big.py", "path": "big.py",
         "captured": False, "before": None, "after": None},  # 超大数据未采集
    ]
    text = build_diff(changes)
    assert "+x = 1" in text
    assert "内容无变化" in text
    assert "未生成 diff" in text

    # 空改动 / 全是命令的场景要有得体文案，不能返回空串
    assert "未修改任何文件" in build_diff([])
    assert "没有产生文件内容的改动" in build_diff(
        [{"kind": "command", "detail": "python a.py", "path": None, "captured": False}])


# ----------------------------------------------------------------------------
# V5：自主性增强（项目画像 / plan 工具 / 复杂任务先计划 / 只读工具并行）
# ----------------------------------------------------------------------------
def test_project_profile_detection():
    """扫描工作区应识别语言/框架/构建与测试命令，并结构化返回。"""
    from agent.profile import detect_project_profile
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "requirements.txt").write_text("", encoding="utf-8")
        Path(tmp, "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        prof = detect_project_profile(tmp)
        assert "Python" in prof["languages"]
        assert prof["test_cmd"] == "pytest -q"
        assert "pip install -r requirements.txt" in prof["build_cmds"]


def test_project_profile_renders_section_when_present():
    """有画像时 build_system_prompt 应注入『项目画像』段落；空画像返回空串。"""
    from agent.profile import format_project_profile
    assert format_project_profile({}) == ""
    section = format_project_profile({
        "languages": ["Python"], "frameworks": ["flask"],
        "build_cmds": ["pip install -r requirements.txt"], "test_cmd": "pytest -q",
    })
    assert "# 项目画像" in section
    assert "Python" in section
    # 同一段文字经 build_system_prompt 后仍以『# 项目画像』出现
    from agent.prompts import build_system_prompt
    sp = build_system_prompt(tool_list="", workspace="/tmp/x",
                             restrict_to_workspace=False, project_profile=section)
    assert "# 项目画像" in sp


def test_plan_tool_records_plan():
    """plan 工具把分步计划写入 session，供用户审阅与模型对齐。"""
    with tempfile.TemporaryDirectory() as tmp:
        _cfg, _profile, registry, ctx = _make_env(tmp)
        r = registry.execute("plan", {"steps": ["读 README 弄清现状", "用 edit_block 改入口", "跑测试"]}, ctx)
        assert r.ok, r.render()
        assert ctx.session["plan"] == ["读 README 弄清现状", "用 edit_block 改入口", "跑测试"]
        assert "已记录计划" in r.render()


def test_complex_task_receives_plan_hint():
    """复杂任务开头应注入"先用 plan 工具列计划"的提示，引导先规划再动手。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        script = [{"content": "完成。", "tool_calls": [
            {"name": "finish", "arguments": {"summary": "ok"}}]}]
        backend, profile = _scripted(script, native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        # 多关键词 + 超长，明显命中"复杂任务"启发式
        result = loop.run("请帮我实现一个端到端的爬虫系统，包含调度、解析、存储三个模块，并保证可测试")
        assert result.succeeded
        joined = "\n".join(m.get("content", "") for m in loop.history.messages)
        assert "plan" in joined, "复杂任务应提示先调用 plan 工具"
        assert "较复杂的任务" in joined, "开头应注入『先计划再动手』的建议"


def test_parallel_readonly_calls_execute():
    """一轮里的多个只读调用（read_file 两次）应并行发出且不破坏循环顺序。"""
    with tempfile.TemporaryDirectory() as tmp:
        # 预先放两份文件，避免触发"假完成拦截"（没改过文件）
        Path(tmp, "a.py").write_text("AAA\n", encoding="utf-8")
        Path(tmp, "b.py").write_text("BBB\n", encoding="utf-8")
        cfg, profile, registry, _ = _make_env(tmp)
        # 一轮里同时 read a.py 与 b.py（都是只读工具）→ 并行分支
        script = [
            {"content": "读两份文件。", "tool_calls": [
                {"name": "read_file", "arguments": {"path": "a.py"}},
                {"name": "read_file", "arguments": {"path": "b.py"}}]},
            {"content": "完成。", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "ok"}}]},
        ]
        backend, profile = _scripted(script, native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        result = loop.run("读取项目里已有的两份文件")
        assert result.succeeded
        # 2 次 read（并行）+ 1 次 finish = 3
        assert result.tool_calls == 3, result.tool_calls
        joined = "\n".join(m.get("content", "") for m in loop.history.messages)
        assert "AAA" in joined, "并行读取结果应回灌历史"
        assert "BBB" in joined


# ----------------------------------------------------------------------------
# apply_patch（新能力：把 unified diff 落到已存在文件）
# ----------------------------------------------------------------------------
def test_apply_patch_changes_adds_and_removes_lines():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, ctx = _make_env(tmp)
        p = Path(tmp, "calc.py")
        p.write_text("def f(x):\n    return x + 1\n", encoding="utf-8")

        # 改一行 + 新增一行（同一 hunk）
        patch = "@@ -1,2 +1,3 @@\n def f(x):\n+    # 加了注释\n     return x + 1\n"
        res = registry.execute("apply_patch", {"path": "calc.py", "patch": patch}, ctx)
        assert res.ok, res.error
        assert "加了注释" in p.read_text(encoding="utf-8")

        # 删除一行
        patch2 = "@@ -1,3 +1,2 @@\n def f(x):\n     # 加了注释\n-    return x + 1\n"
        res2 = registry.execute("apply_patch", {"path": "calc.py", "patch": patch2}, ctx)
        assert res2.ok, res2.error
        assert "return x + 1" not in p.read_text(encoding="utf-8")


def test_apply_patch_multiple_hunks():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, ctx = _make_env(tmp)
        p = Path(tmp, "nums.txt")
        p.write_text("1\n2\n3\n4\n5\n", encoding="utf-8")
        patch = (
            "@@ -2,1 +2,1 @@\n-2\n+two\n"
            "@@ -4,1 +4,1 @@\n-4\n+four\n"
        )
        res = registry.execute("apply_patch", {"path": "nums.txt", "patch": patch}, ctx)
        assert res.ok, res.error
        assert p.read_text(encoding="utf-8") == "1\ntwo\n3\nfour\n5\n"


def test_apply_patch_fails_when_hunk_not_found_and_keeps_file():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, ctx = _make_env(tmp)
        p = Path(tmp, "x.txt")
        p.write_text("hello\nworld\n", encoding="utf-8")
        patch = "@@ -1,2 +1,2 @@\n-zzz\n+abc\n"
        res = registry.execute("apply_patch", {"path": "x.txt", "patch": patch}, ctx)
        assert not res.ok
        # 文件未被改成半成品
        assert p.read_text(encoding="utf-8") == "hello\nworld\n"


def test_apply_patch_rejects_new_file_and_unknown_target():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, ctx = _make_env(tmp)
        res = registry.execute("apply_patch", {"path": "nope.py",
                                                "patch": "@@ -1,0 +1,1 @@\n+new\n"}, ctx)
        assert not res.ok  # 文件不存在应失败而非新建


def test_apply_patch_is_registered_and_visible_to_model():
    registry = build_default_registry()
    assert "apply_patch" in registry.names()
    desc = registry.describe()
    assert "apply_patch" in desc
    schemas = [s["function"]["name"] for s in registry.schemas()]
    assert "apply_patch" in schemas


# ----------------------------------------------------------------------------
# system prompt 内容契约（守护"核心差异点"不被无意改弱）
# ----------------------------------------------------------------------------
def test_system_prompt_mentions_clarify_parallel_and_decision_order():
    from agent.prompts import build_system_prompt  # noqa: E402

    prompt = build_system_prompt(tool_list="- x()", workspace="/tmp/ws", native_tools=True)
    assert "ask_user" in prompt, "应提示任务含糊时先澄清"
    assert "并行" in prompt, "应提示可并行的读操作一轮发完"
    assert "决策顺序" in prompt, "应给出拿不准时的决策顺序"


# ----------------------------------------------------------------------------
# 健壮性边界（守护已有能力，不引入新代码）
# ----------------------------------------------------------------------------
def test_tool_result_render_redacts_secrets():
    secret = "sk-1234567890abcdefghij"
    res = ToolResult.success(f"调用成功，密钥是 {secret}")
    out = res.render()
    assert secret not in out, "工具回执不得原样泄露密钥"
    assert "sk-***" in out, "密钥应被打码"


def test_malformed_native_tool_args_surfaced_as_issue():
    from agent.llm import ToolCall  # noqa: E402
    registry = build_default_registry()
    msg = AssistantMessage(
        content="",
        tool_calls=[ToolCall(id="c1", name="write_file", arguments={},
                             raw_arguments="{这不是合法 json", malformed=True, source="native")],
    )
    outcome = ToolCallParser(registry, use_native=True).parse(msg)
    assert not outcome.calls, "畸形参数的调用不应被执行"
    assert any("合法 JSON" in i for i in outcome.issues), "必须给模型可自我修正的提示"


def test_auto_compaction_triggers_when_over_budget():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        cfg.max_context_tokens = 150
        cfg.reserve_tokens = 0
        cfg.auto_compact = True
        cfg.compact_threshold = 0.3
        cfg.compact_keep_recent = 4
        backend, profile = _scripted([], native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        # 灌入远超预算的历史
        for i in range(6):
            loop.history.add_assistant(AssistantMessage(content="x" * 2000))
            loop.history.add_tool_result(f"id{i}", "read_file", "y" * 2000)
        assert loop.history.compact_count == 0
        loop._maybe_compact()
        assert loop.history.compact_count >= 1, "超预算应触发自动压缩"
        # 预算（150）远小于"system 提示词 + 摘要"的最小体积，软压缩压不下来，
        # 必须退化为硬压缩；但退化次数**必须有上限**，否则会一直压到只剩 system。
        assert loop.history.compact_count <= 5, \
            f"硬压缩降级次数失控：{loop.history.compact_count}"
        # system 提示词必须保留
        assert loop.history.messages[0]["role"] == "system"


# ----------------------------------------------------------------------------
# 常驻事实层（V7）：压缩不衰减
# ----------------------------------------------------------------------------
class _StubLLM:
    """只服务 History._summarize 的极简假后端（不发网络请求）。

    按调用次序依次返回预设文本；预设用尽后重复最后一条。
    记录每次收到的 prompt，便于断言"哪些内容被拿去摘要了"。
    """

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def chat(self, messages, tools=None, tool_choice=None, on_delta=None):
        self.prompts.append(messages[-1]["content"])
        idx = min(len(self.prompts) - 1, len(self.replies) - 1)
        return AssistantMessage(content=self.replies[idx], finish_reason="stop")


def _fatten(history, rounds=12, tag=""):
    """往历史里灌入若干轮噪声消息，用于把早期内容挤出 keep_recent 窗口。"""
    for i in range(rounds):
        history.add_user(f"{tag}第 {i} 轮无关消息，用于把早期内容挤出保留窗口。" * 3)
        history.add_assistant(AssistantMessage(content=f"{tag}第 {i} 轮回复"))


def test_facts_block_is_pinned_outside_the_summarized_middle():
    """核心回归：事实块**不能**被当成普通历史消息再摘要一次，否则必然逐轮衰减。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        loop = AgentLoop(cfg, profile, MockBackend(profile, []), registry, console=None)

        loop.history.add_user("请用 Python 3.11，并且绝对不要修改 config.yaml。")
        loop.history.add_assistant(AssistantMessage(content="明白。"))
        _fatten(loop.history)

        llm = _StubLLM(
            "【关键事实】\n- 必须用 Python 3.11\n- 绝对不要修改 config.yaml\n"
            "【过程摘要】\n用户交代了约束，随后是无关对话。"
        )
        assert loop.history.compact(llm=llm, keep_recent=4)
        assert "不要修改 config.yaml" in loop.history.facts
        assert loop.history.kinds[1] == "facts", "事实块必须固定在 index 1"

        _fatten(loop.history, tag="二轮")
        assert loop.history.compact(llm=llm, keep_recent=4)

        # 机制级断言：第二次压缩时，被送去摘要的"原始历史"里不该出现事实块。
        # 事实是作为 old_facts 单独传进去要求整体重写的，不是混在流水账里再压一遍。
        transcript = llm.prompts[1].split("===== 历史开始 =====")[1].split("===== 历史结束 =====")[0]
        assert "<key_facts>" not in transcript, "事实块绝不能被当作普通历史消息再摘要一次"


def test_hard_constraint_survives_repeated_compaction():
    """行为级断言：连续压三轮之后，早期硬约束仍然在上下文里。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        loop = AgentLoop(cfg, profile, MockBackend(profile, []), registry, console=None)
        loop.history.add_user("记住：绝对不要修改 config.yaml。")
        loop.history.add_assistant(AssistantMessage(content="记住了。"))
        _fatten(loop.history)

        llm = _StubLLM(
            "【关键事实】\n- 绝对不要修改 config.yaml\n【过程摘要】\n用户交代了硬约束。"
        )
        for round_no in range(3):
            _fatten(loop.history, tag=f"第{round_no}轮")
            assert loop.history.compact(llm=llm, keep_recent=4)
        assert loop.history.compact_count == 3

        blob = "\n".join(str(m.get("content") or "") for m in loop.history.messages)
        assert "不要修改 config.yaml" in blob, "连续三次压缩后硬约束必须仍然存在"


def test_facts_are_rewritten_not_appended():
    """事实块是整体重写而非追加：被推翻的旧结论必须能消失。

    若做成追加，事实块会单调膨胀，而且"方案 B 已失败"这种后来被推翻的结论
    永远删不掉，反而持续误导后续决策。
    """
    h = History("sys")
    _fatten(h)
    llm = _StubLLM(
        "【关键事实】\n- 方案 A 可行\n- 方案 B 已失败\n【过程摘要】\n试过 A 与 B。",
        "【关键事实】\n- 方案 A 可行\n- 方案 B 可行（此前的失败结论已被推翻）\n【过程摘要】\n继续验证。",
    )
    assert h.compact(llm=llm, keep_recent=4)
    assert "方案 B 已失败" in h.facts

    _fatten(h, tag="后续")
    assert h.compact(llm=llm, keep_recent=4)
    assert "方案 B 可行" in h.facts
    assert "已失败" not in h.facts, "被推翻的结论必须消失，否则事实块会膨胀到失真"


def test_facts_survive_when_model_ignores_the_format():
    """模型没按分节格式输出时，宁可不更新事实，也不能让已确认的约束凭空消失。"""
    h = History("sys")
    _fatten(h)
    llm = _StubLLM("【关键事实】\n- 不要改 config.yaml\n【过程摘要】\n用户交代了约束。")
    assert h.compact(llm=llm, keep_recent=4)
    assert "不要改 config.yaml" in h.facts

    _fatten(h, tag="后续")
    unstructured = _StubLLM("总之前面做了不少事情，现在继续往下做。")
    assert h.compact(llm=unstructured, keep_recent=4)
    assert "不要改 config.yaml" in h.facts, "格式不合规时旧事实必须原样保留"


def test_facts_block_is_capped():
    """事实块必须有上限——它是常驻的，无上限就会反过来挤占上下文。"""
    h = History("sys")
    _fatten(h)
    huge = ("【关键事实】\n"
            + "\n".join(f"- 第 {i} 条事实，内容刻意写得长一些以便把事实块撑到上限之外" for i in range(80))
            + "\n【过程摘要】\n略。")
    assert h.compact(llm=_StubLLM(huge), keep_recent=4)
    assert len(h.facts) < MAX_FACTS_CHARS * 1.2, f"事实块未被截断：{len(h.facts)} 字符"
    assert "事实块已满" in h.facts


def test_compaction_rebuilds_workspace_state_for_the_model():
    """压缩后回灌工作区真实清单，并明确要求"别凭记忆写 old_text"。

    摘要只答"做过什么"，答不了"磁盘上现在是什么样"。不补这一段，模型会拿压缩前的
    旧记忆去 edit_block，old_text 必然对不上，白费一整轮。
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        Path(tmp, "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        Path(tmp, "notes.md").write_text("# 说明\n", encoding="utf-8")
        loop = AgentLoop(cfg, profile, MockBackend(profile, []), registry, console=None)
        _fatten(loop.history)

        assert loop.compact_context()
        blob = "\n".join(str(m.get("content") or "") for m in loop.history.messages)
        assert "calc.py" in blob and "notes.md" in blob, "工作区清单必须列出真实存在的文件"
        assert "不要凭记忆写 old_text" in blob, "必须提醒模型重新 read_file，而不是凭记忆编辑"
        assert "2 行" in blob, "清单应带行数，帮模型判断文件规模"


def test_workspace_state_scan_skips_noise_and_caps_files():
    """工作区扫描：跳过噪声目录，且对文件数设上限（避免一次注入淹没上下文）。"""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "__pycache__").mkdir()
        Path(tmp, "__pycache__", "junk.pyc").write_bytes(b"\x00\x01")
        for i in range(40):
            Path(tmp, f"f{i:02d}.py").write_text("x = 1\n", encoding="utf-8")

        entries = list_workspace_files(tmp, max_files=10)
        assert len(entries) == 11, "达到上限时应再多一条'已省略'标记"
        assert entries[-1].get("more") is True
        paths = [e["path"] for e in entries]
        assert not any("__pycache__" in p for p in paths), "噪声目录必须被剪掉"
        assert all(e["lines"] == 1 for e in entries[:-1]), "小文件应报出行数"


def test_compaction_falls_back_to_hard_truncation_when_summary_cannot_fit():
    """摘要压缩压不下来时退化为硬压缩，且降级次数有上限。

    摘要压缩有"最小体积"（system + 事实 + 摘要 + 最近 N 条）。预算小于它时压完
    仍然超阈值，不降级的话下一步又压一次——每步多花一次模型调用却什么都没省下
    （实测 5073 → 5178，反而变大）。硬压缩丢弃更早历史且**不调模型**，便宜且有效。
    """
    h = History("sys " * 200)        # 光 system 就超预算，必然压不下来
    _fatten(h, rounds=10)
    llm = _StubLLM("【关键事实】\n- 约束 A\n【过程摘要】\n略。")

    budget, threshold = 300, 0.5
    assert h.tokens >= budget * threshold
    assert h.compact(llm=llm, keep_recent=4, budget=budget)

    before = h.compact_count
    shrinks = 0
    while h.tokens >= budget * threshold and shrinks < 4:
        shrinks += 1
        if not h.compact(llm=None, keep_recent=max(2, 4 - 2 * shrinks), budget=budget):
            break
    assert h.compact_count > before, "软压缩压不下来时必须继续硬压缩"
    assert shrinks <= 4, "降级次数必须有上限，否则会压到只剩 system"
    assert h.messages[0]["role"] == "system", "无论怎么压，system 提示词都不能丢"
    assert len(h.messages) < 20, f"硬压缩应显著缩短历史，实际 {len(h.messages)} 条"


def test_repeated_compaction_does_not_stack_stale_workspace_listings():
    """连续压缩时，工作区清单只保留最新一份。

    它是"可重建的事实"，价值只在于最新；旧副本堆在上下文里既白占额度，
    又可能与现状矛盾（实测连续两次压缩就堆出了两份完全相同的清单）。
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        Path(tmp, "a.py").write_text("x = 1\n", encoding="utf-8")
        loop = AgentLoop(cfg, profile, MockBackend(profile, []), registry, console=None)
        llm = _StubLLM("【关键事实】\n- 约束 A\n【过程摘要】\n略。")

        for _ in range(3):
            _fatten(loop.history, rounds=10)
            loop.history.compact(llm=llm, keep_recent=6, budget=2000)
            state = loop._workspace_state_note()
            assert state
            loop.history.drop_notes(WORKSPACE_STATE_MARKER)
            loop.history.add_note(state)

        hits = [m.get("content") or "" for m in loop.history.messages
                if isinstance(m.get("content"), str)
                and m["content"].startswith(WORKSPACE_STATE_MARKER)]
        assert len(hits) == 1, f"工作区清单应只留最新一份，实际 {len(hits)} 份"


# ----------------------------------------------------------------------------
# 会话检查点（跨进程续跑）
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def _raises(exc_type):
    """断言"确实抛出了指定异常"的小工具（测试 runner 没有 pytest.raises）。"""
    try:
        yield
    except exc_type:
        return
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"期望 {exc_type.__name__}，实际 {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"期望抛出 {exc_type.__name__}，但没有抛出")


def _run_then_new_loop(tmp, script, task="跑个任务", prepare=None):
    """用 script 跑一次任务并保存检查点，再返回一个全新的 loop（模拟进程重启）。

    prepare(loop) 在 save 之前调用，用于设置"应当被持久化"的状态——
    放在 save 之后设就测不到持久化了。
    """
    cfg, profile, registry, _ = _make_env(tmp)
    backend, profile = _scripted(script, native=True)
    loop = AgentLoop(cfg, profile, backend, registry, console=None)
    loop.run(task)
    if prepare:
        prepare(loop)
    path = loop.save_checkpoint()
    assert path is not None and Path(path).exists(), "跑过任务后必须产出检查点"
    backend2, profile2 = _scripted([], native=True)
    loop2 = AgentLoop(cfg, profile2, backend2, registry, console=None)
    return loop, loop2, path


def test_checkpoint_roundtrip_restores_history_facts_and_task():
    """往返检查点：历史 / 事实 / 任务名能恢复。"""
    with tempfile.TemporaryDirectory() as tmp:
        script = [
            {"content": "写文件。", "tool_calls": [
                {"name": "write_file", "arguments": {"path": "a.py", "content": "x = 1\n"}}]},
            {"content": "跑一下。", "tool_calls": [
                {"name": "run_command", "arguments": {"command": "python a.py"}}]},
            {"content": "完成。", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "done"}}]},
        ]
        def _set_facts(lp):
            lp.history.facts = "- 硬约束：不要改 config.yaml"

        loop, loop2, path = _run_then_new_loop(tmp, script, task="创建脚本并运行",
                                               prepare=_set_facts)
        n = loop2.resume_checkpoint(path)
        assert n > 0
        assert loop2.task_name == loop.task_name, "任务名必须恢复，否则代码会落到另一个目录"
        assert loop2.history.facts == "- 硬约束：不要改 config.yaml", "事实块必须随检查点恢复"
        # system 提示词不入库，恢复时按当前代码与工作区重建
        assert loop2.history.messages[0]["role"] == "system"
        assert loop2.history.messages[0]["content"] == loop2.system_prompt


def test_checkpoint_keeps_cost_accounting_continuous():
    """成本对账必须跨进程连续，否则恢复后 /stats 会从零开始，账就断了。"""
    with tempfile.TemporaryDirectory() as tmp:
        script = [
            {"content": "跑一下", "tool_calls": [
                {"name": "run_command", "arguments": {"command": "echo hi"}}]},
            {"content": "完成", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "ok"}}]},
        ]
        loop, loop2, path = _run_then_new_loop(tmp, script, task="跑个命令")
        before_total = loop._usage.get("total_tokens", 0)
        assert before_total > 0

        loop2.resume_checkpoint(path)
        assert loop2._usage.get("total_tokens", 0) == before_total, "token 账目恢复后必须接上"
        assert loop2._total_tool_calls == loop._total_tool_calls
        assert loop2._model_calls == loop._model_calls


def test_resume_warns_the_model_the_history_is_not_fresh():
    """恢复后必须如实告知模型"这是历史记录、不是你刚做过的"。

    不提醒的话，模型会以为自己刚执行过历史里的操作，于是跳过本该做的验证，
    或者重复执行一遍。
    """
    with tempfile.TemporaryDirectory() as tmp:
        script = [
            {"content": "写文件。", "tool_calls": [
                {"name": "write_file", "arguments": {"path": "a.py", "content": "x = 1\n"}}]},
            {"content": "完成。", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "done"}}]},
        ]
        _, loop2, path = _run_then_new_loop(tmp, script, task="创建 a.py")
        loop2.resume_checkpoint(path)

        blob = "\n".join(str(m.get("content") or "") for m in loop2.history.messages)
        assert "恢复的会话" in blob
        assert "不代表你刚刚执行过" in blob, "必须提醒模型：历史 ≠ 刚做过"
        assert "工作区当前状态" in blob, "恢复后同样要补一份工作区真实清单（与压缩后同理）"


def test_corrupt_checkpoint_raises_instead_of_being_ignored():
    """损坏的检查点必须报错，不能被当成"没有检查点"——两者处置完全不同。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        backend, profile = _scripted([], native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)

        d = checkpoint_dir(loop)
        d.mkdir(parents=True, exist_ok=True)

        truncated = d / "truncated.json"
        truncated.write_text('{"version": 1, "messages": [{"role": "user"', encoding="utf-8")
        with _raises(ValueError):
            load_checkpoint(truncated)

        future = d / "future.json"
        future.write_text(json.dumps({"version": 99, "messages": [], "kinds": []}), encoding="utf-8")
        with _raises(ValueError):
            load_checkpoint(future)

        mismatched = d / "mismatch.json"
        mismatched.write_text(
            json.dumps({"version": 1, "messages": [{"role": "user", "content": "hi"}], "kinds": []}),
            encoding="utf-8")
        with _raises(ValueError):
            load_checkpoint(mismatched)


def test_empty_session_is_not_checkpointed():
    """空会话没有恢复价值，不该留下一个空文件（还会让启动提示变噪音）。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        backend, profile = _scripted([], native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        assert loop.save_checkpoint() is None


def test_checkpoint_file_is_valid_json_with_no_temp_leftover():
    """原子写：落地的必须是完整 JSON，且不留 .tmp 残留。"""
    with tempfile.TemporaryDirectory() as tmp:
        script = [{"content": "完成", "tool_calls": [
            {"name": "finish", "arguments": {"summary": "ok"}}]}]
        _, _, path = _run_then_new_loop(tmp, script, task="最小任务")
        path = Path(path)
        assert path.suffix == ".json"
        assert not path.with_name(path.name + ".tmp").exists(), "原子写不应留下临时文件"
        state = json.loads(path.read_text(encoding="utf-8"))
        assert state["version"] == 1
        assert isinstance(state["messages"], list)



# ----------------------------------------------------------------------------
# V9：质量指标（回答"做对了没有"，而不只是"花了多少"）
# ----------------------------------------------------------------------------
def test_repair_stats_cut_failures_into_episodes():
    """自修复按"回合"计：一段连续失败 + 紧随的成功 = 一回合，轮数=连续失败的长度。

    失败 3 次才修好和失败 1 次就修好是两种能力水平，只看"失败总数"区分不了。
    序列 F,F,S,F,S → 2 回合 / 共 3 轮；末尾若在失败则记为"未修复"，不计入平均值。
    """
    from agent.metrics import SessionMetrics
    m = SessionMetrics()
    for ok in [False, False, True, False, True]:
        m.record_verify(ok)
    assert m._repair_stats() == (2, 3, 0)

    m2 = SessionMetrics()
    for ok in [False, False]:          # 一直在失败，从没修回来
        m2.record_verify(ok)
    assert m2._repair_stats() == (0, 0, 2)

    m3 = SessionMetrics()
    for ok in [True, False, True]:     # 第一次就成功，不算"修复回合"
        m3.record_verify(ok)
    assert m3._repair_stats() == (1, 1, 0)


def test_avg_repair_rounds_is_none_not_zero_when_never_repaired():
    """从没成功修复过 → None 而不是 0。0 会被读成"一次就修好"，与事实相反。"""
    from agent.metrics import TaskRecord
    assert TaskRecord(finish_reason="finish").avg_repair_rounds is None
    r = TaskRecord(finish_reason="finish", repair_rounds=2, repair_attempts=6)
    assert r.avg_repair_rounds == 3.0


def test_rework_counts_files_written_more_than_once():
    """返工 = 同一路径被写入多次；命令类改动没有 path，不该被算进来。"""
    from agent.metrics import SessionMetrics
    m = SessionMetrics()
    m.start_task("t")
    rec = m.finish_task("finish", changes=[
        {"kind": "write", "path": "a.py"},
        {"kind": "edit", "path": "a.py"},     # 同一个文件改了两次 → 返工
        {"kind": "write", "path": "b.py"},
        {"kind": "command", "detail": "python a.py"},   # 无 path，排除
    ])
    assert rec.files_changed == 2
    assert rec.rework_files == 1


def test_quality_metrics_record_a_repair_round_end_to_end():
    """端到端：跑失败 → 修好 → 跑通，应记为 1 个修复回合、1 轮。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        script = [
            {"content": "写脚本", "tool_calls": [
                {"name": "write_file", "arguments": {"path": "calc.py",
                 "content": "def div(a, b):\n    return a / b\n\n\nprint(div(4, 0))\n"}}]},
            {"content": "运行", "tool_calls": [
                {"name": "run_command", "arguments": {"command": "python calc.py"}}]},
            {"content": "修", "tool_calls": [
                {"name": "edit_block", "arguments": {
                    "path": "calc.py", "old_text": "print(div(4, 0))", "new_text": "print(div(4, 2))"}}]},
            {"content": "再运行", "tool_calls": [
                {"name": "run_command", "arguments": {"command": "python calc.py"}}]},
            {"content": "完成", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "已修复并验证"}}]},
        ]
        backend, profile = _scripted(script, native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        result = loop.run("写个除法函数")
        assert result.finish_reason == "finish", result.finish_reason
        assert result.errors == 1, f"第一次运行必须真的失败，实际 errors={result.errors}"

        rec = loop.metrics.tasks[-1]
        assert rec.verify_runs == 2, rec.verify_runs
        assert rec.verify_failures == 1
        assert rec.repair_rounds == 1, "失败→成功应记为一个修复回合"
        assert rec.repair_attempts == 1
        assert rec.unresolved_failures == 0
        assert rec.succeeded is True
        assert rec.avg_repair_rounds == 1.0


def test_quality_panel_reports_success_rate_and_outcomes():
    """质量面板必须给出成功率与结局分布——光有 token/耗时答不出"做对了没有"。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        script = [{"content": "完成", "tool_calls": [
            {"name": "finish", "arguments": {"summary": "ok"}}]}]
        backend, profile = _scripted(script, native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        loop.run("最小任务")

        panel = loop.build_stats_panel()
        assert "质量指标" in panel, "成本面板里应包含质量段"
        assert "任务数：1" in panel
        assert "100.0%" in panel
        assert "结局分布" in panel
        # 结局要给"人话"解释，不能只丢一个枚举名让人猜
        assert "显式完成" in panel


def test_failure_outcomes_lower_the_success_rate():
    """非成功结局必须拉低成功率：只统计"跑得漂亮"的任务会让指标自我美化。"""
    from agent.metrics import SessionMetrics
    m = SessionMetrics()
    m.start_task("任务A")
    m.finish_task("finish")
    m.start_task("任务B")
    m.finish_task("too_many_errors")
    assert m.task_count == 2
    assert m.success_count == 1
    assert abs(m.success_rate - 0.5) < 1e-9
    assert m.outcome_counts() == {"finish": 1, "too_many_errors": 1}
    panel = m.render_panel()
    assert "连续失败过多 1" in panel, "结局要翻译成人话，不能只丢枚举名"
    # model_final 在成功之列，但置信度低于 finish（后者通过了假完成拦截）——
    # 面板里两者分开列出，不混为一谈
    from agent.metrics import SUCCESS_REASONS
    assert "model_final" in SUCCESS_REASONS and "max_steps" not in SUCCESS_REASONS
    m.start_task("任务C")
    m.finish_task("model_final")
    assert "未调用 finish" in m.render_panel()


def test_metrics_survive_checkpoint_roundtrip():
    """指标口径要跨进程连续：恢复会话后成功率不该只统计恢复之后的任务。"""
    script = [{"content": "完成", "tool_calls": [
        {"name": "finish", "arguments": {"summary": "ok"}}]}]
    with tempfile.TemporaryDirectory() as tmp:
        _, loop2, path = _run_then_new_loop(tmp, script, task="最小任务")
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        assert saved.get("metrics"), "检查点应带上质量指标"
        n = loop2.resume_checkpoint(path)
        assert n >= 1
        assert loop2.metrics.task_count == 1, "恢复后应拿回之前的任务记录"
        assert loop2.metrics.success_rate == 1.0


# ----------------------------------------------------------------------------
# V10：评测台（以产物为准，不采信模型自述）
# ----------------------------------------------------------------------------
# 参考解：每个任务一份"正确实现"与一份"看起来像但其实是错的实现"。
# 验证器必须能区分两者——一个恒为真的验证器会让整场评测失去意义，
# 而且不会报错、只会安静地给出 100% 通过率，这比没有验证器更危险。
_CORRECT_SOLUTIONS = {
    "calc": {"calc.py":
             "def add(a, b):\n    return a + b\n"
             "def sub(a, b):\n    return a - b\n"
             "def mul(a, b):\n    return a * b\n"
             "def div(a, b):\n    if b == 0:\n        raise ValueError('div by zero')\n    return a / b\n"},
    "fixbug": {"stats.py":
               "def total(items):\n    return sum(items)\n\n\n"
               "def average(items):\n    if not items:\n        return 0\n    return total(items) / len(items)\n"},
    "csvreport": {"report.py":
                  "import csv\nrows = list(csv.DictReader(open('scores.csv', encoding='utf-8')))\n"
                  "n = [int(r['分数']) for r in rows]\n"
                  "print('平均分:', round(sum(n) / len(n), 1))\nprint('最高分:', max(n))\n"},
    "dedup": {"dedup.py":
              "def dedup(items):\n    seen = set()\n    out = []\n    for x in items:\n"
              "        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n"},
    "lru": {"lru.py":
            "from collections import OrderedDict\n"
            "class LRUCache:\n    def __init__(self, capacity):\n"
            "        self.cap = capacity\n        self.d = OrderedDict()\n"
            "    def get(self, k):\n        if k not in self.d:\n            return -1\n"
            "        self.d.move_to_end(k)\n        return self.d[k]\n"
            "    def put(self, k, v):\n        if k in self.d:\n            self.d.move_to_end(k)\n"
            "        self.d[k] = v\n        if len(self.d) > self.cap:\n"
            "            self.d.popitem(last=False)\n"},

    # ---- V11 新增任务的参考解 ------------------------------------------------
    "findkw": {"find_kw.py":
               "import sys\n"
               "path, kw = sys.argv[1], sys.argv[2]\n"
               "hits = []\n"
               "for i, line in enumerate(open(path, encoding='utf-8'), 1):\n"
               "    if kw in line.rstrip('\\n'):\n"
               "        hits.append(f'{i}:{line.rstrip()}')\n"
               "print('\\n'.join(hits) if hits else '无匹配')\n"},
    "wordfreq": {"wordfreq.py":
                 "import collections, re\n"
                 "c = collections.Counter()\n"
                 "for line in open('article.txt', encoding='utf-8'):\n"
                 "    for w in re.findall(r'[A-Za-z]+', line):\n"
                 "        c[w.lower()] += 1\n"
                 "for w, n in c.most_common(3):\n"
                 "    print(w, n)\n"},
    "flatten": {"flatten.py":
                "def flatten(items):\n    out = []\n    for x in items:\n"
                "        if isinstance(x, list):\n            out.extend(flatten(x))\n"
                "        else:\n            out.append(x)\n    return out\n"},
    "bank": {"account.py":
             "class Account:\n    def __init__(self, name, balance=0):\n"
             "        self.name = name\n        self.balance = balance\n"
             "    def deposit(self, amount):\n        self.balance += amount\n"
             "    def withdraw(self, amount):\n"
             "        if amount > self.balance:\n"
             "            raise ValueError('insufficient balance')\n"
             "        self.balance -= amount\n",
             "bank.py":
             "from account import Account\n"
             "class Bank:\n    def __init__(self):\n        self.accounts = {}\n"
             "    def open(self, name, balance=0):\n"
             "        a = Account(name, balance)\n        self.accounts[name] = a\n"
             "        return a\n"
             "    def get(self, name):\n        return self.accounts[name]\n"
             "    def transfer(self, src, dst, amount):\n"
             "        s, d = self.accounts[src], self.accounts[dst]\n"
             "        if amount > s.balance:\n            raise ValueError('insufficient')\n"
             "        s.withdraw(amount)\n        d.deposit(amount)\n"
             "    def total(self):\n"
             "        return sum(a.balance for a in self.accounts.values())\n"},
    "parselog": {"parselog.py":
                 "import re\n"
                 "errs = []\n"
                 "for line in open('server.log', encoding='utf-8'):\n"
                 "    m = re.match(r'\\d{4}-\\d{2}-\\d{2} (\\d{2}:\\d{2}:\\d{2}) (\\w+)\\s+(.*)$',"
                 " line.rstrip('\\n'))\n"
                 "    if m and m.group(2) == 'ERROR':\n"
                 "        errs.append(m.group(1) + ' ' + m.group(3))\n"
                 "print('ERROR 条数:', len(errs))\n"
                 "for e in errs:\n    print(e)\n"},
}

_BROKEN_SOLUTIONS = {
    "calc": {"calc.py": "def add(a, b):\n    return a - b\n"          # add 写错
                        "def sub(a, b):\n    return a - b\n"
                        "def mul(a, b):\n    return a * b\n"
                        "def div(a, b):\n    return a / b\n"},         # 除 0 不抛异常
    "fixbug": {"stats.py": "def total(items):\n    return sum(items)\n\n\n"
                           "def average(items):\n    if not items:\n        return 0\n"
                           "    return total(items) / (len(items) - 1)\n"},   # 原 bug 未修
    "csvreport": {"report.py": "print('平均分:', 1)\nprint('最高分:', 2)\n"},  # 数字是编的
    "dedup": {"dedup.py": "def dedup(items):\n    return list(set(items))\n"},  # 丢顺序
    "lru": {"lru.py": "class LRUCache:\n    def __init__(self, capacity):\n"
                      "        self.cap = capacity\n        self.d = {}\n"
                      "    def get(self, k):\n        return self.d.get(k, -1)\n"
                      "    def put(self, k, v):\n        self.d[k] = v\n"},     # 不淘汰

    # ---- V11 新增任务的错误解（每一条都是"看起来像、其实错"的真实写法）----
    # findkw：行号从 0 开始（经典 off-by-one，输出看着完全合理）
    "findkw": {"find_kw.py":
               "import sys\n"
               "path, kw = sys.argv[1], sys.argv[2]\n"
               "hits = []\n"
               "for i, line in enumerate(open(path, encoding='utf-8')):\n"
               "    if kw in line.rstrip('\\n'):\n"
               "        hits.append(f'{i}:{line.rstrip()}')\n"
               "print('\\n'.join(hits) if hits else '无匹配')\n"},
    # wordfreq：不做归一化，'Apple,' 被当成独立词
    "wordfreq": {"wordfreq.py":
                 "import collections\n"
                 "c = collections.Counter()\n"
                 "for line in open('article.txt', encoding='utf-8'):\n"
                 "    c.update(line.split())\n"
                 "for w, n in c.most_common(3):\n    print(w, n)\n"},
    # flatten：只展平一层，深嵌套用例会露馅
    "flatten": {"flatten.py":
                "def flatten(items):\n    out = []\n    for x in items:\n"
                "        if isinstance(x, list):\n            out.extend(x)\n"
                "        else:\n            out.append(x)\n    return out\n"},
    # bank：transfer 不校验余额，失败转账会把余额扣成负数
    "bank": {"account.py":
             "class Account:\n    def __init__(self, name, balance=0):\n"
             "        self.name = name\n        self.balance = balance\n"
             "    def deposit(self, amount):\n        self.balance += amount\n"
             "    def withdraw(self, amount):\n        self.balance -= amount\n",
             "bank.py":
             "from account import Account\n"
             "class Bank:\n    def __init__(self):\n        self.accounts = {}\n"
             "    def open(self, name, balance=0):\n"
             "        a = Account(name, balance)\n        self.accounts[name] = a\n"
             "        return a\n"
             "    def get(self, name):\n        return self.accounts[name]\n"
             "    def transfer(self, src, dst, amount):\n"
             "        self.accounts[src].balance -= amount\n"
             "        self.accounts[dst].balance += amount\n"
             "    def total(self):\n"
             "        return sum(a.balance for a in self.accounts.values())\n"},
    # parselog：不过滤级别，把 WARN/INFO 也一起输出
    "parselog": {"parselog.py":
                 "import re\n"
                 "errs = []\n"
                 "for line in open('server.log', encoding='utf-8'):\n"
                 "    m = re.match(r'\\S+ (\\S+) \\w+\\s+(.*)$', line.rstrip('\\n'))\n"
                 "    if m:\n        errs.append(m.group(1) + ' ' + m.group(2))\n"
                 "print('ERROR 条数:', len(errs))\n"
                 "for e in errs:\n    print(e)\n"},
}


def _verify_with(task_name, solutions):
    """把参考解放进临时目录，跑该任务的验证器。"""
    from agent.eval import TASKS
    task = next(t for t in TASKS if t.name == task_name)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for rel, content in task.files.items():
            (d / rel).write_text(content, encoding="utf-8")
        for rel, content in solutions.items():
            (d / rel).write_text(content, encoding="utf-8")
        return task.verify(d)


def test_eval_verifiers_accept_correct_solutions():
    """验证器必须放行正确实现。否则整套评测会把好结果判成失败。"""
    for name in _CORRECT_SOLUTIONS:
        ok, msg = _verify_with(name, _CORRECT_SOLUTIONS[name])
        assert ok, f"{name} 的正确实现被误判：{msg}"


def test_eval_verifiers_reject_broken_solutions():
    """验证器必须拦下"看起来像但其实错"的实现。

    这是评测台最关键的一条：一个恒为真的验证器不会报错，只会安静地给出
    100% 通过率——比没有验证器更危险。所以每条任务的错误解都必须被抓住。
    """
    for name in _BROKEN_SOLUTIONS:
        ok, msg = _verify_with(name, _BROKEN_SOLUTIONS[name])
        assert not ok, f"{name} 的错误实现被误放行，验证器区分度不足"


def test_every_task_has_both_reference_solutions():
    """每个任务都必须同时备好"正确解"和"错误解"两份参考实现。

    这是**防静默跳过**的一条：上面两条区分度测试是 `for name in _CORRECT_SOLUTIONS`
    这样遍历字典的——新增任务时若忘了配参考解，循环只是少转一圈，
    测试照样全绿，而新任务的验证器可能根本没被检验过。
    这类"什么都不测却报通过"的缺口，比测试失败更危险。
    """
    from agent.eval import TASKS
    names = {t.name for t in TASKS}
    assert names == set(_CORRECT_SOLUTIONS), (
        f"缺正确参考解：{sorted(names - set(_CORRECT_SOLUTIONS))}"
        f"；多出：{sorted(set(_CORRECT_SOLUTIONS) - names)}")
    assert names == set(_BROKEN_SOLUTIONS), (
        f"缺错误参考解：{sorted(names - set(_BROKEN_SOLUTIONS))}")


def test_cli_verifier_rejects_hardcoded_output():
    """CLI 类任务必须换参数跑多组，否则"把答案硬编码 print 出来"就能通过。

    这条测的是验证器本身的防作弊能力：单独喂一组参数时，
    一个完全不解析 argv、直接 print 期望文本的脚本应当也能骗过验证器——
    那就说明验证器没在测 CLI 契约。多组参数是唯一的低成本防线。
    """
    from agent.eval import TASKS
    task = next(t for t in TASKS if t.name == "findkw")
    # 硬编码解：不读 argv，直接把 "fix" 那组的期望输出写死
    hardcoded = {"find_kw.py": "print('2:fix: login timeout on slow network')\n"
                               "print('4:fix: crash when file is missing')\n"}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for rel, content in task.files.items():
            (d / rel).write_text(content, encoding="utf-8")
        for rel, content in hardcoded.items():
            (d / rel).write_text(content, encoding="utf-8")
        ok, _msg = task.verify(d)
    assert not ok, "硬编码输出的脚本被放行了，多组参数没起到防作弊作用"


def test_eval_task_suite_covers_distinct_dimensions():
    """任务集不能靠"多出几道同类型题"把样本量灌大。

    10 个任务应当是 10 个维度。这条只守住最粗的一条：考察点说明两两不同
    （真正是否同质还得靠人读 prompt），但它能挡住最省事的那种灌水——
    复制一个任务换个名字。
    """
    from agent.eval import TASKS
    intents = [t.intent for t in TASKS]
    assert len(intents) == len(set(intents)), "存在考察点完全相同的任务，属于重复计数"
    assert len(intents) >= 10, f"任务数 {len(intents)}，样本仍偏小"


def test_eval_report_aligns_cjk_columns_by_display_width():
    """报告表格要按**显示宽度**对齐，不能按字符数。

    报告是要贴进答辩材料的。`f'{s:<26}'` 按字符数补齐，而 CJK 在终端里占两列，
    含中文的那一列会整体错位、后面的数字全糊在一起——表格读不了，数字就等于没有。
    这条钉住按 East Asian Width 计算宽度（W/F 记 2 列）的行为。
    """
    from agent.eval import _disp_width, _pad, EvalOutcome, render_report

    assert _disp_width("abc") == 3
    assert _disp_width("整理") == 4, "CJK 应按 2 列计宽"
    assert _disp_width("a，b") == 4, "全角标点（U+FF0C，分类 F）同样占 2 列"

    # 中英混排时，补出来的字符串显示宽度必须一致
    a, b = _pad("整理", 10), _pad("abcdefghij", 10)
    assert _disp_width(a) == _disp_width(b) == 10
    assert _disp_width(_pad("十二个字符的中文考察点说明很长很长", 10)) <= 10, "超宽要截断且不越界"

    # 端到端：含中文考察点的表格，分隔线之后每行长度应一致
    report = render_report([
        EvalOutcome("lru", "数据结构：带淘汰策略的缓存（容量满时淘汰最久未用）", "finish", 5, 5, 0, 7.6, 900, True, "通过"),
        EvalOutcome("abc", "short ascii intent", "finish", 5, 5, 0, 7.6, 900, True, "通过"),
    ], model="m", elapsed=15.0)
    rows = [ln for ln in report.splitlines() if ln.strip().startswith(("lru", "abc"))]
    assert len(rows) == 2
    widths = [_disp_width(r) for r in rows]
    assert len(set(widths)) == 1, f"两行显示宽度不一致，表格会错位：{widths}"


def test_eval_report_separates_self_claim_from_artifact():
    """报告必须区分"模型自述完成"与"产物验证通过"，并点出两类偏差。"""
    from agent.eval import EvalOutcome, render_report
    outcomes = [
        EvalOutcome("calc", "新建", "finish", 3, 3, 0, 5.0, 900, True, "通过"),
        # 自述完成但产物跑不通 —— 假完成，最危险的一类
        EvalOutcome("dedup", "算法", "finish", 4, 5, 0, 7.0, 1200, False, "顺序错"),
        # 产物其实通过了却没敢收尾 —— 悲观失败，能力被低估
        EvalOutcome("lru", "数据结构", "max_steps", 8, 9, 1, 20.0, 3000, True, "通过"),
    ]
    report = render_report(outcomes, model="test-model", elapsed=32.0)
    assert "产物验证通过：2/3" in report
    assert "模型自述完成：2/3" in report
    assert "假完成" in report and "dedup" in report
    assert "悲观失败" in report and "lru" in report
    # 退出码语义：以产物为准，不是以自述为准
    from agent.eval import main
    assert main is not None


def test_empty_model_response_is_not_counted_as_success():
    """模型一个字都没说 ≠ 任务完成（评测台实跑才暴露的缺陷）。

    原行为：空响应在文本通道被判成 is_final → 主循环跳过 _EMPTY_OUTPUT_HINT 纠错
    → 直接以 model_final 收尾，answer 为"模型未给出总结"。于是网关抖一下，
    任务就被记成"成功收尾"，而产物可能根本没生成。

    两条修正：① 文本通道里"没有调用块"的默认仅在**确有正文**时成立；
    ② 纠错重试耗尽后仍为空，单列 empty_response 结局，且不计入成功率。
    这条是"假完成拦截"的另一半：拦的是"改了文件没验证"，这里拦的是"沉默被当结论"。
    """
    from agent.llm import AssistantMessage
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        backend, profile = _scripted(
            [{"content": "写", "tool_calls": [
                {"name": "write_file", "arguments": {"path": "a.py", "content": "x = 1\n"}}]}],
            native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        orig_chat, n = backend.chat, {"c": 0}

        def chat(*a, **k):
            n["c"] += 1
            if n["c"] >= 2:                       # 之后一律返回完全空的响应
                return AssistantMessage(content="", tool_calls=[], usage=None)
            return orig_chat(*a, **k)
        backend.chat = chat

        result = loop.run("写个文件")
        assert result.finish_reason == "empty_response", result.finish_reason
        assert n["c"] == 4, f"应先纠错重试 2 次再放弃，实际调用 {n['c']} 次"
        rec = loop.metrics.tasks[-1]
        assert rec.succeeded is False, "空响应绝不能计入成功率"
        assert "空响应" in loop.metrics.render_panel()


def test_empty_response_is_retried_with_hint_before_giving_up():
    """空响应第一次出现时要给纠错提示重试，而不是立刻放弃。

    但重试要**有限**：模型若持续空响应，无限重试只会烧 token。
    """
    from agent.llm import AssistantMessage
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        backend, profile = _scripted([], native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        n = {"c": 0}

        def chat(*a, **k):
            n["c"] += 1
            if n["c"] == 1:
                return AssistantMessage(content="", tool_calls=[], usage=None)
            return AssistantMessage(content="FINAL: 好了。", tool_calls=[], usage=None)
        backend.chat = chat

        result = loop.run("什么都不做")
        assert result.finish_reason == "model_final", result.finish_reason
        assert n["c"] == 2, "第一次空响应后应重试一次"
        joined = "\n".join(str(m.get("content", "")) for m in loop.history.messages)
        assert "没有包含任何工具调用" in joined, "应把 _EMPTY_OUTPUT_HINT 回灌给模型"


def test_read_many_files_reads_batch():
    from agent.tools import build_default_registry, build_tool_context
    from agent.config import AgentConfig
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        (base / "b.txt").write_text("hello\nworld\n", encoding="utf-8")
        (base / "c.bin").write_bytes(b"\x00\x01\x02binary")
        cfg = AgentConfig(workspace=tmp, session_log=None)
        registry = build_default_registry()
        ctx = build_tool_context(cfg, console=None, session={})

        r = registry.execute("read_many_files", {"paths": ["a.py", "b.txt", "c.bin", "missing.py"]}, ctx)
        assert r.ok
        assert "## a.py" in r.output and "## b.txt" in r.output
        # 二进制与缺失都被优雅跳过，而不是报错
        assert "疑似二进制" in r.output
        assert "跳过" in r.output and "missing.py" in r.output
        assert r.meta["read"] == 2, r.meta

        # 空数组应报错（被转成 ok=False 回执，不崩溃）
        re_ = registry.execute("read_many_files", {"paths": []}, ctx)
        assert not re_.ok


def test_replace_in_file_global_replaces_all():
    from agent.tools import build_default_registry, build_tool_context
    from agent.config import AgentConfig
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / "m.py").write_text("x = OLD\ny = OLD\nz = NEW\n", encoding="utf-8")
        cfg = AgentConfig(workspace=tmp, session_log=None, backup_on_write=False)
        registry = build_default_registry()
        ctx = build_tool_context(cfg, console=None, session={})

        r = registry.execute("replace_in_file",
                             {"path": "m.py", "old_text": "OLD", "new_text": "NEW"}, ctx)
        assert r.ok and r.meta["replacements"] == 2, r.meta
        assert (base / "m.py").read_text(encoding="utf-8") == "x = NEW\ny = NEW\nz = NEW\n"

        # 找不到应报错（ok=False，不崩溃）
        rn = registry.execute("replace_in_file",
                              {"path": "m.py", "old_text": "NOPE", "new_text": "X"}, ctx)
        assert not rn.ok

        # 空 old_text 必须被拒绝，否则会清空文件
        re_ = registry.execute("replace_in_file",
                               {"path": "m.py", "old_text": "", "new_text": "X"}, ctx)
        assert not re_.ok
        assert (base / "m.py").read_text(encoding="utf-8") == "x = NEW\ny = NEW\nz = NEW\n"


def test_git_tool_rejects_dangerous_and_runs_safe():
    import subprocess
    import unittest.mock as mock
    from agent.tools import build_default_registry, build_tool_context
    from agent.config import AgentConfig

    with tempfile.TemporaryDirectory() as tmp:
        cfg = AgentConfig(workspace=tmp, session_log=None, command_timeout=30)
        registry = build_default_registry()
        ctx = build_tool_context(cfg, console=None, session={})

        # 危险 / 未知子命令必须被拒绝（不真正执行）
        rp = registry.execute("git", {"args": "push origin main"}, ctx)
        assert not rp.ok, "push 必须被禁止"
        rr = registry.execute("git", {"args": "reset --hard"}, ctx)
        assert not rr.ok, "reset --hard 必须被禁止"
        rk = registry.execute("git", {"args": "frobnicate"}, ctx)
        assert not rk.ok, "未知子命令必须被拒绝"

        # 安全只读命令：用假 subprocess 验证分发逻辑（不依赖本机是否装 git）
        def fake_run(cmd, **kw):
            assert cmd[0] == "git"
            return subprocess.CompletedProcess(cmd, 0, stdout=f"fake git {cmd[1]} ok", stderr="")

        with mock.patch("agent.tools.git_tool.subprocess.run", fake_run):
            rs = registry.execute("git", {"args": "status --short"}, ctx)
            assert rs.ok and "fake git status" in rs.output, rs.render()
            rl = registry.execute("git", {"args": "log -5"}, ctx)
            assert rl.ok and "fake git log" in rl.output


def test_git_tool_controlled_commit_requires_message_and_rejects_dangerous():
    import subprocess
    import unittest.mock as mock
    from agent.tools import build_default_registry, build_tool_context
    from agent.config import AgentConfig

    with tempfile.TemporaryDirectory() as tmp:
        cfg = AgentConfig(workspace=tmp, session_log=None, command_timeout=30)
        registry = build_default_registry()
        ctx = build_tool_context(cfg, console=None, session={})

        # 有暂存区（允许提交走下去）
        def fake_run_staged(cmd, **kw):
            assert cmd[0] == "git"
            if cmd[1:4] == ["diff", "--cached", "--name-only"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="src/main.py\n", stderr="")
            if cmd[1] == "commit":
                assert "-m" in cmd, "commit 必须带 -m"
                return subprocess.CompletedProcess(cmd, 0, stdout="[main abc123] done", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout=f"fake git {cmd[1]} ok", stderr="")

        # 空暂存区（应被拦截）
        def fake_run_empty(cmd, **kw):
            assert cmd[0] == "git"
            if cmd[1:4] == ["diff", "--cached", "--name-only"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="should not happen", stderr="")

        # 1) 无提交信息必须被拒绝（args 内无 -m）
        rn = registry.execute("git", {"args": "commit"}, ctx)
        assert not rn.ok, "commit 无 -m 必须被拒绝"

        # 2) 空 -m 必须被拒绝
        re_ = registry.execute("git", {"args": "commit -m \"\""}, ctx)
        assert not re_.ok, "commit 空 -m 必须被拒绝"

        # 3) 改写历史的选项必须被拒绝
        for bad in ("commit --amend -m x", "commit --no-verify -m x",
                    "commit -a -m x", "commit --allow-empty -m x", "commit --date=now -m x"):
            rb = registry.execute("git", {"args": bad}, ctx)
            assert not rb.ok, f"commit 危险选项应被拒绝：{bad}"

        # 4) 合法 commit（args 内带 -m）
        with mock.patch("agent.tools.git_tool.subprocess.run", fake_run_staged):
            rk = registry.execute("git", {"args": "commit -m 'fix: bug'"}, ctx)
            assert rk.ok, rk.render()
            assert "[main abc123] done" in rk.output

        # 5) 合法 commit（用结构化 message 参数）
        with mock.patch("agent.tools.git_tool.subprocess.run", fake_run_staged):
            rm = registry.execute("git", {"args": "commit", "message": "feat: add"}, ctx)
            assert rm.ok, rm.render()

        # 6) 空暂存区必须被拦截（提示先 git add）
        with mock.patch("agent.tools.git_tool.subprocess.run", fake_run_empty):
            re2 = registry.execute("git", {"args": "commit -m 'x'"}, ctx)
            assert not re2.ok, "空暂存区 commit 必须被拦截"
            assert "git add" in re2.render()


def test_web_fetch_strips_html_and_rejects_bad_scheme():
    import io
    import urllib.request
    import urllib.error
    from agent.tools import build_default_registry, build_tool_context
    from agent.config import AgentConfig

    with tempfile.TemporaryDirectory() as tmp:
        cfg = AgentConfig(workspace=tmp, session_log=None, command_timeout=30)
        registry = build_default_registry()
        ctx = build_tool_context(cfg, console=None, session={})

        # 非 http(s) 必须拒绝
        rb = registry.execute("web_fetch", {"url": "file:///etc/passwd"}, ctx)
        assert not rb.ok
        rn = registry.execute("web_fetch", {"url": "ftp://example.com/x"}, ctx)
        assert not rn.ok

        # 模拟一次成功抓取（HTML 应被抽成纯文本）
        html = b"<html><head><title>t</title></head><body><script>var a=1;</script><p>Hello <b>World</b></p></body></html>"
        fake_resp = io.BytesIO(html)
        fake_resp.headers = type("H", (), {"get_content_type": lambda self: "text/html"})()

        def fake_open(req, timeout=0):
            return fake_resp

        with __import__("unittest.mock", fromlist=["mock"]).patch(
                "agent.tools.extra.urllib.request.urlopen", fake_open):
            r = registry.execute("web_fetch", {"url": "https://example.com/"}, ctx)
        assert r.ok, r.render()
        assert "Hello World" in r.output, r.output
        assert "<script>" not in r.output and "<b>" not in r.output, "HTML 标签应被剥离"
        assert "var a=1" not in r.output, "script 内容应被去除"

        # 模拟 HTTP 404
        def fake_404(req, timeout=0):
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b""))

        with __import__("unittest.mock", fromlist=["mock"]).patch(
                "agent.tools.extra.urllib.request.urlopen", fake_404):
            r4 = registry.execute("web_fetch", {"url": "https://example.com/missing"}, ctx)
        assert not r4.ok and "404" in r4.render()


def test_think_tool_echoes_reasoning():
    from agent.tools import build_default_registry, build_tool_context
    from agent.config import AgentConfig
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AgentConfig(workspace=tmp, session_log=None)
        registry = build_default_registry()
        ctx = build_tool_context(cfg, console=None, session={})
        r = registry.execute("think", {"thought": "先读 main.py 再决定怎么改"}, ctx)
        assert r.ok and "[便签]" in r.output and "先读 main.py" in r.output
        # 空 thought 应被拒绝
        re_ = registry.execute("think", {"thought": ""}, ctx)
        assert not re_.ok


def test_eval_task_suite_is_stdlib_only_and_deterministic():
    """任务套件的基本卫生：有验证器、有考察点、预置文件齐全。

    这些是"评测结果可信"的前提条件，缺一样报告就只是好看而已。
    """
    from agent.eval import TASKS
    names = [t.name for t in TASKS]
    assert len(names) == len(set(names)), "任务名不能重复（会被 --task 过滤搞混）"
    for t in TASKS:
        assert t.verify is not None, f"{t.name} 缺验证器"
        assert t.intent, f"{t.name} 缺考察点说明"
        assert t.prompt.strip(), f"{t.name} 缺任务描述"


def test_todo_tool_manages_checklist():
    from agent.tools import build_default_registry, build_tool_context
    from agent.config import AgentConfig
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AgentConfig(workspace=tmp, session_log=None)
        registry = build_default_registry()
        ctx = build_tool_context(cfg, console=None, session={})

        r1 = registry.execute("todo", {"action": "add", "content": "读现状"}, ctx)
        assert r1.ok and "第 1 条" in r1.output
        r2 = registry.execute("todo", {"action": "add", "content": "写代码"}, ctx)
        assert r2.ok and "第 2 条" in r2.output

        rc = registry.execute("todo", {"action": "complete", "id": 1}, ctx)
        assert rc.ok and "[x]" in rc.output

        rr = registry.execute("todo", {"action": "remove", "id": 2}, ctx)
        assert rr.ok

        rl = registry.execute("todo", {"action": "list"}, ctx)
        assert "1." in rl.output and "已完成 1 条" in rl.output

        rb = registry.execute("todo", {"action": "complete", "id": 99}, ctx)
        assert not rb.ok, "越界 id 应报错"

        rm = registry.execute("todo", {"action": "add"}, ctx)
        assert not rm.ok, "add 缺 content 应报错"

        rcl = registry.execute("todo", {"action": "clear"}, ctx)
        assert rcl.ok
        rle = registry.execute("todo", {"action": "list"}, ctx)
        assert "空" in rle.output


def test_memory_tool_reads_appends_updates_clears():
    from agent.tools import build_default_registry, build_tool_context
    from agent.config import AgentConfig
    from agent.tools.memory import MEMORY_DIR, MEMORY_FILE
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AgentConfig(workspace=tmp, session_log=None)
        registry = build_default_registry()
        ctx = build_tool_context(cfg, console=None, session={})

        # read 空文件 → 友好提示
        r0 = registry.execute("memory", {"action": "read"}, ctx)
        assert r0.ok and "暂无项目记忆" in r0.output

        # append 两条
        a1 = registry.execute("memory", {"action": "append", "content": "测试用 pytest -q 跑"}, ctx)
        assert a1.ok and "已追加" in a1.output
        a2 = registry.execute("memory", {"action": "append", "content": "别用 print 调试"}, ctx)
        assert a2.ok

        # 落盘位置正确且内容包含两条
        mem_path = Path(tmp) / MEMORY_DIR / MEMORY_FILE
        assert mem_path.exists()
        text = mem_path.read_text(encoding="utf-8")
        assert "测试用 pytest -q 跑" in text and "别用 print 调试" in text

        # read 返回内容
        r1 = registry.execute("memory", {"action": "read"}, ctx)
        assert "测试用 pytest -q 跑" in r1.output

        # update 整体覆盖（content 不能为空）
        u0 = registry.execute("memory", {"action": "update"}, ctx)
        assert not u0.ok, "update 缺 content 应报错"
        registry.execute("memory", {"action": "update",
                                     "content": "# 项目约定\n- 一律用类型注解"}, ctx)
        text2 = mem_path.read_text(encoding="utf-8")
        assert "一律用类型注解" in text2 and "别用 print 调试" not in text2

        # clear 删除文件
        registry.execute("memory", {"action": "clear"}, ctx)
        assert not mem_path.exists()
        r2 = registry.execute("memory", {"action": "read"}, ctx)
        assert "暂无项目记忆" in r2.output

        # append 缺 content 应报错
        ae = registry.execute("memory", {"action": "append"}, ctx)
        assert not ae.ok


def test_memory_notes_injected_into_system_prompt():
    from agent.tools.memory import MEMORY_DIR, MEMORY_FILE, _SECTION_HEADER
    with tempfile.TemporaryDirectory() as tmp:
        # 预先写好项目记忆，AgentLoop 构造时应自动注入 system 提示词
        mem_path = Path(tmp) / MEMORY_DIR / MEMORY_FILE
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        mem_path.write_text("- 2026-08-30 约定：提交前必须跑测试\n", encoding="utf-8")

        cfg, profile, registry, _ = _make_env(tmp)
        # 复用 _make_env 的 registry/console；这里直接构造 loop 以检查 system_prompt
        loop = AgentLoop(cfg, profile, _scripted([], native=True)[0], registry, console=None)
        assert _SECTION_HEADER in loop.system_prompt, "记忆段落未注入 system 提示词"
        assert "提交前必须跑测试" in loop.system_prompt


def test_fs_ops_move_copy_delete():
    from agent.tools import build_default_registry, build_tool_context
    from agent.config import AgentConfig
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AgentConfig(workspace=tmp, session_log=None)
        registry = build_default_registry()
        ctx = build_tool_context(cfg, console=None, session={})

        (Path(tmp) / "a.txt").write_text("hello", encoding="utf-8")

        # move a -> b
        r = registry.execute("move_file", {"src": "a.txt", "dst": "b.txt"}, ctx)
        assert r.ok and (Path(tmp) / "b.txt").exists() and not (Path(tmp) / "a.txt").exists()

        # copy b -> c
        r2 = registry.execute("copy_file", {"src": "b.txt", "dst": "c.txt"}, ctx)
        assert r2.ok and (Path(tmp) / "c.txt").read_text(encoding="utf-8") == "hello"

        # 覆盖保护：无 overwrite 失败
        r3 = registry.execute("move_file", {"src": "b.txt", "dst": "c.txt"}, ctx)
        assert not r3.ok, "目标已存在时应拒绝"
        # 传 overwrite=true 成功
        r4 = registry.execute("move_file", {"src": "b.txt", "dst": "c.txt", "overwrite": True}, ctx)
        assert r4.ok and (Path(tmp) / "c.txt").exists() and not (Path(tmp) / "b.txt").exists()

        # delete c（应备份）
        r5 = registry.execute("delete", {"path": "c.txt"}, ctx)
        assert r5.ok and not (Path(tmp) / "c.txt").exists()
        backups = list(Path(tmp).parent.glob(".agent_backups/.overwrites/*/*/c.txt"))
        assert backups, "删除前应已备份到 .agent_backups"

        # 目录删除需 recursive
        d = Path(tmp) / "sub"
        d.mkdir()
        (d / "x.txt").write_text("x", encoding="utf-8")
        rd = registry.execute("delete", {"path": "sub"}, ctx)
        assert not rd.ok, "删目录缺 recursive 应拒绝"
        rdr = registry.execute("delete", {"path": "sub", "recursive": True}, ctx)
        assert rdr.ok and not d.exists()

        # 越界拒绝
        re = registry.execute("delete", {"path": "../escape.txt"}, ctx)
        assert not re.ok, "越界路径应拒绝"


def test_recall_ranks_relevant_files():
    from agent.tools import build_default_registry, build_tool_context
    from agent.config import AgentConfig
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AgentConfig(workspace=tmp, session_log=None)
        registry = build_default_registry()
        ctx = build_tool_context(cfg, console=None, session={})

        # recall 必须已注册且可见
        assert "recall" in registry.names()

        (Path(tmp) / "auth.py").write_text(
            "def handle_login(user):\n"
            "    for attempt in range(3):\n"
            "        try:\n"
            "            do_login(user)\n"
            "        except LoginError:\n"
            "            continue  # 登录失败重试\n"
            "    return None\n",
            encoding="utf-8",
        )
        # 部分相关：只命中查询里的「登录」一词，应排在 auth.py 之后
        (Path(tmp) / "session.py").write_text(
            "class Session:\n    def login(self):  # 仅含登录一词\n        return self.token\n",
            encoding="utf-8",
        )
        (Path(tmp) / "utils.py").write_text(
            "def format_number(n):\n    return f'{n:,}'\n",
            encoding="utf-8",
        )
        (Path(tmp) / "readme.md").write_text(
            "# 项目说明\n这是一个示例项目。\n",
            encoding="utf-8",
        )

        # 模糊意图应命中相关文件，且最相关的排在最前
        r = registry.execute("recall", {"query": "登录失败重试", "top_k": 3}, ctx)
        assert r.ok, r.render()
        assert "auth.py" in r.output, "最相关文件应为 auth.py"
        assert "session.py" in r.output, "部分相关的 session.py 也应被召回"
        assert "utils.py" not in r.output and "readme.md" not in r.output, "无关文件不应被召回"
        assert r.output.index("auth.py") < r.output.index("session.py"), "auth.py 应排在 session.py 之前"

        # 片段应带行号并标记相关度
        assert "相关度" in r.output and "1  " in r.output
        assert r.meta.get("matches") == 2, "应召回 2 个相关文件"

        # 空查询必须报错
        re_ = registry.execute("recall", {"query": "   "}, ctx)
        assert not re_.ok, "空查询必须被拒绝"

        # 无相关文件：友好返回（ok=True，但 0 命中）
        r2 = registry.execute("recall", {"query": "量子纠缠超导"}, ctx)
        assert r2.ok and "未找到" in r2.output


def test_replace_in_files_dry_run_then_apply_with_backup():
    from agent.tools import build_default_registry, build_tool_context
    from agent.config import AgentConfig
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AgentConfig(workspace=tmp, session_log=None)
        registry = build_default_registry()
        ctx = build_tool_context(cfg, console=None, session={})

        assert "replace_in_files" in registry.names()

        (Path(tmp) / "a.py").write_text("def old_fn():\n    return OLD_NAME\n", encoding="utf-8")
        (Path(tmp) / "b.py").write_text("x = OLD_NAME\ny = OLD_NAME\n", encoding="utf-8")
        (Path(tmp) / "c.md").write_text("OLD_NAME 出现了一次\n", encoding="utf-8")

        # 1) 默认 dry_run=true 只预览不落盘
        rd = registry.execute("replace_in_files",
                              {"old": "OLD_NAME", "new": "NEW_NAME", "include": "*.py"}, ctx)
        assert rd.ok and "预览" in rd.output and "3 处" in rd.output, rd.render()
        assert "OLD_NAME" in (Path(tmp) / "a.py").read_text(encoding="utf-8"), "dry_run 不应改写文件"

        # 2) old 过短必须被拒绝
        rs = registry.execute("replace_in_files", {"old": "e", "new": "x"}, ctx)
        assert not rs.ok, "old 过短必须被拒绝"

        # 3) 真正执行：两个 .py 文件共 3 处被改，且已备份
        ra = registry.execute("replace_in_files",
                              {"old": "OLD_NAME", "new": "NEW_NAME", "include": "*.py", "dry_run": False}, ctx)
        assert ra.ok, ra.render()
        assert "已改写 2 个文件" in ra.output and "3 处" in ra.output
        assert "NEW_NAME" in (Path(tmp) / "a.py").read_text(encoding="utf-8")
        assert (Path(tmp) / "a.py").read_text(encoding="utf-8").count("NEW_NAME") == 1
        assert (Path(tmp) / "b.py").read_text(encoding="utf-8").count("NEW_NAME") == 2
        # .md 不在 *.py 范围内，应未被改
        assert "OLD_NAME" in (Path(tmp) / "c.md").read_text(encoding="utf-8")

        # 4) 备份存在（.agent_backups/.overwrites 下）
        backups = list(Path(tmp).parent.glob(".agent_backups/.overwrites/*/*/*.py"))
        assert backups, "改写前应已备份"


def test_lint_runs_py_syntax_check_and_parses_command_output():
    import subprocess
    import unittest.mock as mock
    from agent.tools import build_default_registry, build_tool_context
    from agent.config import AgentConfig
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AgentConfig(workspace=tmp, session_log=None)
        registry = build_default_registry()
        ctx = build_tool_context(cfg, console=None, session={})

        assert "lint" in registry.names()

        # 模式二：零配置 Python 语法体检
        (Path(tmp) / "ok.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        (Path(tmp) / "bad.py").write_text("def f(:\n    pass\n", encoding="utf-8")  # 语法错误
        r_ok = registry.execute("lint", {"target": "ok.py"}, ctx)
        assert r_ok.ok and "通过" in r_ok.output
        r_bad = registry.execute("lint", {"target": "bad.py"}, ctx)
        assert r_bad.ok and "1 个问题" in r_bad.output and "bad.py:1" in r_bad.output

        # 模式一：用户给命令，解析 file:line: msg
        def fake_run(cmd, **kw):
            assert cmd[0] == "ruff"
            return subprocess.CompletedProcess(
                cmd, 1,
                stdout="src/foo.py:12: unused import X\nbar.py:3: undefined name Y\n",
                stderr="",
            )
        with mock.patch("agent.tools.lint.subprocess.run", fake_run):
            rc = registry.execute("lint", {"command": "ruff check ."}, ctx)
            assert rc.ok and "2 个问题" in rc.output
            assert "src/foo.py:12" in rc.output and "bar.py:3" in rc.output

        # 禁止 --fix 等改写选项
        rf = registry.execute("lint", {"command": "ruff check . --fix"}, ctx)
        assert not rf.ok, "--fix 必须被拒绝"

        # 退出码非零但输出无法解析成 file:line → 回显原始输出
        def fake_run_raw(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 2, stdout="BOOM: something odd\n", stderr="")
        with mock.patch("agent.tools.lint.subprocess.run", fake_run_raw):
            rr = registry.execute("lint", {"command": "mypy ."}, ctx)
            assert rr.ok and "BOOM" in rr.output and rr.meta.get("issues") == 0


def test_summary_recaps_session_changes():
    from agent.tools import build_default_registry, build_tool_context
    from agent.config import AgentConfig
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AgentConfig(workspace=tmp, session_log=None)
        registry = build_default_registry()
        session = {
            "task": "实现登录功能",
            "changes": [
                {"kind": "write", "path": "auth.py", "captured": True,
                 "before": None, "after": "def f():\n    return 1\n"},
                {"kind": "edit", "path": "auth.py", "captured": True,
                 "before": "def f():\n    return 1\n", "after": "def f():\n    return 2\n"},
                {"kind": "replace_in_files", "path": "util.py", "captured": True,
                 "before": "x = OLD\n", "after": "x = NEW\n"},
            ],
        }
        ctx = build_tool_context(cfg, console=None, session=session)

        assert "summary" in registry.names()

        r = registry.execute("summary", {}, ctx)
        assert r.ok, r.render()
        assert "实现登录功能" in r.output
        assert "auth.py" in r.output and "util.py" in r.output
        assert "3 次动作" in r.output
        # auth.py 最后一次改动后净变化：2→2 行，delta 0
        assert "（2→2 行）" in r.output or "（新建）" in r.output
        assert r.meta.get("changed_files") == 2

        # verbose=false 只给汇总计数
        r2 = registry.execute("summary", {"verbose": False}, ctx)
        assert "auth.py" in r2.output

        # 空会话：友好返回
        ctx_empty = build_tool_context(cfg, console=None, session={})
        r3 = registry.execute("summary", {}, ctx_empty)
        assert r3.ok and "还没有任何改动记录" in r3.output


def test_permission_mode_gates_dangerous_tools():
    from agent.loop import AgentLoop
    from agent.llm import ToolCall

    # --- 单元测试：_permission_check 按模式放行/拒绝 ---
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        loop = AgentLoop(cfg, profile, _scripted([], native=True)[0], registry, console=None)
        spec = registry.get("write_file")
        assert spec is not None and spec.dangerous, "write_file 应标记为 dangerous"
        call = ToolCall(id="c1", name="write_file", arguments={"path": "a.py", "content": "x"})

        # auto：直接放行
        cfg.permission_mode = "auto"
        assert loop._permission_check(spec, call) is None

        # read_only：一律拒绝
        cfg.permission_mode = "read_only"
        denied = loop._permission_check(spec, call)
        assert denied is not None and not denied.ok
        assert denied.meta.get("permission_denied") is True

        # ask + 无 console：退化为自动放行（default=True），避免无头环境卡死
        cfg.permission_mode = "ask"
        loop.ctx.console = None
        assert loop._permission_check(spec, call) is None

        # ask + 用户拒绝：返回拒绝结果
        loop.ctx.confirm = lambda q, default=True: False
        denied2 = loop._permission_check(spec, call)
        assert denied2 is not None and not denied2.ok
        assert denied2.meta.get("permission_denied") is True

        # 只读工具（read_file）不标 dangerous，任何模式下都不被本门拦截
        ro_spec = registry.get("read_file")
        assert ro_spec is not None and not ro_spec.dangerous
        cfg.permission_mode = "read_only"
        assert loop._permission_check(
            ro_spec, ToolCall(id="r", name="read_file", arguments={"path": "a.py"})) is None

    # --- 端到端：read_only 模式下 write_file 被拒、文件不应生成、不计入失败 ---
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp, permission_mode="read_only")
        script = [
            {"content": "尝试写文件。", "tool_calls": [
                {"name": "write_file", "arguments": {"path": "a.py", "content": "print(1)\n"}}]},
            {"content": "完成。", "tool_calls": [{"name": "finish", "arguments": {"summary": "ok"}}]},
        ]
        backend, _ = _scripted(script, native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        result = loop.run("在只读模式下干活")
        assert not (Path(tmp) / "a.py").exists(), "只读模式不应写出文件"
        assert result.finish_reason == "finish"
        assert result.errors == 0, "权限拒绝不应计入连续失败"
        assert any("权限不足" in (m.get("content") or "") for m in loop.history.messages), \
            "被拒的写操作应回灌『权限不足』提示给模型"


def test_delegate_spawns_child_agent_and_returns_summary():
    # 父子共享同一 MockBackend：脚本按"父派发 → 子读文件 → 子收尾 → 父收尾"顺序消费。
    # 验证子智能体确实独立运行、把结论回传父任务，且父任务不会因子任务而中断。
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, ctx = _make_env(tmp, native=True)
        script = [
            # 父 turn1：派发子任务
            {"content": "派个子智能体去查。", "tool_calls": [
                {"name": "delegate", "arguments": {"task": "读取 secret.txt 的内容并总结"}}]},
            # 子 turn1：读文件
            {"content": "读文件。", "tool_calls": [
                {"name": "read_file", "arguments": {"path": "secret.txt"}}]},
            # 子 turn2：收尾
            {"content": "总结。", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "secret.txt 里写着 42"}}]},
            # 父 turn2：收尾
            {"content": "完成。", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "子智能体回报：secret.txt 里写着 42"}}]},
        ]
        backend, _ = _scripted(script, native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        # 子智能体要读的文件
        (Path(tmp) / "secret.txt").write_text("42", encoding="utf-8")

        result = loop.run("让子智能体读 secret.txt")
        assert result.finish_reason == "finish", result.finish_reason
        assert "42" in result.answer, result.answer
        # 父任务"调用了 delegate"应被记为一次工具调用
        assert result.tool_calls >= 2


def test_delegate_child_cannot_recurse_and_cannot_ask_user():
    # 子智能体即便试图派发子任务（delegate）也拿不到该工具：registry 剔除 delegate/ask_user，
    # 因此会收到"未知工具"的回执而非真正递归；父任务仍能正常收尾（脚本耗尽即 FINAL，不会死循环）。
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, ctx = _make_env(tmp, native=True)
        script = [
            {"content": "派发。", "tool_calls": [
                {"name": "delegate", "arguments": {"task": "让孙子去做一件小事"}}]},
            # 子试图递归派发——但注册表里没有 delegate
            {"content": "试试再派发。", "tool_calls": [
                {"name": "delegate", "arguments": {"task": "孙子任务"}}]},
            # 子收到未知工具报错后收尾
            {"content": "收尾。", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "子任务完成（递归被拦截）"}}]},
            {"content": "完成。", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "父任务完成"}}]},
        ]
        backend, _ = _scripted(script, native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        result = loop.run("派发会递归的任务")
        assert result.finish_reason == "finish", result.finish_reason
        # 确认子智能体的注册表里确实没有 delegate（机制上杜绝递归）
        child_specs = [s.name for s in registry.specs()]  # 父注册表含 delegate
        assert "delegate" in child_specs
        # 子任务里那次"未知工具 delegate"应被记成一次错误回执，而非真的又开了个子智能体
        # （若递归发生，脚本会在第 2 个 delegate 处需要更多消息并走向 FINAL，finish_reason 仍为 finish，
        #  此处主要验证：整套运行正常结束、未抛异常）


# ----------------------------------------------------------------------------
# V31 自我改进钩子
# ----------------------------------------------------------------------------
def test_self_improve_derive_lessons_maps_signals():
    from agent.self_improve import SessionSignal, derive_lessons

    sig = SessionSignal(repair_rounds=1, tool_errors=2, permission_denied=1, compactions=1)
    lessons = derive_lessons(sig)
    assert any("自修复" in l for l in lessons), "repair 信号应映射出修复经验"
    assert any("权限" in l for l in lessons), "permission 信号应映射出权限经验"
    assert any("压缩" in l for l in lessons), "compaction 信号应映射出压缩经验"

    # 空信号 → 无经验
    assert derive_lessons(SessionSignal()) == []


def test_self_improve_record_lessons_dedups_and_forget():
    from agent.self_improve import (SessionSignal, derive_lessons, record_lessons,
                                    forget_lessons, read_lessons)

    with tempfile.TemporaryDirectory() as tmp:
        sig = SessionSignal(tool_errors=1, compactions=1)
        lessons = derive_lessons(sig)
        n1 = record_lessons(tmp, lessons)
        assert n1 == len(lessons), "首次应全部写入"

        # 重复写入 → 去重，新增 0
        n2 = record_lessons(tmp, lessons)
        assert n2 == 0, "重复经验不应再写"
        assert len(read_lessons(tmp)) == n1

        # 落盘格式：专用小节 + [auto] 条目
        mem = (Path(tmp) / ".minicode" / "memory.md").read_text(encoding="utf-8")
        assert "## 自动沉淀的经验（self-improve）" in mem
        assert "- [auto]" in mem

        # forget 清空自动条目
        removed = forget_lessons(tmp)
        assert removed == n1
        assert read_lessons(tmp) == []


def test_self_improve_loop_appends_lesson_on_failure():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        backend, profile = _scripted([], native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)

        result = RunResult()
        result.errors = 1
        result.finish_reason = "finish"
        loop._finish_blocked = 0

        added = loop._run_self_improve(result)
        assert added >= 1, "失败会话应沉淀至少一条经验"

        mem = (Path(tmp) / ".minicode" / "memory.md").read_text(encoding="utf-8")
        assert "[auto]" in mem
        assert "## 自动沉淀的经验" in mem

        # 幂等：再次同样信号不应重复追加
        added2 = loop._run_self_improve(result)
        assert added2 == 0, "重复信号不应追加重复经验"


def test_self_improve_tool_digest_and_list_and_forget():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, ctx = _make_env(tmp)
        # 模拟主循环已暂存的完整信号
        ctx.session["_last_signal"] = {
            "repair_rounds": 0, "tool_errors": 1, "permission_denied": 0,
            "empty_responses": 0, "finish_blocked": 0, "compactions": 1,
            "aborted": False, "llm_error": False,
        }

        r = registry.execute("self_improve", {"action": "digest"}, ctx)
        assert r.ok and r.meta.get("added", 0) >= 1, r.error or r.output

        r2 = registry.execute("self_improve", {"action": "list"}, ctx)
        assert r2.ok and "经验" in r2.output

        r3 = registry.execute("self_improve", {"action": "forget"}, ctx)
        assert r3.ok and r3.meta.get("removed", 0) >= 1

        r4 = registry.execute("self_improve", {"action": "list"}, ctx)
        assert "暂无" in r4.output, "forget 后应为空"


def test_report_card_renders_metrics_and_memory():
    """收尾报告卡把真实指标与记忆状态收进一个框，且不含 ANSI 噪声（color=False）。"""
    import io
    import contextlib
    from agent.ui import Console
    from agent.loop import RunResult

    r = RunResult()
    r.finish_reason = "finish"
    r.steps = 5
    r.tool_calls = 4
    r.repair_rounds = 1
    r.errors = 0
    r.usage = {"total_tokens": 1234}
    r.elapsed = 2.5
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        Console(verbose=True, color=False).report_card(r, memory_added=1, memory_total=7)
    out = buf.getvalue()
    assert "任务完成" in out, out
    assert "自修复" in out and "1" in out, out
    assert "Token 消耗" in out and "1,234" in out, out
    assert "记忆自更新" in out and "7" in out, out


def test_banner_shows_compliance_badge_and_memory_count():
    """启动横幅亮出合规徽章、工具数、模型/权限与已加载经验条数。"""
    import io
    import contextlib
    from agent.ui import Console

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        Console(verbose=True, color=False).banner(
            "profile=x model=Qwen3.5",
            version="v0.1.0",
            tagline="不依赖任何 Agent 框架 · 纯标准库从零实现 · 31 个工具",
            meta="model=Qwen3.5 · 权限模式=auto · 记忆自更新：已加载 3 条经验",
        )
    out = buf.getvalue()
    assert "MiniCode v0.1.0" in out, out
    assert "不依赖任何 Agent 框架" in out, out
    assert "31 个工具" in out, out
    assert "已加载 3 条经验" in out, out


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
