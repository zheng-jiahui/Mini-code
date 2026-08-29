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
from agent.tools.base import DIFF_CAPTURE_CAP, ToolResult  # noqa: E402
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
    """写带 bug 的脚本 → 运行失败 → 循环回灌出错位置 → 模型读上下文修好 → 跑通。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, profile, registry, _ = _make_env(tmp)
        # 第 1 轮：写坏脚本（c 未定义）并运行（失败）；第 2 轮：模型据回灌的出错位置修好；第 3 轮：跑通并 finish
        script = [
            {"content": "写脚本", "tool_calls": [
                {"name": "write_file", "arguments": {"path": "calc.py",
                 "content": "def div(a, b):\n    return a / c\n"}}]},
            {"content": "运行", "tool_calls": [
                {"name": "run_command", "arguments": {"command": "python calc.py"}}]},
            {"content": "修", "tool_calls": [
                {"name": "edit_block", "arguments": {
                    "path": "calc.py", "old_text": "return a / c", "new_text": "return a / b"}}]},
            {"content": "再运行", "tool_calls": [
                {"name": "run_command", "arguments": {"command": "python calc.py"}}]},
            {"content": "完成", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "已修复并验证"}}]},
        ]
        backend, profile = _scripted(script, native=True)
        loop = AgentLoop(cfg, profile, backend, registry, console=None)
        result = loop.run("写个除法函数")
        assert result.finish_reason == "finish", result.finish_reason
        # 失败那轮，循环注入了带出错位置的提示
        joined = "\n".join(m.get("content", "") for m in loop.history.messages)
        assert "第 4 行" in joined or "calc.py" in joined   # 自修复上下文被回灌


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
