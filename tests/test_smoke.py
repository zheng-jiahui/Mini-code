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
