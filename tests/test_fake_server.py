"""
端到端协议测试：在本机起一个"假的 OpenAI 兼容服务端"，验证真实 HTTP 链路。

这一层补上了 Mock 后端覆盖不到的部分：
    · 请求体是否按 OpenAI 规范组装（messages / tools / tool_choice / temperature）
    · 是否把工具 schema 正确传给服务端
    · 服务端返回 tool_calls 时，能否正确解析并驱动下一轮
    · 鉴权头、URL 拼接、超时、错误码是否正确处理

运行：python tests/test_fake_server.py（无需 API key，只监听 127.0.0.1）
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import AgentConfig, LLMProfile  # noqa: E402
from agent.llm import RawHTTPBackend  # noqa: E402
from agent.loop import AgentLoop  # noqa: E402
from agent.tools import build_default_registry  # noqa: E402

RECEIVED: list = []

# 服务端脚本：第一轮要求写文件，第二轮要求运行命令，第三轮结束
SCRIPT = [
    {
        "content": "创建脚本。",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"path": "hello.py", "content": "print('from fake server')\n"}),
                },
            }
        ],
    },
    {
        "content": "运行它。",
        "tool_calls": [
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "run_command", "arguments": json.dumps({"command": "python hello.py"})},
            }
        ],
    },
    {
        "content": "完成",
        "tool_calls": [
            {
                "id": "call_3",
                "type": "function",
                "function": {"name": "finish", "arguments": json.dumps({"summary": "fake server 验证通过"})},
            }
        ],
    },
]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # 静音
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        RECEIVED.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})

        token = (self.headers.get("Authorization") or "").removeprefix("Bearer ")
        if token != "test-key-1234567890":
            self._send(401, {"error": {"message": "missing auth"}})
            return

        idx = min(len(RECEIVED) - 1, len(SCRIPT) - 1)
        step = SCRIPT[idx]
        self._send(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": body.get("model", "fake"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": step["content"], "tool_calls": step["tool_calls"]},
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    def _send(self, code: int, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    failures = []

    try:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AgentConfig(workspace=tmp, session_log=None, max_steps=10)
            profile = LLMProfile(
                base_url=f"http://127.0.0.1:{port}/v1",
                api_key="test-key-1234567890",
                model="fake-model",
                native_tools=True,
            )
            backend = RawHTTPBackend(profile)
            registry = build_default_registry()
            loop = AgentLoop(cfg, profile, backend, registry, console=None)
            result = loop.run("创建一个脚本并运行它")

            # 1) 任务应正常完成
            if result.finish_reason != "finish":
                failures.append(f"finish_reason={result.finish_reason} answer={result.answer}")
            if result.answer != "fake server 验证通过":
                failures.append(f"answer={result.answer!r}")

            # 2) 文件真的被写出来了，且脚本真的被执行过
            created = Path(tmp) / "hello.py"
            if not created.exists() or "from fake server" not in created.read_text(encoding="utf-8"):
                failures.append("hello.py 未正确写入")

            # 3) 请求协议校验
            first = RECEIVED[0]
            if first["path"] != "/v1/chat/completions":
                failures.append(f"URL 拼接错误：{first['path']}")
            if first["auth"] != "Bearer test-key-1234567890":
                failures.append(f"鉴权头错误：{first['auth']}")
            body = first["body"]
            if body.get("model") != "fake-model":
                failures.append("model 未正确传递")
            if not body.get("tools") or not any(
                t["function"]["name"] == "write_file" for t in body["tools"]
            ):
                failures.append("工具 schema 未随请求发送")
            if body.get("tool_choice") != "auto":
                failures.append(f"tool_choice={body.get('tool_choice')}")
            if not any(m["role"] == "system" for m in body["messages"]):
                failures.append("system 提示词缺失")

            # 4) 多轮：工具回执应以 tool 角色回灌
            second = RECEIVED[1]["body"]["messages"]
            if not any(m.get("role") == "tool" for m in second):
                failures.append("工具回执未以 tool 角色回灌")

            # 5) 401 不会被无限重试
            bad = LLMProfile(base_url=f"http://127.0.0.1:{port}/v1", api_key="", model="fake", max_retries=1)
            from agent.errors import LLMError
            try:
                RawHTTPBackend(bad).chat([{"role": "user", "content": "hi"}])
                failures.append("401 未抛出 LLMError")
            except LLMError as exc:
                if exc.retryable:
                    failures.append("401 被误判为可重试")
    finally:
        server.shutdown()

    for f in failures:
        print(f"  FAIL  {f}")
    print(f"\n{'HTTP 链路全部通过' if not failures else str(len(failures)) + ' 项失败'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
