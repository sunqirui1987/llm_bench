"""HTTP 以外的 grok/codex/cmd 通道。"""

from __future__ import annotations

import io
import json
import unittest

from llm_bench.drivers.cmd import CmdDriver, render_cmd
from llm_bench.drivers.codex import CodexDriver
from llm_bench.drivers.grok import GrokDriver, grok_argv
from llm_bench.drivers.http import HttpDriver, as_driver
from llm_bench.drivers.registry import parse_via, resolve_driver
from llm_bench.drivers.types import StreamRequest, combined_prompt


class FakeStdin:
    def __init__(self):
        self.buf = bytearray()

    def write(self, data):
        self.buf.extend(data if isinstance(data, (bytes, bytearray)) else str(data).encode())

    def close(self):
        return None

    def getvalue(self):
        return bytes(self.buf)


class FakeProc:
    def __init__(self, lines, code=0, err=b""):
        payload = ("\n".join(lines) + "\n").encode()
        self.stdout = io.BytesIO(payload)
        self.stderr = io.BytesIO(err)
        self.stdin = FakeStdin()
        self.pid = 1
        self._code = code

    def poll(self):
        return self._code

    def wait(self, timeout=None):
        return self._code

    def terminate(self):
        pass

    def kill(self):
        pass


def popen_with(lines, code=0, err=b"", captured=None):
    def factory(argv, **kwargs):
        if captured is not None:
            captured.append({"argv": argv, "kwargs": kwargs})
        return FakeProc(lines, code=code, err=err)

    return factory


def req(**fields):
    base = dict(
        model="grok-4.6",
        messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USER"},
        ],
        max_tokens=32,
        timeout=5,
        reasoning_effort="xhigh",
    )
    base.update(fields)
    return StreamRequest(**base)


class DriverTest(unittest.TestCase):
    def test_parse_via_aliases(self):
        self.assertEqual(parse_via("http"), "http")
        self.assertEqual(parse_via("API"), "http")
        self.assertEqual(parse_via("grok"), "grok")
        self.assertEqual(parse_via("program"), "cmd")
        with self.assertRaises(ValueError):
            parse_via("ssh")

    def test_combined_prompt_keeps_system_and_user(self):
        text = combined_prompt(
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "USER"},
            ]
        )
        self.assertIn("SYS", text)
        self.assertIn("USER", text)

    def test_as_driver_wraps_protocol_objects(self):
        class Proto:
            name = "x"
            endpoint = "/v1"
            stream = lambda *a, **k: {}

        wrapped = as_driver(Proto())
        self.assertIsInstance(wrapped, HttpDriver)

    def test_grok_argv_is_completion_like(self):
        argv = grok_argv(
            binary="grok",
            model="grok-4.6",
            effort="xhigh",
            prompt_file="/tmp/p.txt",
        )
        joined = " ".join(argv)
        self.assertIn("--output-format streaming-json", joined)
        self.assertIn("--verbatim", joined)
        self.assertIn("--system-prompt-override", joined)
        self.assertNotIn("--max-turns", joined)
        self.assertIn("--effort xhigh", joined)
        self.assertIn("--prompt-file /tmp/p.txt", joined)
        self.assertNotIn("-p ", joined + " ")

    def test_grok_parses_streaming_json_and_cache(self):
        lines = [
            json.dumps({"type": "thought", "data": "plan"}),
            json.dumps({"type": "text", "data": "hello"}),
            json.dumps(
                {
                    "type": "end",
                    "usage": {
                        "input_tokens": 100,
                        "cache_read_input_tokens": 900,
                        "output_tokens": 4,
                    },
                }
            ),
        ]
        captured = []
        driver = GrokDriver("/opt/fake-grok", popen=popen_with(lines, captured=captured))
        result = driver.stream(req())
        self.assertEqual(result["text"], "hello")
        self.assertEqual(result["input_tokens"], 1000)
        self.assertEqual(result["cached_tokens"], 900)
        self.assertEqual(result["output_tokens"], 4)
        self.assertGreater(result["ttft_ms"], 0)
        argv = captured[0]["argv"]
        self.assertEqual(argv[0], "/opt/fake-grok")
        self.assertIn("xhigh", argv)

    def test_grok_nonzero_exit_raises(self):
        driver = GrokDriver("/opt/fake-grok", popen=popen_with([], code=2, err=b"boom"))
        with self.assertRaises(RuntimeError):
            driver.stream(req())

    def test_grok_max_turns_keeps_streamed_output(self):
        lines = [
            json.dumps({"type": "text", "data": "local Game = {}\n"}),
            json.dumps({"type": "error", "message": "max turns reached"}),
        ]
        driver = GrokDriver(
            "/opt/fake-grok",
            popen=popen_with(lines, code=1, err=b"Error: max turns reached"),
        )
        result = driver.stream(req())
        self.assertIn("local Game", result["text"])

    def test_cmd_renders_prompt_file_placeholder(self):
        argv = render_cmd(
            "my-llm --model {model} --file {prompt_file}",
            model="m",
            effort="high",
            max_tokens=8,
            prompt_file="/tmp/p.txt",
            session_id="",
        )
        self.assertEqual(argv, ["my-llm", "--model", "m", "--file", "/tmp/p.txt"])

    def test_cmd_driver_reads_plain_stdout(self):
        driver = CmdDriver(
            "echo {prompt_file}",
            popen=popen_with(["line-one", "line-two"]),
        )
        result = driver.stream(req())
        self.assertIn("line-one", result["text"])
        self.assertIn("line-two", result["text"])

    def test_codex_uses_stdin_and_turn_usage(self):
        captured = []
        lines = [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "i1", "type": "agent_message", "text": "ok"},
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 200,
                        "cached_input_tokens": 80,
                        "output_tokens": 2,
                    },
                }
            ),
        ]
        holders = []

        def factory(argv, **kwargs):
            proc = FakeProc(lines)
            holders.append(proc)
            captured.append({"argv": argv})
            return proc

        driver = CodexDriver("/opt/fake-codex", popen=factory)
        result = driver.stream(req())
        self.assertEqual(result["text"], "ok")
        self.assertEqual(result["input_tokens"], 200)
        self.assertEqual(result["cached_tokens"], 80)
        self.assertEqual(
            captured[0]["argv"][:4],
            ["/opt/fake-codex", "exec", "--json", "--ephemeral"],
        )
        self.assertIn("SYS", holders[0].stdin.getvalue().decode())
        self.assertIn("USER", holders[0].stdin.getvalue().decode())

    def test_resolve_driver_cmd_requires_template_at_stream(self):
        driver = resolve_driver("cmd", cmd="")
        with self.assertRaises(ValueError):
            driver.stream(req())

    def test_resolve_http_uses_named_protocol(self):
        driver = resolve_driver("http", format_name="responses")
        self.assertIsInstance(driver, HttpDriver)
        self.assertIn("responses", driver.endpoint)


if __name__ == "__main__":
    unittest.main()
