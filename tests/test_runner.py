"""同一条命令再发 vs 每次新命令。"""

from __future__ import annotations

import inspect
import unittest

from llm_bench import session
from llm_bench.config import parse_cache_mode
from llm_bench.conversation import Conversation
from llm_bench.engine import run_pool
import requests

from llm_bench.runner import _turn_request, bench
from llm_bench.transport import configure_pool, raise_fd_limit
import llm_bench.transport as transport


def result(cache_percent: float) -> dict:
    return {
        "input_tokens": 1000,
        "cached_tokens": int(cache_percent * 10),
        "cache_reported": True,
        "usage_reported": True,
        "retry_count": 0,
        "session_id": session.current(),
        "incomplete_stream": False,
        "ttft_ms": 80.0,
        "output_tps": 40.0,
        "text": "ok",
    }


class CaptureProtocol:
    def __init__(self):
        self.captured = []
        self.endpoint = "/v1/responses"

    def stream(
        self,
        base_url,
        api_key,
        model,
        system,
        user,
        max_tokens,
        timeout=180,
        messages=None,
        session_id="",
        on_progress=None,
    ):
        self.captured.append(
            {
                "messages": messages,
                "session_id_arg": session_id,
                "max_tokens": max_tokens,
            }
        )
        payload = result(80 if session_id else 0)
        payload["session_id"] = session_id
        return payload


class RunnerTest(unittest.TestCase):
    def test_parse_cache_mode_aliases(self):
        self.assertEqual(parse_cache_mode("miss"), "miss")
        self.assertEqual(parse_cache_mode("off"), "miss")
        self.assertEqual(parse_cache_mode("HIT"), "hit")
        self.assertEqual(parse_cache_mode("sticky"), "hit")
        with self.assertRaises(ValueError):
            parse_cache_mode("maybe")

    def test_bench_defaults(self):
        params = inspect.signature(bench).parameters
        self.assertEqual(params["cache_mode"].default, "hit")
        self.assertEqual(params["system"].default, "long")
        self.assertEqual(params["formats"].default, "responses")
        self.assertEqual(params["max_tokens"].default, 500000)
        self.assertEqual(params["rounds"].default, 2)
        self.assertNotIn("steps", params)
        self.assertEqual(params["workers"].default, 1)

    def test_bench_defaults_to_one_worker(self):
        self.assertEqual(configure_pool(16), 16)
        limit = raise_fd_limit(1024)
        self.assertTrue(limit == 0 or limit >= 1024)

    def test_hit_resends_the_same_command(self):
        protocol = CaptureProtocol()
        conv = Conversation(0, system="sys", user="hello", cache=True)
        first = _turn_request(
            protocol=protocol,
            base_url="https://example.test",
            api_key="key",
            model="model",
            conversation=conv,
            max_tokens=64,
            timeout=30,
            retries=0,
            retry_delay=0,
        )
        second = _turn_request(
            protocol=protocol,
            base_url="https://example.test",
            api_key="key",
            model="model",
            conversation=conv,
            max_tokens=64,
            timeout=30,
            retries=0,
            retry_delay=0,
        )
        captured = protocol.captured
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0]["session_id_arg"], conv.session_id)
        self.assertEqual(captured[1]["session_id_arg"], conv.session_id)
        self.assertEqual(captured[0]["messages"], captured[1]["messages"])
        self.assertEqual(first["session_id"], conv.session_id)
        self.assertEqual(second["session_id"], conv.session_id)

    def test_connection_retry_reuses_the_same_miss_payload(self):
        class Flaky(CaptureProtocol):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def stream(self, *args, messages=None, session_id="", **kwargs):
                self.calls += 1
                self.captured.append(messages)
                if self.calls == 1:
                    raise requests.exceptions.ConnectionError("boom")
                payload = result(0)
                payload["session_id"] = session_id
                return payload

        proto = Flaky()
        original_sleep = transport.time.sleep
        transport.time.sleep = lambda _seconds: None
        try:
            conv = Conversation(0, system="sys", user="hello", cache=False)
            _turn_request(
                protocol=proto,
                base_url="https://example.test",
                api_key="key",
                model="model",
                conversation=conv,
                max_tokens=64,
                timeout=30,
                retries=1,
                retry_delay=0,
            )
        finally:
            transport.time.sleep = original_sleep
        self.assertEqual(proto.calls, 2)
        self.assertEqual(proto.captured[0], proto.captured[1])

    def test_miss_sends_a_new_command_and_no_session(self):
        protocol = CaptureProtocol()
        conv = Conversation(0, system="sys", user="hello", cache=False)
        _turn_request(
            protocol=protocol,
            base_url="https://example.test",
            api_key="key",
            model="model",
            conversation=conv,
            max_tokens=64,
            timeout=30,
            retries=0,
            retry_delay=0,
        )
        _turn_request(
            protocol=protocol,
            base_url="https://example.test",
            api_key="key",
            model="model",
            conversation=conv,
            max_tokens=64,
            timeout=30,
            retries=0,
            retry_delay=0,
        )
        captured = protocol.captured
        self.assertEqual(conv.session_id, "")
        self.assertEqual(captured[0]["session_id_arg"], "")
        self.assertEqual(captured[1]["session_id_arg"], "")
        self.assertNotEqual(captured[0]["messages"], captured[1]["messages"])
        self.assertTrue(captured[0]["messages"][0]["content"].startswith("CACHE_BYPASS"))

    def test_write_report_groups_by_worker(self):
        import tempfile
        from pathlib import Path

        from llm_bench.reporting import write_report

        rows = [
            {
                "worker": 1,
                "wave": 1,
                "turn": 1,
                "ttft_ms": 100.0,
                "output_tps": 40.0,
                "e2e_ms": 200.0,
                "input_tokens": 1000,
                "cached_tokens": 0,
                "output_tokens": 10,
                "cache_reported": True,
            },
            {
                "worker": 2,
                "wave": 1,
                "turn": 1,
                "ttft_ms": 80.0,
                "output_tps": 50.0,
                "e2e_ms": 180.0,
                "input_tokens": 1000,
                "cached_tokens": 800,
                "output_tokens": 12,
                "cache_reported": True,
            },
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.md"
            write_report(
                path,
                meta={
                    "started_at": "2026-01-01 00:00:00",
                    "formats": ["responses"],
                    "models": ["demo"],
                    "cache_mode": "hit",
                    "workers": 2,
                    "rounds": 2,
                    "system": "long",
                    "base_url": "http://127.0.0.1:8080",
                },
                summary={("responses", "demo"): rows},
            )
            text = path.read_text(encoding="utf-8")
        self.assertIn("work1", text)
        self.assertIn("work2", text)
        self.assertIn("cache_mode: hit", text)
        self.assertIn("第 1 次命令", text)
        self.assertNotIn("steps:", text)

    def test_write_report_includes_full_output_text(self):
        import tempfile
        from pathlib import Path

        from llm_bench.reporting import write_report

        rows = [
            {
                "worker": 1,
                "wave": 1,
                "ttft_ms": 10.0,
                "output_tps": 20.0,
                "e2e_ms": 30.0,
                "input_tokens": 8,
                "cached_tokens": 0,
                "output_tokens": 4,
                "cache_reported": True,
                "text": "full model output body",
            }
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.md"
            write_report(
                path,
                meta={
                    "started_at": "2026-01-01 00:00:00",
                    "formats": ["responses"],
                    "models": ["demo"],
                    "cache_mode": "hit",
                    "workers": 1,
                    "rounds": 2,
                    "system": "long",
                    "base_url": "http://127.0.0.1:8080",
                },
                summary={("responses", "demo"): rows},
            )
            text = path.read_text(encoding="utf-8")
        self.assertIn("完整输出:", text)
        self.assertIn("full model output body", text)

    def test_run_pool_replays_full_worker_set_each_round(self):
        protocol = CaptureProtocol()
        rows, stats, _gate = run_pool(
            workers=2,
            rounds=2,
            duration=0,
            system="sys",
            user="hello",
            followup="",
            cache=True,
            session_prefix="pool",
            protocol=protocol,
            base_url="https://example.test",
            api_key="key",
            model="model",
            max_tokens=32,
            timeout=30,
            retries=0,
            retry_delay=0,
            throttle=0,
            max_input=200,
            context_window=2000,
            pad=True,
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(stats.snapshot().ok, 4)
        self.assertEqual({row["wave"] for row in rows}, {1, 2})
        self.assertNotIn("step", rows[0])
        self.assertEqual(len(protocol.captured), 4)
        by_session: dict[str, list] = {}
        for item in protocol.captured:
            by_session.setdefault(item["session_id_arg"], []).append(item["messages"])
        self.assertEqual(len(by_session), 2)
        for messages in by_session.values():
            self.assertGreaterEqual(len(messages), 1)
            self.assertTrue(all(item == messages[0] for item in messages))
        sessions = list(by_session)
        self.assertNotEqual(by_session[sessions[0]][0], by_session[sessions[1]][0])


if __name__ == "__main__":
    unittest.main()
