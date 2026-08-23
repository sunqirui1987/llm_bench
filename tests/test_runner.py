"""连续对话压测：session 开/关与缓存诊断。"""

from __future__ import annotations

import inspect
import unittest

from llm_bench import session
from llm_bench.config import parse_cache_mode
from llm_bench.conversation import Conversation
from llm_bench.runner import _turn_request, bench
from llm_bench.transport import configure_pool, raise_fd_limit


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


class RunnerTest(unittest.TestCase):
    def test_parse_cache_mode_aliases(self):
        self.assertEqual(parse_cache_mode("miss"), "miss")
        self.assertEqual(parse_cache_mode("off"), "miss")
        self.assertEqual(parse_cache_mode("HIT"), "hit")
        self.assertEqual(parse_cache_mode("sticky"), "hit")
        with self.assertRaises(ValueError):
            parse_cache_mode("maybe")

    def test_bench_defaults_to_hit_cache_mode(self):
        self.assertEqual(inspect.signature(bench).parameters["cache_mode"].default, "hit")
        self.assertEqual(inspect.signature(bench).parameters["system"].default, "long")
        self.assertEqual(inspect.signature(bench).parameters["formats"].default, "responses")
        self.assertEqual(inspect.signature(bench).parameters["max_tokens"].default, 500000)

    def test_bench_defaults_to_one_worker(self):
        self.assertEqual(inspect.signature(bench).parameters["workers"].default, 1)
        self.assertEqual(configure_pool(16), 16)
        limit = raise_fd_limit(1024)
        self.assertTrue(limit == 0 or limit >= 1024)

    def test_hit_turn_keeps_session_across_commits(self):
        captured = []

        class CaptureProtocol:
            @staticmethod
            def stream(
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
                captured.append(
                    {
                        "messages": messages,
                        "session_id_arg": session_id,
                        "max_tokens": max_tokens,
                    }
                )
                payload = result(80)
                payload["session_id"] = session_id
                return payload

        conv = Conversation(0, system="sys", user="hello", cache=True)
        first = _turn_request(
            protocol=CaptureProtocol,
            base_url="https://example.test",
            api_key="key",
            model="model",
            conversation=conv,
            max_tokens=64,
            timeout=30,
            retries=0,
            retry_delay=0,
        )
        conv.commit("ok")
        second = _turn_request(
            protocol=CaptureProtocol,
            base_url="https://example.test",
            api_key="key",
            model="model",
            conversation=conv,
            max_tokens=64,
            timeout=30,
            retries=0,
            retry_delay=0,
        )
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0]["session_id_arg"], conv.session_id)
        self.assertEqual(captured[1]["session_id_arg"], conv.session_id)
        self.assertTrue(conv.session_id)
        self.assertGreater(len(captured[1]["messages"]), len(captured[0]["messages"]))
        self.assertEqual(first["session_id"], conv.session_id)
        self.assertEqual(second["session_id"], conv.session_id)
        self.assertEqual(captured[0]["messages"][0]["content"], "sys")
        self.assertEqual(captured[1]["messages"][0]["content"], "sys")

    def test_write_report_groups_by_worker(self):
        import tempfile
        from pathlib import Path

        from llm_bench.reporting import write_report

        rows = [
            {
                "worker": 1,
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
                    "system": "long",
                    "base_url": "http://127.0.0.1:8080",
                },
                summary={("responses", "demo"): rows},
            )
            text = path.read_text(encoding="utf-8")
        self.assertIn("work1", text)
        self.assertIn("work2", text)
        self.assertIn("cache_mode: hit", text)

    def test_miss_turn_sends_no_session(self):
        captured = []

        class CaptureProtocol:
            @staticmethod
            def stream(
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
                captured.append({"session_id_arg": session_id, "messages": messages})
                payload = result(0)
                payload["session_id"] = session_id
                return payload

        conv = Conversation(0, system="sys", user="hello", cache=False)
        first = _turn_request(
            protocol=CaptureProtocol,
            base_url="https://example.test",
            api_key="key",
            model="model",
            conversation=conv,
            max_tokens=64,
            timeout=30,
            retries=0,
            retry_delay=0,
        )
        conv.commit("ok")
        second = _turn_request(
            protocol=CaptureProtocol,
            base_url="https://example.test",
            api_key="key",
            model="model",
            conversation=conv,
            max_tokens=64,
            timeout=30,
            retries=0,
            retry_delay=0,
        )
        self.assertEqual(conv.session_id, "")
        self.assertEqual(captured[0]["session_id_arg"], "")
        self.assertEqual(captured[1]["session_id_arg"], "")
        self.assertEqual(first["session_id"], "")
        self.assertEqual(second["session_id"], "")
        self.assertGreater(len(captured[1]["messages"]), len(captured[0]["messages"]))
        self.assertTrue(captured[0]["messages"][0]["content"].startswith("CACHE_BYPASS"))
        self.assertTrue(captured[1]["messages"][0]["content"].startswith("CACHE_BYPASS"))
        self.assertNotEqual(
            captured[0]["messages"][0]["content"],
            captured[1]["messages"][0]["content"],
        )


if __name__ == "__main__":
    unittest.main()
