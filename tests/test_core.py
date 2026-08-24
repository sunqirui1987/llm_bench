"""公共配置、缓存口径、断流与 session 重试回归测试。"""

from __future__ import annotations

import threading
import unittest

import requests

from llm_bench import session
from llm_bench.config import resolve_base_urls
from llm_bench.metrics import StreamMeasurement
from llm_bench.prompts import pad_to_tokens
from llm_bench.reporting import aggregate_cache_percent, cache_percent
from llm_bench.transport import (
    HttpStatusError,
    call_with_retries,
    ensure_success,
    headers,
    iter_sse_lines,
)


class BrokenResponse:
    def __init__(self, lines):
        self.lines = lines

    def iter_lines(self, **_kwargs):
        yield from (line.encode() for line in self.lines)
        raise requests.exceptions.ChunkedEncodingError("Response ended prematurely")


class CoreTest(unittest.TestCase):
    def test_session_is_sent_with_proxy_safe_affinity_header(self):
        session.configure("stable-affinity")
        request_headers = headers("secret")
        self.assertEqual(request_headers["session_id"], "stable-affinity")
        self.assertEqual(
            request_headers["X-Session-Affinity"], "stable-affinity"
        )

    def test_empty_session_omits_affinity_headers(self):
        session.clear()
        request_headers = headers("secret", session_id="")
        self.assertNotIn("session_id", request_headers)
        self.assertNotIn("X-Session-Affinity", request_headers)

    def test_independent_base_urls(self):
        self.assertEqual(
            resolve_base_urls(
                "https://common/",
                "https://chat/",
                "https://responses/",
                "https://messages/",
            ),
            {
                "chat": "https://chat",
                "responses": "https://responses",
                "messages": "https://messages",
            },
        )

    def test_omitted_cached_field_is_cold_miss_when_input_exists(self):
        self.assertEqual(
            cache_percent(
                {
                    "input_tokens": 100,
                    "cached_tokens": 0,
                    "cache_reported": True,
                }
            ).strip(),
            "0.0%",
        )

    def test_aggregate_cache_skips_first_request(self):
        rows = [
            {
                "wave": 1,
                "input_tokens": 1000,
                "cached_tokens": 0,
                "cache_reported": True,
            },
            {
                "wave": 2,
                "input_tokens": 1000,
                "cached_tokens": 1000,
                "cache_reported": True,
            },
            {
                "wave": 3,
                "input_tokens": 1000,
                "cached_tokens": 1000,
                "cache_reported": True,
            },
        ]
        self.assertEqual(aggregate_cache_percent(rows).strip(), "66.7%")
        self.assertEqual(
            aggregate_cache_percent(rows, skip_first=True).strip(), "100.0%"
        )

    def test_aggregate_cache_skips_each_workers_first_success(self):
        rows = [
            {
                "worker": 1,
                "wave": 2,
                "input_tokens": 1000,
                "cached_tokens": 0,
                "cache_reported": True,
            },
            {
                "worker": 2,
                "wave": 1,
                "input_tokens": 1000,
                "cached_tokens": 0,
                "cache_reported": True,
            },
            {
                "worker": 2,
                "wave": 2,
                "input_tokens": 1000,
                "cached_tokens": 1000,
                "cache_reported": True,
            },
        ]
        self.assertEqual(
            aggregate_cache_percent(rows, skip_first=True).strip(), "100.0%"
        )

    def test_first_wave_cache_label_is_warmup(self):
        from llm_bench.reporting import _round_cache_label

        self.assertEqual(
            _round_cache_label(
                {"wave": 1, "input_tokens": 1000, "cached_tokens": 0},
                warmup=True,
            ),
            "预热·不计命中",
        )
        self.assertIn(
            "cache=100.0%",
            _round_cache_label(
                {
                    "wave": 2,
                    "input_tokens": 1000,
                    "cached_tokens": 1000,
                    "cache_turn": 2,
                },
                turn=2,
                warmup=False,
            ),
        )
        self.assertEqual(
            _round_cache_label(
                {"wave": 2, "input_tokens": 1000, "cached_tokens": 0},
                show_cache=False,
            ),
            "新命令",
        )

    def test_allow_incomplete_recovers_progressed_stream(self):
        state = {}
        lines = list(
            iter_sse_lines(
                BrokenResponse(
                    ['data: {"type":"response.reasoning_text.delta","delta":"x"}']
                ),
                allow_incomplete=True,
                stream_state=state,
            )
        )
        self.assertEqual(len(lines), 1)
        self.assertTrue(state["incomplete_stream"])

    def test_allow_incomplete_recovers_event_line_only_responses_delta(self):
        state = {}
        lines = list(
            iter_sse_lines(
                BrokenResponse(
                    [
                        "event: response.reasoning_text.delta",
                        'data: {"delta":"x"}',
                    ]
                ),
                allow_incomplete=True,
                stream_state=state,
            )
        )
        self.assertEqual(len(lines), 2)
        self.assertTrue(state["incomplete_stream"])

    def test_retry_rotates_session(self):
        session.configure("initial-session")
        seen = []

        def flaky():
            seen.append(session.current())
            if len(seen) < 3:
                raise requests.exceptions.ChunkedEncodingError("broken")
            return {}

        import llm_bench.transport as transport

        original_sleep = transport.time.sleep
        transport.time.sleep = lambda _seconds: None
        try:
            result = call_with_retries(flaky, (), 2, 0)
        finally:
            transport.time.sleep = original_sleep
        self.assertEqual(len(set(seen)), 3)
        self.assertEqual(result["retry_count"], 2)
        self.assertEqual(result["session_id"], session.current())

    def test_session_is_isolated_across_threads(self):
        seen = {}

        def worker(name: str):
            session.configure(name)
            seen[name] = (session.current(), headers("secret")["session_id"])

        threads = [
            threading.Thread(target=worker, args=("alpha",)),
            threading.Thread(target=worker, args=("beta",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(seen["alpha"], ("alpha", "alpha"))
        self.assertEqual(seen["beta"], ("beta", "beta"))

    def test_configure_return_matches_current_and_headers(self):
        configured = session.configure("stable-affinity")
        self.assertEqual(configured, "stable-affinity")
        self.assertEqual(session.current(), "stable-affinity")
        request_headers = headers("secret")
        self.assertEqual(request_headers["session_id"], "stable-affinity")
        self.assertEqual(request_headers["X-Session-Affinity"], "stable-affinity")

    def test_429_raises_http_status_error(self):
        class DummyResponse:
            status_code = 429
            text = '{"error":{"message":"Too many pending requests"}}'
            headers = {"Retry-After": "1.5"}

        with self.assertRaises(HttpStatusError) as ctx:
            ensure_success(DummyResponse())
        self.assertTrue(ctx.exception.rate_limited)
        self.assertEqual(ctx.exception.retry_after, 1.5)

    def test_429_bubbles_out_of_call_with_retries(self):
        def boom():
            raise HttpStatusError(429, "pending", 0.1)

        with self.assertRaises(HttpStatusError) as ctx:
            call_with_retries(boom, (), 5, 0)
        self.assertEqual(ctx.exception.status_code, 429)

    def test_503_is_capacity_limited_not_rate_limited(self):
        error = HttpStatusError(503, "No healthy Grok OAuth account is currently available")
        self.assertTrue(error.capacity_limited)
        self.assertTrue(error.retryable)
        self.assertFalse(error.rate_limited)

    def test_output_tps_from_stream_measurement(self):
        measurement = StreamMeasurement()
        measurement.started_at = 0.0
        measurement.first_at = 0.2
        measurement.last_at = 1.2
        measurement.chunk_times = [200.0, 200.0, 200.0, 200.0]
        measurement.text_parts = ["abcd"]
        result = measurement.finalize(
            input_tokens=100,
            output_tokens=5,
            cached_tokens=80,
            cache_reported=True,
            usage_reported=True,
            allow_empty=False,
            empty_error="empty",
        )
        self.assertAlmostEqual(result["ttft_ms"], 200.0, places=3)
        self.assertAlmostEqual(result["output_tps"], 4.0, places=3)
        self.assertAlmostEqual(result["tpot_ms"], 250.0, places=3)
        live = measurement.live_snapshot()
        self.assertGreater(live["out_tokens"], 0)
        self.assertIn("text", live)

    def test_pad_to_tokens_reaches_target(self):
        from llm_bench.prompts import estimate_tokens

        padded = pad_to_tokens("hello", 300, salt="bench")
        est = estimate_tokens(padded)
        self.assertGreater(est, 200)
        self.assertLessEqual(est, 300)

    def test_window_overhead_matches_gateway_budget(self):
        from llm_bench.prompts import (
            CONTEXT_WINDOW,
            DEFAULT_OUTPUT_RESERVE,
            _window_overhead,
            fit_max_input,
        )

        overhead = _window_overhead(CONTEXT_WINDOW)
        self.assertGreaterEqual(overhead, 24_000)
        cap = fit_max_input(CONTEXT_WINDOW, CONTEXT_WINDOW, CONTEXT_WINDOW)
        self.assertLessEqual(cap + DEFAULT_OUTPUT_RESERVE + overhead, CONTEXT_WINDOW)
        self.assertLessEqual(cap, 475_424)


if __name__ == "__main__":
    unittest.main()
