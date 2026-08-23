"""RPM/TPM 统计与 429 自适应限流回归测试。"""

from __future__ import annotations

import unittest

from llm_bench.stress import AdaptiveGate, StressReporter, StressStats


class Clock:
    def __init__(self, start: float = 0.0):
        self.t = start

    def __call__(self) -> float:
        return self.t


class StressStatsTest(unittest.TestCase):
    def test_rpm_tpm_from_one_success(self):
        clock = Clock()
        stats = StressStats(workers=10, window=15, now=clock)
        stats.begin()
        clock.t = 1.0
        stats.succeed(
            {
                "input_tokens": 100,
                "output_tokens": 200,
                "cached_tokens": 0,
                "ttft_ms": 80,
                "output_tps": 40,
            }
        )
        snap = stats.snapshot()
        self.assertEqual(snap.ok, 1)
        self.assertEqual(snap.fail, 0)
        self.assertAlmostEqual(snap.rpm, 60.0, places=3)
        self.assertAlmostEqual(snap.tpm, 18000.0, places=3)
        self.assertAlmostEqual(snap.tpm_in, 6000.0, places=3)
        self.assertAlmostEqual(snap.tpm_out, 12000.0, places=3)
        self.assertEqual(snap.cache_percent, 0.0)
        self.assertAlmostEqual(snap.ttft_avg, 80.0, places=3)
        self.assertAlmostEqual(snap.output_tps_avg, 40.0, places=3)

    def test_429_does_not_count_as_fail_sample(self):
        clock = Clock()
        stats = StressStats(workers=4, window=15, now=clock)
        stats.begin()
        stats.note_429()
        snap = stats.snapshot()
        self.assertEqual(snap.in_flight, 0)
        self.assertEqual(snap.rate_limited, 1)
        self.assertEqual(snap.fail, 0)
        self.assertEqual(snap.ok, 0)


    def test_fail_method_remains_callable(self):
        clock = Clock()
        stats = StressStats(workers=2, window=15, now=clock)
        stats.begin()
        clock.t = 1.0
        stats.fail()
        snap = stats.snapshot()
        self.assertEqual(snap.fail, 1)
        self.assertEqual(snap.ok, 0)
        self.assertTrue(callable(stats.fail))

    def test_unavailable_does_not_count_as_fail_sample(self):
        stats = StressStats(workers=4, window=15, now=Clock())
        stats.begin()
        stats.note_unavailable()
        snap = stats.snapshot()
        self.assertEqual(snap.in_flight, 0)
        self.assertEqual(snap.unavailable, 1)
        self.assertEqual(snap.fail, 0)
        self.assertEqual(snap.ok, 0)



    def test_reporter_can_hide_cache_percent(self):
        clock = Clock()
        stats = StressStats(workers=2, window=15, now=clock)
        stats.begin()
        clock.t = 1.0
        stats.succeed({"input_tokens": 100, "output_tokens": 20, "cached_tokens": 80})
        hidden = []
        StressReporter(stats, interval=5, printer=hidden.append, show_cache=False).emit()
        shown = []
        StressReporter(stats, interval=5, printer=shown.append, show_cache=True).emit()
        self.assertTrue(hidden)
        self.assertNotIn("cache=", hidden[0])
        self.assertIn("cache=", shown[0])
        self.assertIn("ttft=", shown[0])
        self.assertIn("tok/s=", shown[0])

    def test_reporter_uses_live_board_tokens(self):
        from llm_bench.reporting import WorkerBoard

        clock = Clock()
        stats = StressStats(workers=2, window=15, now=clock)
        board = WorkerBoard(2)
        board.update(1, phase="stream", turn=1, ttft_ms=800, out_tokens=120, tok_s=40, started=clock.t)
        board.update(2, phase="wait", turn=1, out_tokens=0, started=clock.t)
        printed = []
        reporter = StressReporter(stats, interval=1, printer=printed.append, show_cache=True)
        reporter.board = board
        reporter.emit()
        self.assertTrue(printed)
        self.assertIn("live_out=120", printed[0])
        self.assertIn("live_tok/s=", printed[0])
        self.assertIn("work1", printed[1])
        self.assertIn("out=120", printed[1])

    def test_live_footer_compose_is_short(self):
        from llm_bench.reporting import LiveFooter, WorkerBoard

        clock = Clock()
        stats = StressStats(workers=3, window=15, now=clock)
        board = WorkerBoard(3)
        board.update(
            1,
            phase="stream",
            turn=1,
            ttft_ms=800,
            out_tokens=120,
            tok_s=40,
            started=0.0,
            input_tokens=1000,
            cached_tokens=800,
            cache_percent=80.0,
            cache_turn=1,
        )
        board.update(2, phase="wait", turn=1, out_tokens=0, started=0.0)
        footer = LiveFooter(stats=stats, board=board)
        footer._tty = False
        lines = footer._compose()
        joined = "\n".join(lines)
        self.assertIn("work1", joined)
        self.assertIn("--round", joined)
        self.assertIn("cache=80.0%", joined)
        self.assertTrue(any(line.strip() == "work1" for line in lines))

class AdaptiveGateTest(unittest.TestCase):
    def test_rate_limit_halves_pending_limit(self):
        gate = AdaptiveGate(8)
        self.assertTrue(gate.acquire())
        self.assertEqual(gate.in_flight, 1)
        self.assertEqual(gate.limit, 8)
        gate.release(rate_limited=True)
        self.assertEqual(gate.in_flight, 0)
        self.assertEqual(gate.limit, 4)

    def test_success_slowly_raises_limit(self):
        gate = AdaptiveGate(4, initial=1)
        for _ in range(10):
            self.assertTrue(gate.acquire())
            gate.release(rate_limited=False)
        self.assertEqual(gate.limit, 2)


if __name__ == "__main__":
    unittest.main()
