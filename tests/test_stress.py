"""RPM/TPM 统计与 429 自适应限流回归测试。"""

from __future__ import annotations

import unittest

from llm_bench.stress import AdaptiveGate, StressStats


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



    def test_live_footer_compose_is_short(self):
        from llm_bench.reporting import LiveFooter, WorkerBoard

        clock = Clock()
        stats = StressStats(workers=3, window=15, now=clock)
        board = WorkerBoard(3, waves=3)
        board.wave = 2
        board.finished.append(
            {
                "worker": 1,
                "wave": 1,
                "turn": 1,
                "input_tokens": 1000,
                "cached_tokens": 0,
                "cache_reported": True,
            }
        )
        board.update(
            1,
            phase="stream",
            turn=2,
            wave=2,
            ttft_ms=800,
            out_tokens=120,
            tok_s=40,
            started=0.0,
            input_tokens=1000,
            cached_tokens=800,
            cache_percent=80.0,
            cache_turn=2,
        )
        board.update(2, phase="wait", turn=1, wave=1, out_tokens=0, started=0.0)
        footer = LiveFooter(stats=stats, board=board)
        footer._tty = False
        lines = footer._compose(cols=80, rows=40)
        joined = "\n".join(lines)
        self.assertIn("work1", joined)
        self.assertIn("缓存", joined)
        self.assertIn("预热", joined)
        self.assertIn("cache=80.0%", joined)
        self.assertIn("RPM=", joined)
        self.assertIn("TPM=", joined)
        self.assertNotIn("快捷键", joined)
        self.assertTrue(any(line.strip() == "work1" for line in lines))
        self.assertEqual(joined.count("work1"), 1)
        rate_at = next(i for i, line in enumerate(lines) if "TPM=" in line)
        live_at = next(i for i, line in enumerate(lines) if "tok/s" in line)
        self.assertEqual(live_at, rate_at + 1)
        self.assertNotIn("tok/s", lines[rate_at])
        self.assertNotIn("TPM=", lines[live_at])

    def test_live_footer_miss_does_not_use_warmup_labels(self):
        from llm_bench.reporting import LiveFooter, WorkerBoard

        clock = Clock()
        stats = StressStats(workers=1, window=15, now=clock)
        board = WorkerBoard(1, waves=2, show_cache=False)
        board.wave = 2
        board.update(
            1,
            phase="stream",
            turn=2,
            wave=2,
            ttft_ms=400,
            out_tokens=10,
            tok_s=20,
            started=0.0,
            input_tokens=1000,
            cached_tokens=0,
        )
        footer = LiveFooter(stats=stats, board=board, show_cache=False)
        footer._tty = False
        joined = "\n".join(footer._compose(cols=80, rows=40))
        self.assertNotIn("预热", joined)
        self.assertNotIn("【缓存", joined)
        self.assertIn("换新", joined)
        self.assertIn("每次换新命令", joined)
        self.assertIn("新命令", joined)
        lines = footer._compose(cols=80, rows=40)
        status = [line for line in lines if "TPM=" in line or "tok/s" in line]
        self.assertEqual(len(status), 2)
        self.assertIn("TPM=", status[0])
        self.assertIn("每次换新命令", status[1])
        self.assertIn("tok/s", status[1])
        for line in status:
            self.assertFalse(line.endswith("…"))

    def test_live_footer_clears_screen_and_keeps_header(self):
        from io import StringIO
        from unittest.mock import patch

        from llm_bench.reporting import LiveFooter, WorkerBoard

        stats = StressStats(workers=1, window=15, now=Clock())
        footer = LiveFooter(
            stats=stats,
            board=WorkerBoard(1),
            header=["LLM Bench · 不要缓存  workers=10（一波全量线程）"],
        )
        footer._tty = True
        buf = StringIO()
        size = __import__("os").terminal_size((80, 40))
        with patch("sys.stdout", buf), patch(
            "llm_bench.reporting.shutil.get_terminal_size", return_value=size
        ):
            footer.refresh()
        out = buf.getvalue()
        self.assertTrue(out.startswith("\033[H\033[J"))
        self.assertIn("LLM Bench · 不要缓存  workers=10（一波全量线程）", out)
        self.assertEqual(out.count("work1"), 1)
        self.assertNotIn("快捷键", out)
        self.assertFalse(out.endswith("\n"))

    def test_compose_fits_small_terminal_with_many_workers(self):
        from llm_bench.reporting import LiveFooter, WorkerBoard, display_width

        stats = StressStats(workers=40, window=15, now=Clock())
        board = WorkerBoard(40, waves=2)
        board.wave = 1
        for i in range(1, 41):
            board.update(
                i,
                phase="stream",
                turn=1,
                wave=1,
                game="潮汐港务",
                out_tokens=120,
                tok_s=40,
                started=0.0,
                text="很长的模型输出 " * 20,
            )
        header = [
            "═" * 88,
            "LLM Bench · 要缓存  workers=40（一波全量线程）  rounds=2",
            "   chat      : http://127.0.0.1:8080",
            "   responses : http://127.0.0.1:8080",
            "   messages  : http://127.0.0.1:8080",
            "   models    : grok-4.6",
            "   formats   : responses",
            "   session   : sticky",
            "   cache     : hit",
            "   retries   : 2",
            "   window    : context=500000",
            "   prefix    : padded",
            "   prompt    : games",
            "═" * 88,
            "▶ responses  http://127.0.0.1:8080/v1/responses  model=grok-4.6",
        ]
        footer = LiveFooter(stats=stats, board=board, header=header)
        footer._tty = False
        lines = footer._compose(cols=80, rows=24)
        self.assertLessEqual(len(lines), 24)
        self.assertTrue(any("LLM Bench" in line for line in lines))
        self.assertRegex("\n".join(lines), r"(?<![\d])work1(?!\d)")
        for line in lines:
            self.assertLessEqual(display_width(line), 80)

    def test_draw_does_not_write_past_terminal(self):
        from io import StringIO
        from unittest.mock import patch
        import os

        from llm_bench.reporting import LiveFooter, WorkerBoard, display_width

        stats = StressStats(workers=30, window=15, now=Clock())
        board = WorkerBoard(30, waves=2)
        for i in range(1, 31):
            board.update(i, phase="wait", turn=1, wave=1, game="夜市烟火", started=0.0)
        footer = LiveFooter(
            stats=stats,
            board=board,
            header=["═" * 88, "LLM Bench · 不要缓存  workers=30（一波全量线程）"],
        )
        footer._tty = True
        buf = StringIO()
        size = os.terminal_size((60, 20))
        with patch("sys.stdout", buf), patch(
            "llm_bench.reporting.shutil.get_terminal_size", return_value=size
        ):
            footer.refresh()
        body = buf.getvalue().split("\033[H\033[J", 1)[-1]
        rows = body.split("\n")
        self.assertLessEqual(len(rows), 20)
        for row in rows:
            self.assertLessEqual(display_width(row), 60)
        self.assertFalse(body.endswith("\n"))

    def test_output_log_is_silent_and_writes_file(self):
        import tempfile
        from io import StringIO
        from pathlib import Path
        from unittest.mock import patch

        from llm_bench.reporting import OutputLog

        with tempfile.TemporaryDirectory() as folder:
            log = OutputLog(log_dir=folder)
            buf = StringIO()
            with patch("sys.stdout", buf):
                log.finish_step(1, 2, {"text": "Hello world!"})
            self.assertEqual(buf.getvalue(), "")
            saved = Path(folder) / "w01-round2.txt"
            self.assertEqual(saved.read_text(encoding="utf-8"), "Hello world!")

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
