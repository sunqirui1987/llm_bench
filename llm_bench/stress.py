"""并发压测的 RPM / TPM 统计与自适应在途限流。"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional


Now = Callable[[], float]


@dataclass(frozen=True)
class StressSnapshot:
    elapsed: float
    in_flight: int
    peak_in_flight: int
    workers: int
    ok: int
    fail: int
    rate_limited: int
    unavailable: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    rpm: float
    rpm_window: float
    peak_rpm: float
    tpm_in: float
    tpm_out: float
    tpm: float
    tpm_window: float
    peak_tpm: float
    ttft_avg: float
    output_tps_avg: float
    cache_percent: float


class StressStats:
    """线程安全地累计请求数、token 数，并给出整体/窗口 RPM、TPM。"""

    def __init__(self, workers: int, window: float = 15.0, now: Now = time.perf_counter):
        self.workers = max(int(workers), 1)
        self.window = max(float(window), 1.0)
        self._now = now
        self._lock = threading.Lock()
        self.started_at = now()
        self.in_flight = 0
        self.peak_in_flight = 0
        self.ok = 0
        self.fail_count = 0
        self.rate_limited = 0
        self.unavailable = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.ttft_ms_sum = 0.0
        self.ttft_count = 0
        self.output_tps_sum = 0.0
        self.output_tps_count = 0
        self.peak_rpm = 0.0
        self.peak_tpm = 0.0
        self._events: deque[tuple[float, bool, int, int, int]] = deque()

    def begin(self) -> None:
        with self._lock:
            self.in_flight += 1
            if self.in_flight > self.peak_in_flight:
                self.peak_in_flight = self.in_flight

    def succeed(self, result: dict) -> None:
        self._finish(
            ok=True,
            input_tokens=int(result.get("input_tokens") or 0),
            output_tokens=int(result.get("output_tokens") or 0),
            cached_tokens=int(result.get("cached_tokens") or 0),
            ttft_ms=result.get("ttft_ms"),
            output_tps=result.get("output_tps"),
        )

    def fail(self) -> None:
        self._finish(ok=False, input_tokens=0, output_tokens=0, cached_tokens=0)

    def note_429(self) -> None:
        """在途请求被限流拒绝：释放 inflight，不计失败样本。"""
        with self._lock:
            self.in_flight = max(self.in_flight - 1, 0)
            self.rate_limited += 1

    def note_unavailable(self) -> None:
        """上游 503/502/504：释放 inflight，退避后由 worker 重试同一轮。"""
        with self._lock:
            self.in_flight = max(self.in_flight - 1, 0)
            self.unavailable += 1

    def _finish(
        self,
        *,
        ok: bool,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        ttft_ms=None,
        output_tps=None,
    ) -> None:
        now = self._now()
        with self._lock:
            self.in_flight = max(self.in_flight - 1, 0)
            if ok:
                self.ok += 1
                self.input_tokens += max(input_tokens, 0)
                self.output_tokens += max(output_tokens, 0)
                self.cached_tokens += max(cached_tokens, 0)
                if ttft_ms is not None:
                    self.ttft_ms_sum += float(ttft_ms)
                    self.ttft_count += 1
                if output_tps is not None:
                    self.output_tps_sum += float(output_tps)
                    self.output_tps_count += 1
            else:
                self.fail_count += 1
            self._events.append(
                (now, ok, max(input_tokens, 0), max(output_tokens, 0), max(cached_tokens, 0))
            )
            self._trim(now)
            snap = self._snapshot_locked(now)
            self.peak_rpm = max(self.peak_rpm, snap.rpm_window)
            self.peak_tpm = max(self.peak_tpm, snap.tpm_window)

    def snapshot(self) -> StressSnapshot:
        now = self._now()
        with self._lock:
            self._trim(now)
            return self._snapshot_locked(now)

    def _trim(self, now: float) -> None:
        oldest = now - self.window
        while self._events and self._events[0][0] < oldest:
            self._events.popleft()

    def _snapshot_locked(self, now: float) -> StressSnapshot:
        elapsed = max(now - self.started_at, 1e-9)
        window_ok = 0
        window_tokens = 0
        for _ts, ok, input_tokens, output_tokens, _cached in self._events:
            if ok:
                window_ok += 1
                window_tokens += input_tokens + output_tokens
        window_seconds = min(elapsed, self.window)
        rpm_window = _per_minute(window_ok, window_seconds)
        tpm_window = _per_minute(window_tokens, window_seconds)
        input_tokens = self.input_tokens
        output_tokens = self.output_tokens
        cached = self.cached_tokens
        return StressSnapshot(
            elapsed=elapsed,
            in_flight=self.in_flight,
            peak_in_flight=self.peak_in_flight,
            workers=self.workers,
            ok=self.ok,
            fail=self.fail_count,
            rate_limited=self.rate_limited,
            unavailable=self.unavailable,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached,
            rpm=_per_minute(self.ok, elapsed),
            rpm_window=rpm_window,
            peak_rpm=max(self.peak_rpm, rpm_window),
            tpm_in=_per_minute(input_tokens, elapsed),
            tpm_out=_per_minute(output_tokens, elapsed),
            tpm=_per_minute(input_tokens + output_tokens, elapsed),
            tpm_window=tpm_window,
            peak_tpm=max(self.peak_tpm, tpm_window),
            cache_percent=(
                100.0 * cached / input_tokens if input_tokens > 0 else 0.0
            ),
            ttft_avg=(
                self.ttft_ms_sum / self.ttft_count if self.ttft_count else 0.0
            ),
            output_tps_avg=(
                self.output_tps_sum / self.output_tps_count
                if self.output_tps_count
                else 0.0
            ),
        )


def _per_minute(count: float, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return count / seconds * 60.0


class AdaptiveGate:
    """按 429 收缩、按连续成功缓慢抬升的在途上限。"""

    def __init__(self, max_concurrency: int, initial: int | None = None):
        self.max_concurrency = max(int(max_concurrency), 1)
        start = self.max_concurrency if initial is None else max(int(initial), 1)
        self.limit = min(start, self.max_concurrency)
        self.in_flight = 0
        self.success_streak = 0
        self._cv = threading.Condition()

    def acquire(self, stop_event: threading.Event | None = None) -> bool:
        with self._cv:
            while self.in_flight >= self.limit:
                if stop_event is not None and stop_event.is_set():
                    return False
                self._cv.wait(timeout=0.2)
            if stop_event is not None and stop_event.is_set():
                return False
            self.in_flight += 1
            return True

    def release(self, *, rate_limited: bool = False) -> None:
        with self._cv:
            self.in_flight = max(self.in_flight - 1, 0)
            if rate_limited:
                shrunk = max(1, self.limit // 2)
                if shrunk == self.limit and self.limit > 1:
                    shrunk -= 1
                self.limit = shrunk
                self.success_streak = 0
            else:
                self.success_streak += 1
                if self.success_streak >= 10 and self.limit < self.max_concurrency:
                    self.limit += 1
                    self.success_streak = 0
            self._cv.notify_all()


class StressReporter:
    """后台定时打印 RPM/TPM，不打断压测线程。"""

    def __init__(
        self,
        stats: StressStats,
        interval: float,
        printer: Callable[[str], None] = print,
        limit_provider: Callable[[], int] | None = None,
        show_cache: bool = True,
    ):
        self.stats = stats
        self.interval = max(float(interval), 0.5)
        self._print = printer
        self.limit_provider = limit_provider
        self.show_cache = bool(show_cache)
        self.board = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="llm-bench-rpm",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.emit()

    def emit(self) -> None:
        snap = self.stats.snapshot()
        limit = self.limit_provider() if self.limit_provider else snap.workers
        live = self.board.live_totals() if self.board is not None else None
        line = (
            "⏱ "
            f"{snap.elapsed:6.1f}s  inflight={snap.in_flight}/{limit}"
            f"  ok={snap.ok}  fail={snap.fail}  429={snap.rate_limited}  5xx={snap.unavailable}"
        )
        if live is not None:
            ttft = live["ttft_avg"]
            ttft_s = f"{ttft:.0f}ms" if ttft is not None else "wait"
            line += (
                f"  wait={live['waiting']} stream={live['streaming']}"
                f"  live_ttft={ttft_s}"
                f"  live_out={live['out_tokens']}"
                f"  live_tok/s={live['tok_s']:.1f}"
            )
        if snap.ok:
            line += (
                f"  done_ttft={snap.ttft_avg:.0f}ms"
                f"  done_tok/s={snap.output_tps_avg:.1f}"
                f"  rpm={snap.rpm_window:.1f}"
                f"  tpm={snap.tpm_window:.0f}"
            )
            if self.show_cache:
                line += f"  cache={snap.cache_percent:.1f}%"
        self._print(line)
        if self.board is not None:
            board = self.board.render()
            if board:
                self._print(board)
