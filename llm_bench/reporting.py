"""终端表格格式化与跨轮次聚合计算。"""

from __future__ import annotations

import shutil
import sys
import threading
import time
from typing import Optional


def format_ms(value: Optional[float]) -> str:
    return "      n/a" if value is None else f"{value:8.1f}ms"


def format_tpot(value: Optional[float]) -> str:
    return "       n/a  " if value is None else f"{value:7.2f}ms/t"


def format_tps(value: Optional[float]) -> str:
    return "     n/a" if value is None else f"{value:7.1f}/s"


def format_percent(hit: int, total: int, reported: bool = True) -> str:
    if not reported or not total:
        return "   n/a  "
    return f"{100.0 * hit / total:5.1f}%"


def cache_percent(result: dict) -> str:
    return format_percent(
        result["cached_tokens"],
        result["input_tokens"],
        result["cache_reported"],
    )


def is_first_request(row: dict) -> bool:
    return int(row.get("wave") or row.get("turn") or 1) <= 1


def warm_rows(rows: list[dict]) -> list[dict]:
    """去掉每个 worker 的第 1 次（冷启动），只留同样命令再来的样本。"""
    return [row for row in rows if not is_first_request(row)]


def aggregate_cache_percent(rows: list[dict], *, skip_first: bool = False) -> str:
    sample = warm_rows(rows) if skip_first else rows
    reported = [
        row for row in sample
        if row["cache_reported"] and row["input_tokens"] > 0
    ]
    if not reported:
        return "   n/a  "
    return format_percent(
        sum(row["cached_tokens"] for row in reported),
        sum(row["input_tokens"] for row in reported),
    )


def average(values: list) -> float:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else 0.0


def average_or_none(values: list) -> Optional[float]:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _p95(values: list) -> Optional[float]:
    present = sorted(value for value in values if value is not None)
    if not present:
        return None
    return present[min(len(present) - 1, int(len(present) * 0.95))]


def separator(width: int = 100) -> None:
    print("─" * width, flush=True)


def worker_label(worker: int) -> str:
    return f"work{max(int(worker), 1)}"


OUTPUT_TAIL_LINES = 4
OUTPUT_TAIL_WIDTH = 86


def live_tail_lines(workers: int, header_lines: int = 18) -> int:
    """按终端高度给每个 work 分尾部行数，避免 10 路把屏幕顶出去。"""
    workers = max(int(workers), 1)
    try:
        rows = int(shutil.get_terminal_size(fallback=(80, 40)).lines)
    except Exception:
        rows = 40
    budget = max(int(rows) - int(header_lines) - 2, workers * 2)
    per = budget // workers - 2
    return max(0, min(OUTPUT_TAIL_LINES, per))


class OutputLog:
    """不往控制台打字。结束时把全文写入 logs/，面板自己刷。"""

    def __init__(self, log_dir=None):
        self.log_dir = log_dir

    def start_step(self, *args, **kwargs) -> None:
        return

    def append_text(self, *args, **kwargs) -> None:
        return

    def finish_step(
        self,
        worker: int,
        wave: int,
        result: dict,
    ) -> None:
        if self.log_dir is None:
            return
        from pathlib import Path

        folder = Path(self.log_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"w{worker:02d}-round{wave}.txt"
        path.write_text(str(result.get("text") or ""), encoding="utf-8")

    def fail_step(self, *args, **kwargs) -> None:
        return


def _step_label(worker: int, wave: int, step: int, steps: int) -> str:
    return f"{worker_label(worker)} wave{wave} step{step}/{steps}"


def _round_cache_label(state: dict, turn: int | None = None) -> str:
    wave = int(state.get("wave") or turn or 1)
    if wave <= 1:
        return "预热·不计命中"
    inp = int(state.get("input_tokens") or 0)
    cached = int(state.get("cached_tokens") or 0)
    pct = state.get("cache_percent")
    cache_turn = int(state.get("cache_turn") or 0)
    if inp > 0:
        pct = 100.0 * cached / inp
        text = f"cache={pct:.1f}% ({cached}/{inp})"
    elif pct is None:
        return "等命中结果"
    else:
        text = f"cache={float(pct):.1f}%"
    if cache_turn and turn and cache_turn != turn:
        return f"上轮 {text}"
    return text


def print_bench_header(*, show_cache: bool = True) -> None:
    extra = f"{'cache%':>8}" if show_cache else ""
    print(
        "完成的对话从这里往上滚（每路固定 5 步，输入 1/5 → 满；rounds 是再拉起一波全量线程）：",
        flush=True,
    )
    print(
        f"{'对话':<8}{'波/步':<10}{'TTFT':>10}{'tok/s':>9}{'E2E':>10}"
        f"{'用户输入':>10}{'模型输出':>10}{extra}",
        flush=True,
    )
    separator()


def format_bench_row(index: int, result: dict, *, show_cache: bool = True) -> str:
    retry_mark = (
        f"  [retry={result['retry_count']}]" if result.get("retry_count") else ""
    )
    extra = f"{cache_percent(result):>8}" if show_cache else ""
    worker = worker_label(result.get("worker") or 1)
    wave = int(result.get("wave") or result.get("turn") or index)
    return (
        f"{worker:<8}r{wave:<3}"
        f"{format_ms(result['ttft_ms'])}"
        f"{format_tps(result.get('output_tps'))}"
        f"{format_ms(result['e2e_ms'])}"
        f"{int(result.get('input_tokens') or 0):>10}"
        f"{int(result.get('output_tokens') or 0):>10}"
        f"{extra}{retry_mark}"
    )


def print_bench_row(index: int, result: dict, *, show_cache: bool = True) -> None:
    print(format_bench_row(index, result, show_cache=show_cache), flush=True)


class WorkerBoard:
    """多路并发：每一波每人发一条命令。"""

    def __init__(self, workers: int, rounds: int | None = None, steps: int = 1, waves: int = 1):
        self.workers = max(int(workers), 1)
        total = int(rounds if rounds is not None else waves)
        self.waves = max(total, 1)
        self.steps = self.waves
        self.rounds = self.waves
        self.wave = 1
        self.tail_lines = OUTPUT_TAIL_LINES
        self.finished: list[dict] = []
        self._lock = threading.Lock()
        self._states = {
            i: {
                "phase": "idle",
                "turn": 0,
                "wave": 1,
                "done": 0,
                "ttft_ms": None,
                "chunks": 0,
                "chars": 0,
                "out_tokens": 0,
                "tok_s": 0.0,
                "started": None,
                "input_tokens": 0,
                "cached_tokens": 0,
                "cache_percent": None,
                "text": "",
                "game": "",
            }
            for i in range(1, self.workers + 1)
        }

    def set_wave(self, wave: int) -> None:
        self.wave = max(int(wave), 1)
        with self._lock:
            for state in self._states.values():
                state.update(phase="idle", turn=0, wave=self.wave, done=0, error="")

    def begin_round(
        self,
        worker: int,
        turn: int,
        wave: int | None = None,
        game: str = "",
    ) -> None:
        self.update(
            worker,
            phase="wait",
            turn=int(turn),
            wave=int(wave or self.wave),
            game=game or "",
            out_tokens=0,
            tok_s=0.0,
            error="",
            text="",
            started=time.perf_counter(),
        )

    def fail_round(self, worker: int, turn: int, error: str, wave: int | None = None) -> None:
        self.update(
            worker,
            phase="error",
            turn=int(turn),
            wave=int(wave or self.wave),
            error=str(error)[:80],
        )

    def finish_round(self, worker: int, turn: int, result: dict, wave: int | None = None) -> None:
        inp = int(result.get("input_tokens") or 0)
        cached = int(result.get("cached_tokens") or 0)
        self.update(
            worker,
            phase="idle",
            turn=int(turn),
            wave=int(wave or self.wave),
            done=int(turn),
            out_tokens=int(result.get("output_tokens") or 0),
            tok_s=float(result.get("output_tps") or 0),
            input_tokens=inp,
            cached_tokens=cached,
            cache_percent=(100.0 * cached / inp) if inp > 0 else None,
            cache_turn=int(turn),
            text=str(result.get("text") or ""),
            started=None,
        )
        with self._lock:
            self.finished.append(
                {
                    "worker": worker,
                    "wave": int(wave or turn),
                    "turn": int(turn),
                    "input_tokens": inp,
                    "cached_tokens": cached,
                    "cache_reported": True,
                }
            )

    def update(self, worker: int, **fields) -> None:
        worker = max(int(worker), 1)
        with self._lock:
            if worker not in self._states:
                self._states[worker] = {
                    "phase": "idle",
                    "turn": 0,
                    "wave": self.wave,
                    "done": 0,
                    "ttft_ms": None,
                    "chunks": 0,
                    "chars": 0,
                    "out_tokens": 0,
                    "tok_s": 0.0,
                    "started": None,
                    "input_tokens": 0,
                    "cached_tokens": 0,
                    "cache_percent": None,
                    "text": "",
                    "game": "",
                }
            self._states[worker].update(fields)

    def on_progress(self, worker: int, turn: int):
        def _cb(snap: dict) -> None:
            self.update(
                worker,
                phase="stream" if snap.get("ttft_ms") is not None else "wait",
                turn=turn,
                ttft_ms=snap.get("ttft_ms"),
                chunks=snap.get("chunks") or 0,
                chars=snap.get("chars") or 0,
                out_tokens=snap.get("out_tokens") or 0,
                tok_s=snap.get("tok_s") or 0.0,
                text=snap.get("text") or "",
            )
        return _cb

    def render(self) -> str:
        now = time.perf_counter()
        with self._lock:
            items = sorted(self._states.items())
        cells = []
        for worker, state in items:
            phase = state.get("phase") or "idle"
            turn = int(state.get("turn") or 0)
            started = state.get("started")
            elapsed = (now - started) if started else 0.0
            if phase == "idle":
                cells.append(f"{worker_label(worker)} idle")
            elif phase == "wait":
                cells.append(f"{worker_label(worker)} r{turn} wait {elapsed:.0f}s out=0")
            else:
                ttft = state.get("ttft_ms")
                ttft_s = f"{ttft:.0f}ms" if ttft is not None else "..."
                out_tokens = int(state.get("out_tokens") or 0)
                tok_s = float(state.get("tok_s") or 0.0)
                cells.append(
                    f"{worker_label(worker)} r{turn} ttft={ttft_s} "
                    f"out={out_tokens} tok/s={tok_s:.0f} {elapsed:.0f}s"
                )
        lines = []
        width = 3
        for i in range(0, len(cells), width):
            lines.append("   " + " | ".join(cells[i:i + width]))
        return "\n".join(lines)

    def live_totals(self) -> dict:
        """在途流式汇总：未完成的轮也计入 out / tok/s / TTFT。"""
        now = time.perf_counter()
        with self._lock:
            states = [dict(item) for item in self._states.values()]
        waiting = streaming = idle = 0
        ttfts: list[float] = []
        out_tokens = 0
        tok_s_vals: list[float] = []
        for state in states:
            phase = state.get("phase") or "idle"
            if phase == "wait":
                waiting += 1
            elif phase == "stream":
                streaming += 1
            else:
                idle += 1
            out_tokens += int(state.get("out_tokens") or 0)
            ttft = state.get("ttft_ms")
            if ttft is not None and phase in {"wait", "stream"}:
                ttfts.append(float(ttft))
            tok_s = float(state.get("tok_s") or 0)
            if phase == "stream" and tok_s > 0:
                tok_s_vals.append(tok_s)
            started = state.get("started")
            if phase == "stream" and started and tok_s <= 0:
                elapsed = max(now - started, 0.05)
                out = int(state.get("out_tokens") or 0)
                if out:
                    tok_s_vals.append(out / elapsed)
        return {
            "waiting": waiting,
            "streaming": streaming,
            "idle": idle,
            "out_tokens": out_tokens,
            "ttft_avg": (sum(ttfts) / len(ttfts)) if ttfts else None,
            "tok_s": (sum(tok_s_vals) / len(tok_s_vals)) if tok_s_vals else 0.0,
        }

    def worker_state(self, worker: int) -> dict | None:
        worker = max(int(worker), 1)
        with self._lock:
            state = self._states.get(worker)
            return dict(state) if state is not None else None

    def status_lines(self) -> list[str]:
        now = time.perf_counter()
        with self._lock:
            items = sorted(self._states.items())
        lines = []
        for worker, state in items:
            phase = state.get("phase") or "idle"
            round_no = max(int(state.get("turn") or state.get("wave") or 0), 1)
            out = int(state.get("out_tokens") or 0)
            tok_s = float(state.get("tok_s") or 0.0)
            ttft = state.get("ttft_ms")
            started = state.get("started")
            elapsed = (now - started) if started else 0.0
            name = f"{worker_label(worker):<6}"
            if phase == "stream":
                first = f"  ttft={ttft / 1000:.1f}s" if ttft is not None else ""
                lines.append(
                    f"{name} 输出中  第{round_no}/{self.waves}次  "
                    f"out={out}  {tok_s:.0f} tok/s{first}  {elapsed:.0f}s"
                )
            elif phase == "wait":
                lines.append(
                    f"{name} 等首字  第{round_no}/{self.waves}次  已等 {elapsed:.0f}s"
                )
            elif phase == "error":
                err = (state.get("error") or "").strip()
                lines.append(f"{name} 失败    第{round_no}/{self.waves}次  {err[:50]}")
            elif int(state.get("done") or 0):
                lines.append(
                    f"{name} 完成    第{int(state.get('done'))}/{self.waves}次  out={out}"
                )
            else:
                lines.append(f"{name} 空闲")
        return lines

    def compact_cells(self) -> list[str]:
        return self.worker_lines()

    def warm_cache_label(self) -> str:
        with self._lock:
            rows = list(self.finished)
        return aggregate_cache_percent(rows, skip_first=True).strip()

    def worker_lines(self) -> list[str]:
        now = time.perf_counter()
        with self._lock:
            items = sorted(self._states.items())
        lines = []
        waves = self.waves
        for worker, state in items:
            phase = state.get("phase") or "idle"
            turn = max(int(state.get("turn") or 0), 1)
            wave = max(int(state.get("wave") or self.wave or 1), 1)
            done = int(state.get("done") or 0)
            out_tokens = int(state.get("out_tokens") or 0)
            tok_s = float(state.get("tok_s") or 0.0)
            ttft = state.get("ttft_ms")
            started = state.get("started")
            elapsed = (now - started) if started else 0.0
            err = (state.get("error") or "").strip()
            name = worker_label(worker)
            game = (state.get("game") or "").strip()
            cache_bit = _round_cache_label(state, turn)
            inp = int(state.get("input_tokens") or 0)
            lines.append(f"{name}  《{game}》" if game else name)
            phase_name = "预热" if wave <= 1 else "缓存"
            prefix = f"  └─ {phase_name} {wave}/{waves}"
            if phase == "stream":
                first = f" · 首字 {ttft / 1000:.1f}s" if ttft is not None else ""
                lines.append(
                    f"{prefix}  [输出中]  "
                    f"{out_tokens} tok · {tok_s:.1f}/s · {elapsed:.0f}s"
                    f"{first}  ·  {cache_bit}"
                )
            elif phase == "wait":
                lines.append(
                    f"{prefix}  [等首字]  已等 {elapsed:.0f}s  ·  {cache_bit}"
                )
            elif phase == "error":
                lines.append(f"{prefix}  [失败]  {err[:70]}")
            elif done:
                io_bit = f"输入 {inp} → 输出 {out_tokens}  ·  " if inp or out_tokens else ""
                lines.append(f"{prefix}  [结束]  {io_bit}{cache_bit}")
            else:
                lines.append(f"{prefix}  [待命]")
            for row in _output_tail(
                str(state.get("text") or ""), n=int(getattr(self, "tail_lines", OUTPUT_TAIL_LINES))
            ):
                lines.append(f"     {row}")
        return lines


def _output_tail(text: str, n: int = OUTPUT_TAIL_LINES) -> list[str]:
    n = int(n)
    if n <= 0:
        return []
    rows = [row[:OUTPUT_TAIL_WIDTH] for row in (text or "").replace("\r", "").split("\n") if row]
    if not rows:
        return ["(还没出字)"]
    return rows[-n:]


class LiveFooter:
    """整屏清空后重画：顶部保留启动横幅，下面每个 work 只出现一次。"""

    def __init__(self, *, stats, board: WorkerBoard, gate=None, header: list[str] | None = None):
        self.stats = stats
        self.board = board
        self.gate = gate
        self.header = list(header or [])
        self._lock = threading.Lock()
        self._n = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def start(self) -> None:
        if self._tty:
            sys.stdout.write("\033[?25l")
            sys.stdout.flush()
        self.refresh()
        self._thread = threading.Thread(
            target=self._run, name="llm-bench-live", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(0.4):
            self.refresh()

    def log(self, text: str) -> None:
        return

    def refresh(self) -> None:
        with self._lock:
            self._draw()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        with self._lock:
            if self._tty:
                sys.stdout.write("\033[?25h\n")
                sys.stdout.flush()

    def _draw(self) -> None:
        if not self._tty:
            return
        sys.stdout.write("\033[H\033[J")
        sys.stdout.write("\n".join(self._compose()) + "\n")
        sys.stdout.flush()
        self._n = 0

    def _compose(self) -> list[str]:
        snap = self.stats.snapshot()
        live = self.board.live_totals()
        limit = self.gate.limit if self.gate is not None else snap.workers
        ttft = live["ttft_avg"]
        if ttft is None:
            first = "都还没出第一个字"
        else:
            first = f"平均 {ttft / 1000:.1f} 秒才出第一个字"
        wave = max(int(self.board.wave), 1)
        total = max(int(self.board.waves), 1)
        if wave <= 1:
            stage = f"【预热 第{wave}/{total}次 · 冷启动不计命中率】"
        else:
            warm = self.board.warm_cache_label()
            stage = f"【缓存 第{wave}/{total}次 · 第2次起命中 {warm}】"
        live_line = (
            f"⏱ {snap.elapsed:.0f}s  {stage}  inflight={snap.in_flight}/{limit}  "
            f"ok={snap.ok} fail={snap.fail} 429={snap.rate_limited}  "
            f"wait={live['waiting']} stream={live['streaming']}  "
            f"out={live['out_tokens']}  {live['tok_s']:.1f} tok/s  {first}"
        )
        lines = []
        if self.header:
            lines.extend(self.header)
            lines.append("")
        lines.extend([live_line, "─" * 88, *self.board.worker_lines()])
        return lines


class RoundLiveDisplay:
    """单路时：等首 token / 流式中用同一行刷新，结束再写成完整轮数据。"""

    def __init__(self, index: int, *, worker: int = 1, turn: int = 1):
        self.index = int(index)
        self.worker = max(int(worker), 1)
        self.turn = max(int(turn), 1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._started = time.perf_counter()
        self._snap: dict = {}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._tick, name="llm-bench-live", daemon=True
        )
        self._thread.start()
        self.render()

    def on_progress(self, snap: dict) -> None:
        with self._lock:
            self._snap = dict(snap)
        self.render()

    def note(self, message: str) -> None:
        print(f"\n{message}", flush=True)

    def _tick(self) -> None:
        while not self._stop.wait(0.5):
            self.render()

    def render(self) -> None:
        elapsed = time.perf_counter() - self._started
        with self._lock:
            snap = dict(self._snap)
        ttft = snap.get("ttft_ms")
        if ttft is None:
            phase = "wait"
            ttft_s = "      ..."
        else:
            phase = "stream"
            ttft_s = f"{ttft:8.0f}ms"
        chunks = int(snap.get("chunks") or 0)
        out_tokens = int(snap.get("out_tokens") or 0)
        tok_s = float(snap.get("tok_s") or 0.0)
        line = (
            f"\r{worker_label(self.worker):<8}r{self.turn:<3}{phase:<7}"
            f" TTFT={ttft_s}  e2e={elapsed:6.1f}s  "
            f"out≈{out_tokens:<6} tok/s={tok_s:5.1f}  chunks={chunks:<5}"
        )
        print(line.ljust(100), end="", flush=True)

    def finish(self, final: str = "") -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        if final:
            print("\r" + final.ljust(110), flush=True)
        else:
            print("\r" + " " * 110, flush=True)


def print_model_average(rows: list[dict], *, show_cache: bool = True) -> None:
    separator()
    ttfts = [r["ttft_ms"] for r in rows]
    print(
        f"📊 均值  TTFT={format_ms(average(ttfts)).strip()}"
        f"  p95={format_ms(_p95(ttfts)).strip()}"
        f"  tok/s={format_tps(average_or_none([r.get('output_tps') for r in rows])).strip()}"
        f"  TPOT={format_tpot(average_or_none([r['tpot_ms'] for r in rows])).strip()}"
        f"  CDL_avg={format_ms(average([r['cdl_avg'] for r in rows])).strip()}"
        f"  CDL_p95={format_ms(average([r['cdl_p95'] for r in rows])).strip()}"
        f"  E2E={format_ms(average([r['e2e_ms'] for r in rows])).strip()}"
    )
    if show_cache:
        cold = [row for row in rows if is_first_request(row)]
        print(
            f"   cache 第1次(冷)={aggregate_cache_percent(cold).strip()}  "
            f"第2次起={aggregate_cache_percent(rows, skip_first=True).strip()}  "
            f"（第1次不计入命中率）"
        )


def print_token_totals(rows: list[dict], *, show_cache: bool = True) -> None:
    input_sum = sum(row["input_tokens"] for row in rows)
    output_sum = sum(row["output_tokens"] for row in rows)
    cached_sum = sum(row["cached_tokens"] for row in rows)
    line = f"   tokens  input={input_sum}  output={output_sum}"
    if show_cache:
        line += f"  cached={cached_sum}  uncached_input={input_sum - cached_sum}"
    print(line)


def print_summary(
    summary: dict,
    formats: list[str],
    models: list[str],
    *,
    show_cache: bool = True,
) -> None:
    if not summary:
        return
    print(f"\n{'═' * 100}")
    print("📊 汇总对比（均值）")
    extra = f"{'cached':>9}{'out':>7}{'cache%':>9}" if show_cache else f"{'out':>7}"
    print(
        f"{'格式':<12}{'模型':<26}{'TTFT':>10}{'tok/s':>9}{'TPOT':>13}{'CDL_avg':>11}"
        f"{'CDL_p95':>11}{'E2E':>10}{'input':>8}{extra}"
    )
    separator()
    for format_name in formats:
        for model in models:
            rows = summary.get((format_name, model))
            if not rows:
                continue
            cache_cols = (
                f"{int(average([r['cached_tokens'] for r in rows])):>9}"
                f"{int(average([r['output_tokens'] for r in rows])):>7}"
                f"{aggregate_cache_percent(rows, skip_first=True):>9}"
                if show_cache else
                f"{int(average([r['output_tokens'] for r in rows])):>7}"
            )
            print(
                f"{format_name:<12}{model:<26}"
                f"{format_ms(average([r['ttft_ms'] for r in rows]))}"
                f"{format_tps(average_or_none([r.get('output_tps') for r in rows]))}"
                f"{format_tpot(average_or_none([r['tpot_ms'] for r in rows]))}"
                f"{format_ms(average([r['cdl_avg'] for r in rows]))}"
                f"{format_ms(average([r['cdl_p95'] for r in rows]))}"
                f"{format_ms(average([r['e2e_ms'] for r in rows]))}"
                f"{int(average([r['input_tokens'] for r in rows])):>8}"
                f"{cache_cols}"
            )


def print_stress_summary(
    snapshot,
    *,
    limit: int | None = None,
    show_cache: bool = True,
    rows: list[dict] | None = None,
) -> None:
    separator()
    limit_text = f"  limit={limit}" if limit is not None else ""
    print(
        f"📊 压力  duration={snapshot.elapsed:.1f}s  inflight_peak={snapshot.peak_in_flight}/{snapshot.workers}"
        f"{limit_text}  ok={snapshot.ok}  fail={snapshot.fail}  429={snapshot.rate_limited}  5xx={snapshot.unavailable}"
    )
    print(
        f"   latency  TTFT={snapshot.ttft_avg:.1f}ms  tok/s={snapshot.output_tps_avg:.1f}"
    )
    print(
        f"   RPM   window={snapshot.rpm_window:.1f}  avg={snapshot.rpm:.1f}"
        f"  peak={snapshot.peak_rpm:.1f}"
    )
    print(
        f"   TPM   window={snapshot.tpm_window:.0f}  avg={snapshot.tpm:.0f}"
        f"  peak={snapshot.peak_tpm:.0f}"
    )
    tpm_line = (
        f"   TPM分项  input={snapshot.tpm_in:.0f}  output={snapshot.tpm_out:.0f}"
    )
    if show_cache:
        if rows:
            tpm_line += (
                f"  cache第2次起={aggregate_cache_percent(rows, skip_first=True).strip()}"
                f"  (第1次冷启不计入)"
            )
        else:
            tpm_line += f"  cache={snapshot.cache_percent:.1f}%"
    print(tpm_line)
    token_line = (
        f"   tokens  input={snapshot.input_tokens}  output={snapshot.output_tokens}"
    )
    if show_cache:
        token_line += (
            f"  cached={snapshot.cached_tokens}"
            f"  uncached_input={snapshot.input_tokens - snapshot.cached_tokens}"
        )
    print(token_line)


def _md(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def write_report(
    path,
    *,
    meta: dict,
    summary: dict,
    snapshots: dict | None = None,
    show_cache: bool = True,
) -> str:
    """写出 Markdown 测试报告，按 worker 汇总，便于对照。"""
    from pathlib import Path

    lines = [
        "# LLM Bench 报告",
        "",
        f"- 时间: {meta.get('started_at', '')}",
        f"- 协议: {', '.join(meta.get('formats') or [])}",
        f"- 模型: {', '.join(meta.get('models') or [])}",
        f"- cache_mode: {meta.get('cache_mode', '')}",
        f"- workers: {meta.get('workers', '')}",
        f"- rounds: {meta.get('rounds', '')}",
        f"- system: {meta.get('system', '')}",
        f"- 接入: {meta.get('base_url', '')}",
        "",
    ]
    for (format_name, model), rows in summary.items():
        if not rows:
            continue
        snap = (snapshots or {}).get((format_name, model))
        lines.append(f"## {format_name} / {model}")
        lines.append("")
        if snap is not None:
            lines.append(
                f"ok={snap.ok} fail={snap.fail} 429={snap.rate_limited} "
                f"5xx={snap.unavailable} duration={snap.elapsed:.1f}s"
            )
            lines.append("")
        ttfts = [r["ttft_ms"] for r in rows]
        if show_cache:
            lines.append("cache% 只计第 2 次起（第 1 次冷启动不计入）。")
            lines.append("")
        lines.append("| 范围 | TTFT avg | TTFT p95 | tok/s | cache% | input | output |")
        lines.append("|------|----------|----------|-------|--------|-------|--------|")
        lines.append(
            "| 全部 |"
            f" {_md(average(ttfts))} |"
            f" {_md(_p95(ttfts))} |"
            f" {_md(average_or_none([r.get('output_tps') for r in rows]))} |"
            f" {aggregate_cache_percent(rows, skip_first=True).strip() if show_cache else '-'} |"
            f" {sum(r['input_tokens'] for r in rows)} |"
            f" {sum(r['output_tokens'] for r in rows)} |"
        )
        by_worker: dict[int, list] = {}
        for row in rows:
            by_worker.setdefault(int(row.get("worker") or 1), []).append(row)
        for worker in sorted(by_worker):
            group = by_worker[worker]
            w_ttft = [r["ttft_ms"] for r in group]
            lines.append(
                f"| {worker_label(worker)} |"
                f" {_md(average(w_ttft))} |"
                f" {_md(_p95(w_ttft))} |"
                f" {_md(average_or_none([r.get('output_tps') for r in group]))} |"
                f" {aggregate_cache_percent(group, skip_first=True).strip() if show_cache else '-'} |"
                f" {sum(r['input_tokens'] for r in group)} |"
                f" {sum(r['output_tokens'] for r in group)} |"
            )
        lines.append("")
        for worker in sorted(by_worker):
            lines.append(f"### {worker_label(worker)}（第 {worker} 路并发对话）")
            lines.append("")
            for row in sorted(
                by_worker[worker],
                key=lambda item: int(item.get("wave") or item.get("turn") or 0),
            ):
                wave = int(row.get("wave") or row.get("turn") or 0)
                lines.append(f"#### 第 {wave} 次命令")
                lines.append(
                    f"- TTFT {_md(row.get('ttft_ms'))} ms · "
                    f"tok/s {_md(row.get('output_tps'))} · "
                    f"输入 {row.get('input_tokens', 0)} → 输出 {row.get('output_tokens', 0)} · "
                    f"cache {cache_percent(row).strip()}"
                )
                text = (row.get("text") or row.get("snippet") or "").strip()
                if text:
                    lines.append("")
                    lines.append("完整输出:")
                    lines.append("")
                    lines.append("```")
                    lines.append(text.replace("```", "'''"))
                    lines.append("```")
                lines.append("")
        lines.append("")
    target = Path(path)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(target)
