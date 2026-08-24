"""终端表格格式化与跨轮次聚合计算。"""

from __future__ import annotations

import shutil
import sys
import threading
import time
import unicodedata
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


def first_success_by_worker(rows: list[dict]) -> dict[int, int]:
    """每个 worker 第一次成功对应的 wave。失败的波次不在 rows 里。"""
    first: dict[int, int] = {}
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row.get("worker") or 1),
            int(row.get("wave") or row.get("turn") or 1),
        ),
    )
    for row in ordered:
        worker = int(row.get("worker") or 1)
        if worker not in first:
            first[worker] = int(row.get("wave") or row.get("turn") or 1)
    return first


def is_first_request(row: dict, first_by_worker: dict[int, int] | None = None) -> bool:
    worker = int(row.get("worker") or 1)
    wave = int(row.get("wave") or row.get("turn") or 1)
    if first_by_worker is None:
        return wave <= 1
    return wave == first_by_worker.get(worker, wave)


def warm_rows(rows: list[dict]) -> list[dict]:
    """去掉每个 worker 的第一次成功（冷启动），只留同样命令再来的样本。"""
    first = first_success_by_worker(rows)
    return [row for row in rows if not is_first_request(row, first)]


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


def terminal_size() -> tuple[int, int]:
    """当前终端列数、行数。给不出时用 80×40。"""
    try:
        size = shutil.get_terminal_size(fallback=(80, 40))
        cols, rows = int(size.columns), int(size.lines)
    except Exception:
        cols, rows = 80, 40
    return max(cols, 20), max(rows, 8)


def display_width(text: str) -> int:
    width = 0
    for char in text or "":
        if char in "\n\r":
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def fit_line(text: str, width: int) -> str:
    """按显示宽度截断，避免中文把一行折成两行顶出屏幕。"""
    text = (text or "").replace("\r", "").replace("\n", " ")
    width = max(int(width), 1)
    if display_width(text) <= width:
        return text
    if width <= 1:
        return "…"[:width]
    limit = width - 1
    out: list[str] = []
    used = 0
    for char in text:
        extra = 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
        if used + extra > limit:
            break
        out.append(char)
        used += extra
    return "".join(out) + "…"


class OutputLog:
    """不往控制台打字。结束时把全文写入 logs/，面板自己刷。"""

    def __init__(self, log_dir=None):
        self.log_dir = log_dir

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


def _round_cache_label(
    state: dict,
    turn: int | None = None,
    *,
    show_cache: bool = True,
    warmup: bool = False,
) -> str:
    inp = int(state.get("input_tokens") or 0)
    cached = int(state.get("cached_tokens") or 0)
    pct = state.get("cache_percent")
    cache_turn = int(state.get("cache_turn") or 0)
    if show_cache and warmup:
        return "预热·不计命中"
    if inp > 0:
        pct = 100.0 * cached / inp
        text = f"cache={pct:.1f}% ({cached}/{inp})"
    elif pct is None:
        return "新命令" if not show_cache else "等命中结果"
    else:
        text = f"cache={float(pct):.1f}%"
    if not show_cache:
        if inp > 0 and cached > 0:
            return f"{text} 不应命中"
        return "新命令"
    if cache_turn and turn and cache_turn != turn:
        return f"上轮 {text}"
    return text


class WorkerBoard:
    """多路并发：每一波每人发一条命令。"""

    def __init__(
        self,
        workers: int,
        rounds: int | None = None,
        waves: int = 1,
        *,
        show_cache: bool = True,
    ):
        self.workers = max(int(workers), 1)
        total = int(rounds if rounds is not None else waves)
        self.waves = max(total, 1)
        self.wave = 1
        self.show_cache = bool(show_cache)
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

    def progress_waves(self) -> tuple[int, int]:
        with self._lock:
            waves = [
                max(int(state.get("wave") or 1), 1) for state in self._states.values()
            ]
        if not waves:
            return 1, 1
        return min(waves), max(waves)

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
                    "cache_reported": bool(result.get("cache_reported")),
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

    def warm_cache_label(self) -> str:
        with self._lock:
            rows = list(self.finished)
        return aggregate_cache_percent(rows, skip_first=True).strip()

    def _snapshots(self) -> list[dict]:
        now = time.perf_counter()
        with self._lock:
            items = sorted(self._states.items())
            first_by_worker = first_success_by_worker(self.finished)
        show_cache = bool(getattr(self, "show_cache", True))
        waves = self.waves
        snaps = []
        for worker, state in items:
            phase = state.get("phase") or "idle"
            turn = max(int(state.get("turn") or 0), 1)
            wave = max(int(state.get("wave") or self.wave or 1), 1)
            first_wave = first_by_worker.get(worker)
            warmup = show_cache and (first_wave is None or first_wave == wave)
            if not show_cache:
                phase_name = "换新"
            elif warmup:
                phase_name = "预热"
            else:
                phase_name = "缓存"
            started = state.get("started")
            snaps.append(
                {
                    "worker": worker,
                    "name": worker_label(worker),
                    "game": (state.get("game") or "").strip(),
                    "phase": phase,
                    "turn": turn,
                    "wave": wave,
                    "waves": waves,
                    "done": int(state.get("done") or 0),
                    "out": int(state.get("out_tokens") or 0),
                    "tok_s": float(state.get("tok_s") or 0.0),
                    "ttft": state.get("ttft_ms"),
                    "elapsed": (now - started) if started else 0.0,
                    "err": (state.get("error") or "").strip(),
                    "inp": int(state.get("input_tokens") or 0),
                    "text": str(state.get("text") or ""),
                    "phase_name": phase_name,
                    "cache_bit": _round_cache_label(
                        state, turn, show_cache=show_cache, warmup=warmup
                    ),
                }
            )
        return snaps

    def worker_lines(self, *, budget: int | None = None, width: int = 80) -> list[str]:
        snaps = self._snapshots()
        n = max(len(snaps), 1)
        if budget is None:
            tail = int(getattr(self, "tail_lines", OUTPUT_TAIL_LINES))
            budget = n * (2 + max(tail, 0))
        budget = max(int(budget), 1)
        width = max(int(width), 20)
        per = budget // n
        if per >= 3:
            lines = _lines_detailed(snaps, min(OUTPUT_TAIL_LINES, per - 2), width)
        elif per >= 2:
            lines = _lines_two(snaps, width)
        elif budget >= n:
            lines = _lines_one(snaps, width)
        else:
            lines = _lines_grid(snaps, budget, width)
        return [fit_line(line, width) for line in lines]


def _output_tail(text: str, n: int = OUTPUT_TAIL_LINES, width: int = OUTPUT_TAIL_WIDTH) -> list[str]:
    n = int(n)
    if n <= 0:
        return []
    width = max(int(width), 8)
    rows = [
        fit_line(row, width)
        for row in (text or "").replace("\r", "").split("\n")
        if row
    ]
    if not rows:
        return [fit_line("(还没出字)", width)]
    return rows[-n:]


def _status_long(snap: dict) -> str:
    phase = snap["phase"]
    prefix = f"  └─ {snap['phase_name']} {snap['wave']}/{snap['waves']}"
    if phase == "stream":
        first = f" · 首字 {snap['ttft'] / 1000:.1f}s" if snap["ttft"] is not None else ""
        return (
            f"{prefix}  [输出中]  "
            f"{snap['out']} tok · {snap['tok_s']:.1f}/s · {snap['elapsed']:.0f}s"
            f"{first}  ·  {snap['cache_bit']}"
        )
    if phase == "wait":
        return f"{prefix}  [等首字]  已等 {snap['elapsed']:.0f}s  ·  {snap['cache_bit']}"
    if phase == "error":
        return f"{prefix}  [失败]  {snap['err'][:70]}"
    if snap["done"]:
        io_bit = (
            f"输入 {snap['inp']} → 输出 {snap['out']}  ·  "
            if snap["inp"] or snap["out"]
            else ""
        )
        return f"{prefix}  [结束]  {io_bit}{snap['cache_bit']}"
    return f"{prefix}  [待命]"


def _status_one(snap: dict) -> str:
    game = f" 《{snap['game']}》" if snap["game"] else ""
    phase = snap["phase"]
    if phase == "stream":
        body = (
            f"输出中 {snap['wave']}/{snap['waves']} "
            f"out={snap['out']} {snap['tok_s']:.0f}/s {snap['elapsed']:.0f}s "
            f"{snap['cache_bit']}"
        )
    elif phase == "wait":
        body = (
            f"等首字 {snap['wave']}/{snap['waves']} "
            f"{snap['elapsed']:.0f}s {snap['cache_bit']}"
        )
    elif phase == "error":
        body = f"失败 {snap['err'][:40]}"
    elif snap["done"]:
        body = (
            f"结束 {snap['wave']}/{snap['waves']} "
            f"out={snap['out']} {snap['cache_bit']}"
        )
    else:
        body = "待命"
    return f"{snap['name']}{game}  {body}"


def _status_cell(snap: dict) -> str:
    phase = snap["phase"]
    if phase == "stream":
        bit = f"输出{snap['out']}"
    elif phase == "wait":
        bit = f"等{snap['elapsed']:.0f}s"
    elif phase == "error":
        bit = "失败"
    elif snap["done"]:
        bit = "完成"
    else:
        bit = "待命"
    return f"{snap['name']} {bit}"


def _lines_detailed(snaps: list[dict], tail: int, width: int) -> list[str]:
    tail_width = max(width - 5, 8)
    lines = []
    for snap in snaps:
        title = f"{snap['name']}  《{snap['game']}》" if snap["game"] else snap["name"]
        lines.append(fit_line(title, width))
        lines.append(fit_line(_status_long(snap), width))
        for row in _output_tail(snap["text"], n=tail, width=tail_width):
            lines.append(fit_line(f"     {row}", width))
    return lines


def _lines_two(snaps: list[dict], width: int) -> list[str]:
    tail_width = max(width - 5, 8)
    lines = []
    for snap in snaps:
        lines.append(fit_line(_status_one(snap), width))
        tail = _output_tail(snap["text"], n=1, width=tail_width)
        lines.append(fit_line(f"     {tail[0]}" if tail else "     ", width))
    return lines


def _lines_one(snaps: list[dict], width: int) -> list[str]:
    return [fit_line(_status_one(snap), width) for snap in snaps]


def _lines_grid(snaps: list[dict], budget: int, width: int) -> list[str]:
    n = len(snaps)
    budget = max(int(budget), 1)
    cell_min = 16
    sep = 3
    ncols = max(1, min(n, (width + sep) // (cell_min + sep)))
    capacity = ncols * budget
    extra = n > capacity
    show = snaps[: capacity - 1] if extra else snaps
    cell_w = max(1, (width - (ncols - 1) * sep) // ncols)
    cells = [fit_line(_status_cell(snap), cell_w) for snap in show]
    if extra:
        cells.append(fit_line(f"+{n - len(show)}路", cell_w))
    rows = []
    for i in range(0, len(cells), ncols):
        rows.append(fit_line(" | ".join(cells[i:i + ncols]), width))
    return rows[:budget]


def _shrink_header(header: list[str]) -> list[str]:
    if not header:
        return []
    title = next((line for line in header if "LLM Bench" in line), header[0])
    play = next((line for line in header if line.startswith("▶")), "")
    out = [title]
    if play and play != title:
        out.append(play)
    return out


class LiveFooter:
    """整屏清空后重画：顶部保留启动横幅，下面每个 work 只出现一次。"""

    def __init__(
        self,
        *,
        stats,
        board: WorkerBoard,
        gate=None,
        header: list[str] | None = None,
        show_cache: bool | None = None,
    ):
        self.stats = stats
        self.board = board
        self.gate = gate
        self.header = list(header or [])
        self.show_cache = (
            bool(board.show_cache) if show_cache is None else bool(show_cache)
        )
        self._lock = threading.Lock()
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
        cols, rows = terminal_size()
        lines = [
            fit_line(line, cols) for line in self._compose(cols=cols, rows=rows)[:rows]
        ]
        sys.stdout.write("\033[H\033[J")
        if lines:
            # 最后一行不要再加换行，否则刚好写满屏幕时会把整页顶上去。
            sys.stdout.write("\n".join(lines))
        sys.stdout.flush()

    def _compose(self, cols: int | None = None, rows: int | None = None) -> list[str]:
        tcols, trows = terminal_size()
        cols = max(int(cols or tcols), 20)
        rows = max(int(rows or trows), 8)
        snap = self.stats.snapshot()
        live = self.board.live_totals()
        limit = self.gate.limit if self.gate is not None else snap.workers
        lo, hi = self.board.progress_waves()
        total = max(int(self.board.waves), 1)
        span = f"第{lo}/{total}次" if lo == hi else f"第{lo}–{hi}/{total}次"
        if not self.show_cache:
            stage = f"【各路独立 {span} · 每次换新命令】"
        elif hi <= 1:
            stage = f"【各路独立 {span} · 预热不计命中率】"
        else:
            warm = self.board.warm_cache_label()
            stage = f"【各路独立 {span} · 第2次起命中 {warm}】"
        rate_s = max(float(snap.elapsed), 1.0)
        live_out = int(live["out_tokens"] or 0)
        tpm_in = float(snap.tpm_in)
        tpm_out = (int(snap.output_tokens or 0) + live_out) / rate_s * 60.0
        tpm_now = tpm_in + tpm_out
        rate_line = (
            f"⏱ {snap.elapsed:.0f}s  "
            f"RPM={snap.rpm_window:.1f}  TPM={tpm_now:.0f} "
            f"(in={tpm_in:.0f} out={tpm_out:.0f})  "
            f"inflight={snap.in_flight}/{limit}  "
            f"ok={snap.ok} fail={snap.fail} 429={snap.rate_limited}"
        )
        live_line = (
            f"{stage}  "
            f"wait={live['waiting']} stream={live['streaming']}  "
            f"out={live_out}  {live['tok_s']:.1f} tok/s"
        )
        workers = max(int(self.board.workers), 1)
        full_header = list(self.header)
        short_header = _shrink_header(full_header)
        status_lines = 2

        def overhead(header: list[str]) -> int:
            return len(header) + (1 if header else 0) + status_lines + 1

        header = full_header
        if overhead(full_header) + workers > rows:
            header = short_header
        remain = max(rows - overhead(header), 1)
        sep = "─" * min(88, cols)
        lines: list[str] = []
        if header:
            lines.extend(fit_line(line, cols) for line in header)
            lines.append("")
        lines.extend(
            [
                fit_line(rate_line, cols),
                fit_line(live_line, cols),
                fit_line(sep, cols),
                *self.board.worker_lines(budget=remain, width=cols),
            ]
        )
        return lines[:rows]


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
        first = first_success_by_worker(rows)
        cold = [row for row in rows if is_first_request(row, first)]
        print(
            f"   cache 第1次(冷)={aggregate_cache_percent(cold).strip()}  "
            f"第2次起={aggregate_cache_percent(rows, skip_first=True).strip()}  "
            f"（每路第一次成功不计入命中率）"
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
                f"  (每路第一次成功不计入)"
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
        f"- effort: {meta.get('effort', '')}",
        f"- via: {meta.get('via', 'http')}",
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
            lines.append("cache% 只计每路第一次成功之后（冷启动不计入）。")
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
