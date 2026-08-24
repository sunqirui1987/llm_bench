"""并发引擎：每一波拉起全量 worker；每人发一条命令。hit 原样再发，miss 换新命令。"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import PROJECT_DIR
from .conversation import Conversation
from .prompts import CONTEXT_WINDOW, estimate_tokens, game_prefix, pick_game
from .reporting import LiveFooter, OutputLog, WorkerBoard
from .stress import AdaptiveGate, StressStats
from .transport import HttpStatusError, call_with_retries


def slim_result(result: dict, worker: int, wave: int = 1) -> dict:
    slim = dict(result)
    text = slim.get("text") or ""
    slim["text"] = text
    slim["snippet"] = " ".join(str(text).split())[:80]
    slim["worker"] = int(worker)
    slim["wave"] = int(wave)
    slim["turn"] = int(wave)
    return slim


def turn_request(
    *,
    protocol,
    base_url: str,
    api_key: str,
    model: str,
    conversation: Conversation,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_delay: float,
    on_progress=None,
    messages: list | None = None,
) -> dict:
    affinity = conversation.session_id
    out_tokens = conversation.output_tokens_for()
    if messages is None:
        messages = conversation.outbound()

    def once():
        return protocol.stream(
            base_url,
            api_key,
            model,
            "",
            "",
            out_tokens,
            timeout,
            messages=messages,
            session_id=affinity,
            on_progress=on_progress,
        )

    result = call_with_retries(
        once,
        (),
        retries,
        retry_delay,
        rotate_session_on_retry=False,
    )
    result["session_id"] = affinity
    result["planned_input"] = conversation.input_tokens_for()
    result["planned_output"] = out_tokens
    return result


def play_round(
    *,
    conversation: Conversation,
    wave: int,
    protocol,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    retries: int,
    retry_delay: float,
    throttle: float,
    board: WorkerBoard,
    stats: StressStats,
    gate: AdaptiveGate,
    stop,
    output_log: OutputLog | None = None,
) -> dict | None:
    worker = conversation.worker_id + 1
    board.begin_round(
        worker,
        wave,
        wave=wave,
        game=getattr(conversation, "game_title", ""),
    )
    if output_log is not None:
        output_log.start_step(
            worker,
            wave,
            planned_input=conversation.input_tokens_for(),
            planned_output=conversation.output_tokens_for(),
        )
    progress = board.on_progress(worker, wave)
    messages = conversation.outbound()

    while not stop.is_set():
        if not gate.acquire(stop):
            return None
        stats.begin()
        try:
            if throttle:
                time.sleep(throttle)
            result = turn_request(
                protocol=protocol,
                base_url=base_url,
                api_key=api_key,
                model=model,
                conversation=conversation,
                max_tokens=conversation.output_tokens_for(),
                timeout=timeout,
                retries=retries,
                retry_delay=retry_delay,
                on_progress=progress,
                messages=messages,
            )
        except HttpStatusError as exc:
            if exc.rate_limited:
                stats.note_429()
                gate.release(rate_limited=True)
                board.fail_round(worker, wave, f"⚠️429 {exc}", wave=wave)
                time.sleep(exc.retry_after if exc.retry_after is not None else retry_delay)
                if not stop.is_set():
                    board.begin_round(
                        worker,
                        wave,
                        wave=wave,
                        game=getattr(conversation, "game_title", ""),
                    )
                continue
            if exc.capacity_limited:
                stats.note_unavailable()
                gate.release(rate_limited=False)
                board.fail_round(worker, wave, f"⚠️5xx {exc}", wave=wave)
                time.sleep(exc.retry_after if exc.retry_after is not None else retry_delay)
                if not stop.is_set():
                    board.begin_round(
                        worker,
                        wave,
                        wave=wave,
                        game=getattr(conversation, "game_title", ""),
                    )
                continue
            stats.fail()
            gate.release(rate_limited=False)
            board.fail_round(worker, wave, f"❌ {exc}", wave=wave)
            return None
        except Exception as exc:
            stats.fail()
            gate.release(rate_limited=False)
            board.fail_round(worker, wave, f"❌ {exc}", wave=wave)
            return None
        stats.succeed(result)
        gate.release(rate_limited=False)
        slim = slim_result(result, worker, wave)
        board.finish_round(worker, wave, slim, wave=wave)
        if output_log is not None:
            output_log.finish_step(worker, wave, slim)
        return slim
    return None


def play_worker(
    wid: int,
    *,
    wave: int,
    system: str,
    user: str,
    followup: str,
    cache: bool,
    session_prefix: str,
    max_input: int,
    max_tokens: int,
    context_window: int,
    pad: bool,
    protocol,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    retries: int,
    retry_delay: float,
    throttle: float,
    board: WorkerBoard,
    stats: StressStats,
    gate: AdaptiveGate,
    stop,
    output_log: OutputLog | None = None,
    full_prefix: str = "",
    full_tokens: int = 0,
) -> list[dict]:
    conversation = Conversation(
        wid,
        system=system,
        user=user,
        cache=cache,
        session_prefix=session_prefix,
        followup=followup,
        max_input=max_input,
        max_tokens=max_tokens,
        context_window=context_window,
        pad=pad,
        full_prefix=full_prefix or None,
        full_tokens=full_tokens or None,
        seq=max(int(wave) - 1, 0),
    )
    slim = play_round(
        conversation=conversation,
        wave=wave,
        protocol=protocol,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
        throttle=throttle,
        board=board,
        stats=stats,
        gate=gate,
        stop=stop,
        output_log=output_log,
    )
    return [slim] if slim is not None else []


def run_pool(
    *,
    workers: int,
    rounds: int,
    duration: float,
    system: str,
    user: str,
    followup: str,
    cache: bool,
    session_prefix: str,
    protocol,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_delay: float,
    throttle: float,
    max_input: int = 0,
    context_window: int = CONTEXT_WINDOW,
    pad: bool = True,
    header: list[str] | None = None,
) -> tuple[list[dict], StressStats, AdaptiveGate]:
    """rounds 波全量线程。hit：每一波发同一条命令。miss：每一波换新命令。"""
    import threading

    workers = max(int(workers), 1)
    rounds = max(int(rounds), 1)
    stats = StressStats(workers=workers)
    gate = AdaptiveGate(workers)
    stop = threading.Event()
    board = WorkerBoard(workers, waves=rounds, show_cache=cache)
    worker_prefixes = [""] * workers
    worker_tokens = [0] * workers
    if pad and max_input > 0:
        for wid in range(workers):
            game = pick_game(wid)
            salt = (
                f"{session_prefix}|{game['title']}"
                if cache
                else f"miss-{wid}-{session_prefix}"
            )
            worker_prefixes[wid] = game_prefix(wid, system, max_input, salt)
            worker_tokens[wid] = estimate_tokens(worker_prefixes[wid])
    output_log = OutputLog(log_dir=PROJECT_DIR / "logs")
    footer = LiveFooter(
        stats=stats, board=board, gate=gate, header=header, show_cache=cache
    )
    footer.start()
    rows: list[dict] = []
    deadline = time.monotonic() + duration if duration > 0 else None
    try:
        for wave in range(1, rounds + 1):
            if stop.is_set():
                break
            if deadline is not None and time.monotonic() >= deadline:
                stop.set()
                break
            board.set_wave(wave)
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="llm-bench"
            ) as pool:
                futures = [
                    pool.submit(
                        play_worker,
                        wid,
                        wave=wave,
                        system=system,
                        user=user,
                        followup=followup,
                        cache=cache,
                        session_prefix=session_prefix,
                        max_input=max_input,
                        max_tokens=max_tokens,
                        context_window=context_window,
                        pad=pad,
                        protocol=protocol,
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        timeout=timeout,
                        retries=retries,
                        retry_delay=retry_delay,
                        throttle=throttle,
                        board=board,
                        stats=stats,
                        gate=gate,
                        stop=stop,
                        output_log=output_log,
                        full_prefix=worker_prefixes[wid],
                        full_tokens=worker_tokens[wid],
                    )
                    for wid in range(workers)
                ]
                try:
                    if deadline is not None:
                        while time.monotonic() < deadline and any(
                            not future.done() for future in futures
                        ):
                            time.sleep(0.2)
                        if time.monotonic() >= deadline:
                            stop.set()
                    for future in as_completed(futures):
                        rows.extend(future.result() or [])
                except KeyboardInterrupt:
                    stop.set()
                    print("\n⏹ 收到中断，等待在途请求结束后输出统计", flush=True)
                    for future in as_completed(futures):
                        try:
                            rows.extend(future.result() or [])
                        except Exception:
                            pass
                    break
    finally:
        stop.set()
        footer.stop()
    rows.sort(
        key=lambda item: (
            int(item.get("wave") or 0),
            int(item.get("worker") or 0),
        )
    )
    return rows, stats, gate
