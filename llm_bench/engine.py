"""并发对话引擎：一个 worker 一路聊天，一个 round 一次「用户输入 → 模型输出」。"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .conversation import Conversation
from .reporting import LiveFooter, WorkerBoard
from .stress import AdaptiveGate, StressStats
from .transport import HttpStatusError, call_with_retries


def slim_result(result: dict, worker: int, turn: int) -> dict:
    slim = dict(result)
    text = slim.pop("text", None) or ""
    slim["snippet"] = " ".join(str(text).split())[:80]
    slim["worker"] = int(worker)
    slim["turn"] = int(turn)
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
) -> dict:
    affinity = conversation.session_id

    def once():
        return protocol.stream(
            base_url,
            api_key,
            model,
            "",
            "",
            max_tokens,
            timeout,
            messages=conversation.outbound(),
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
    return result


def play_round(
    *,
    conversation: Conversation,
    turn: int,
    protocol,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_delay: float,
    throttle: float,
    board: WorkerBoard,
    stats: StressStats,
    gate: AdaptiveGate,
    stop,
) -> dict | None:
    """跑完一轮问答。429/5xx 会重试同一轮；硬错误返回 None。"""
    worker = conversation.worker_id + 1
    board.begin_round(worker, turn)
    progress = board.on_progress(worker, turn)
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
                max_tokens=max_tokens,
                timeout=timeout,
                retries=retries,
                retry_delay=retry_delay,
                on_progress=progress,
            )
        except HttpStatusError as exc:
            if exc.rate_limited:
                stats.note_429()
                gate.release(rate_limited=True)
                board.fail_round(worker, turn, f"⚠️429 {exc}")
                time.sleep(exc.retry_after if exc.retry_after is not None else retry_delay)
                if not stop.is_set():
                    board.begin_round(worker, turn)
                continue
            if exc.capacity_limited:
                stats.note_unavailable()
                gate.release(rate_limited=False)
                board.fail_round(worker, turn, f"⚠️5xx {exc}")
                time.sleep(exc.retry_after if exc.retry_after is not None else retry_delay)
                if not stop.is_set():
                    board.begin_round(worker, turn)
                continue
            stats.fail()
            gate.release(rate_limited=False)
            board.fail_round(worker, turn, f"❌ {exc}")
            return None
        except Exception as exc:
            stats.fail()
            gate.release(rate_limited=False)
            board.fail_round(worker, turn, f"❌ {exc}")
            return None
        conversation.commit(result.get("text") or "")
        stats.succeed(result)
        gate.release(rate_limited=False)
        slim = slim_result(result, worker, turn)
        board.finish_round(worker, turn, slim)
        return slim
    return None


def play_worker(
    wid: int,
    *,
    turns: int,
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
    board: WorkerBoard,
    stats: StressStats,
    gate: AdaptiveGate,
    stop,
) -> list[dict]:
    conversation = Conversation(
        wid,
        system=system,
        user=user,
        cache=cache,
        session_prefix=session_prefix,
        followup=followup,
    )
    rows: list[dict] = []
    for turn in range(1, turns + 1):
        if stop.is_set():
            break
        slim = play_round(
            conversation=conversation,
            turn=turn,
            protocol=protocol,
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            throttle=throttle,
            board=board,
            stats=stats,
            gate=gate,
            stop=stop,
        )
        if slim is not None:
            rows.append(slim)
    return rows


def run_pool(
    *,
    workers: int,
    turns: int,
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
) -> tuple[list[dict], StressStats, AdaptiveGate]:
    """拉起 workers 路对话，每路 turns 轮问答。"""
    import threading

    workers = max(int(workers), 1)
    turns = max(int(turns), 1)
    stats = StressStats(workers=workers)
    gate = AdaptiveGate(workers)
    stop = threading.Event()
    board = WorkerBoard(workers, rounds=turns)
    footer = LiveFooter(stats=stats, board=board, gate=gate)
    footer.start()
    rows: list[dict] = []
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="llm-bench") as pool:
            futures = [
                pool.submit(
                    play_worker,
                    wid,
                    turns=turns,
                    system=system,
                    user=user,
                    followup=followup,
                    cache=cache,
                    session_prefix=session_prefix,
                    protocol=protocol,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    retries=retries,
                    retry_delay=retry_delay,
                    throttle=throttle,
                    board=board,
                    stats=stats,
                    gate=gate,
                    stop=stop,
                )
                for wid in range(workers)
            ]
            try:
                if duration > 0:
                    deadline = time.monotonic() + duration
                    while time.monotonic() < deadline and any(
                        not future.done() for future in futures
                    ):
                        time.sleep(0.2)
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
    finally:
        stop.set()
        footer.stop()
    rows.sort(key=lambda item: (int(item.get("worker") or 0), int(item.get("turn") or 0)))
    return rows, stats, gate
