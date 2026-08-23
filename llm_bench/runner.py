"""N 路连续对话压测：测 cache、TTFT、token/s。"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from . import session
from .config import (
    DEFAULT_BASE_URL,
    DEFAULT_CHAT_BASE_URL,
    DEFAULT_MESSAGES_BASE_URL,
    DEFAULT_MODELS,
    DEFAULT_PROMPT,
    DEFAULT_RESPONSES_BASE_URL,
    DEFAULT_WORKERS,
    PROJECT_DIR,
    TARGET_INPUT_TOKENS,
    ensure_api_key,
    parse_cache_mode,
    parse_list,
    resolve_base_urls,
)
from .conversation import Conversation
from .prompts import compose_system, compose_user, estimate_tokens
from .protocols.registry import PROTOCOLS
from .reporting import (
    aggregate_cache_percent,
    cache_percent,
    format_bench_row,
    format_ms,
    format_tps,
    print_bench_header,
    print_bench_row,
    print_model_average,
    RoundLiveDisplay,
    WorkerBoard,
    print_summary,
    print_stress_summary,
    print_token_totals,
    separator,
    worker_label,
    write_report,
)
from .stress import AdaptiveGate, StressReporter, StressStats
from .transport import (
    HttpStatusError,
    call_with_retries,
    configure_pool,
    raise_fd_limit,
)

_print_lock = threading.Lock()


def _prepare(
    models,
    formats,
    base_url: str,
    chat_base_url: str,
    responses_base_url: str,
    messages_base_url: str,
    api_key: str,
    session_id: str,
) -> tuple[list[str], list[str], dict[str, str], str, str]:
    model_list = parse_list(models)
    format_list = parse_list(formats)
    unknown = [name for name in format_list if name not in PROTOCOLS]
    if unknown:
        raise ValueError(
            f"未知格式: {', '.join(unknown)}；可选: {', '.join(PROTOCOLS)}"
        )
    base_urls = resolve_base_urls(
        base_url,
        chat_base_url,
        responses_base_url,
        messages_base_url,
    )
    return (
        model_list,
        format_list,
        base_urls,
        ensure_api_key(api_key),
        session.configure(session_id),
    )


def _turn_request(
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
    """一路对话的一回合。hit 带同一条 session；miss 不带头，且 outbound 会打散前缀。"""
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


def _build_prompts(
    *,
    system: str,
    system_prompt: str,
    system_file: str,
    prompt: str,
    prompt_file: str,
    followup: str,
    context_file: str,
    input_tokens: int,
) -> tuple[str, str, str]:
    return (
        compose_system(
            kind=system,
            text=system_prompt,
            file=system_file,
            context_file=context_file,
            input_tokens=input_tokens,
        ),
        compose_user(prompt, prompt_file),
        followup or "",
    )


def bench(
    models=DEFAULT_MODELS,
    formats="responses",
    base_url: str = DEFAULT_BASE_URL,
    chat_base_url: str = DEFAULT_CHAT_BASE_URL,
    responses_base_url: str = DEFAULT_RESPONSES_BASE_URL,
    messages_base_url: str = DEFAULT_MESSAGES_BASE_URL,
    api_key: str = "",
    max_tokens: int = 500000,
    prompt: str = DEFAULT_PROMPT,
    prompt_file: str = "",
    followup: str = "",
    rounds: int = 5000,
    throttle: float = 0.0,
    system: str = "long",
    system_prompt: str = "",
    system_file: str = "",
    input_tokens: int = TARGET_INPUT_TOKENS,
    context_file: str = "",
    session_id: str = "",
    retries: int = 2,
    retry_delay: float = 1.0,
    timeout: int = 1800,
    workers: int = DEFAULT_WORKERS,
    duration: float = 0.0,
    report_every: float = 5.0,
    verbose: bool = False,
    cache_mode: str = "hit",
):
    """开 workers 个线程，每路都是连续对话，测 cache / TTFT / token/s。

    每个线程是一路正常多轮聊天，成功后把回复和下一条用户消息追加进去。
      --cache_mode hit   要缓存：本路 session 一直带着，system 前缀不变。
      --cache_mode miss  不要缓存：不带 session，并且每次在 prompt 最前面换新盐。

    遇到 429 Too many pending requests 时会立刻收缩在途上限并重试。

    Args:
        models: 模型名列表（逗号分隔）
        formats: 格式列表（chat / responses / messages）
        base_url: 三种格式的公共默认接入点
        chat_base_url: chat 独立接入点，未传则回退 base_url
        responses_base_url: responses 独立接入点，未传则回退 base_url
        messages_base_url: messages 独立接入点，未传则回退 base_url
        api_key: Bearer token
        max_tokens: 最大输出 token，默认 500000
        prompt: 第一轮 user prompt；之后自动追加 followup
        prompt_file: 从文件读取第一轮 user prompt，可与 prompt 叠加
        followup: 第二轮起的 user 模板；空则用内置 Continue...
        rounds: 每路对话的轮数（50 workers × 5 rounds = 250 次），不是全局一共几枪
        throttle: 拿到在途名额后的短暂等待秒数
        system: short（不填充）或 long（填充到 input_tokens，便于测缓存）
        system_prompt: 自定义 system 文本，覆盖内置短提示
        system_file: 从文件读取 system，叠在 system_prompt 前
        input_tokens: system=long 时的目标输入 token
        context_file: 可选长上下文文件，叠在 system 最前面；不传则不加任何内置语料
        session_id: 仅用于日志展示；hit 模式下每路对话使用自己的稳定 ID
        retries: 流传输中断时的最大重试次数
        retry_delay: 重试初始等待秒数（指数退避）
        timeout: 单次流式请求超时秒数
        workers: 对话线程数 / 最大在途上限，默认 1，可用 --workers 或 LLM_WORKERS 改
        duration: 压测秒数；0 表示跑完 rounds
        report_every: 多路并发时的 RPM/TPM 心跳间隔；workers=1 时忽略，改为一轮一行
        verbose: 多路并发时是否打印每一轮；workers=1 默认就会打印每一轮
        cache_mode: hit（粘 session + 固定前缀）或 miss（不带 session + 每轮打散前缀）
    """
    models, formats, base_urls, api_key, initial_session = _prepare(
        models,
        formats,
        base_url,
        chat_base_url,
        responses_base_url,
        messages_base_url,
        api_key,
        session_id,
    )
    cache_mode = parse_cache_mode(cache_mode)
    hit_cache = cache_mode == "hit"
    system_text, prompt, followup_text = _build_prompts(
        system=system,
        system_prompt=system_prompt,
        system_file=system_file,
        prompt=prompt,
        prompt_file=prompt_file,
        followup=followup,
        context_file=context_file,
        input_tokens=input_tokens,
    )
    workers = max(int(workers), 1)
    duration = max(float(duration), 0.0)
    report_every = max(float(report_every), 0.5)
    request_tokens = estimate_tokens(system_text) + estimate_tokens(prompt)
    fd_limit = raise_fd_limit(workers * 8 + 256)
    configure_pool(workers)

    print(f"\n{'═' * 100}")
    if hit_cache:
        title = "LLM Bench · 连续对话（要缓存）"
        worker_desc = (
            f"{workers} 路 work1..work{workers}，每路 {rounds} 轮，"
            f"合计 {workers * max(int(rounds), 0)} 次；本路 session 一直不变"
        )
        session_desc = f"每路钉死一条 session_id（示例 {initial_session}）"
        cache_flow = (
            "带 session / prompt_cache_key；system 前缀固定。"
            "第 1 轮冷启，第 2 轮起应对上同一前缀"
        )
    else:
        title = "LLM Bench · 连续对话（不要缓存）"
        worker_desc = (
            f"{workers} 路 work1..work{workers}，每路 {rounds} 轮，"
            f"合计 {workers * max(int(rounds), 0)} 次；不带 session，每轮打散前缀"
        )
        session_desc = "不发送 session_id / prompt_cache_key"
        cache_flow = (
            "不带 session；每次在 system 最前面插入新的 CACHE_BYPASS 盐，"
            "从 token 0 打断 prefix cache（只去掉 session 不够）"
        )
    print(
        f"🚀 {title}  "
        f"rounds={rounds}  duration={duration or 'off'}s  workers={workers}  cache_mode={cache_mode}"
    )
    print(f"   workers  : {worker_desc}")
    if fd_limit:
        print(f"   fd limit : {fd_limit}")
    print(f"   chat base      : {base_urls['chat']}")
    print(f"   responses base : {base_urls['responses']}")
    print(f"   messages base  : {base_urls['messages']}")
    print(f"   models   : {', '.join(models)}")
    print(f"   formats  : {', '.join(formats)}")
    print(f"   session  : {session_desc}")
    print(f"   retries  : {retries} (initial delay: {retry_delay}s)")
    print(f"   timeout  : {timeout}s")
    per_round = True
    use_live = workers <= 1
    use_ticker = workers > 1 and report_every > 0
    if use_live:
        print("   report   : work1 当前轮实时刷新，结束打完整一行，并写 report.md")
    else:
        print(
            f"   report   : 每轮打印 workN 完整行；每 {report_every}s 列出全部 worker 状态；写 report.md"
        )
    print(f"   cache flow: {cache_flow}")
    prefix_tokens = estimate_tokens(system_text)
    print(
        f"   system   : {system}（{len(system_text)} 字符 / {prefix_tokens} token，"
        f"首轮约 {request_tokens} token）"
    )
    print(f"   prompt   : {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
    print(f"   max_tokens: {max_tokens}")
    if hit_cache and prefix_tokens < 2048:
        print(
            f"⚠️ cache_mode=hit 但 system 前缀只有 {prefix_tokens} token。"
            "多数 Prompt Cache 要 1024~4096 token 才开始命中，"
            "短前缀时 cache% 会一直很低。测缓存请用默认 --system long，"
            "或加大 --input_tokens。"
        )
    elif hit_cache:
        print("   说明    : 每路第 1 轮是冷启（cache≈0），从第 2 轮起同一前缀才会命中")
    print(f"{'═' * 100}")

    summary: dict = {}
    snapshots: dict = {}
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for format_name in formats:
        protocol = PROTOCOLS[format_name]
        protocol_base = base_urls[format_name]
        print(f"\n{'█' * 100}")
        print(f"📦 格式 [{format_name}] {protocol.name}")
        print(f"   endpoint : {protocol_base}{protocol.endpoint}")
        print(f"{'█' * 100}")

        for model in models:
            print(f"\n📌 model = {model}")
            if hit_cache:
                print("   cache: hit · 粘 session + 固定前缀（第 2 轮起应高 cache%、更低 TTFT）")
            else:
                print("   cache: miss · 无 session + 每轮打散前缀（cache% 应≈0，TTFT 应更高）")
            print(
                f"   输出: {workers} 路 work1..work{workers}，每路 {rounds} 轮；"
                "流式中刷实时 out token"
            )
            print_bench_header(show_cache=hit_cache)

            rows: list[dict] = []
            stats = StressStats(workers=workers)
            gate = AdaptiveGate(workers)
            stop = threading.Event()
            board = WorkerBoard(workers)
            error_shown = 0
            turns = max(int(rounds), 1)

            def store(result: dict) -> dict:
                slim = dict(result)
                slim.pop("text", None)
                return slim

            def emit_error(index: int, mark: str, exc: Exception, *, worker=1, turn=1) -> None:
                nonlocal error_shown
                with _print_lock:
                    if verbose or per_round or error_shown < 8:
                        print(
                            f"{worker_label(worker):<8}{turn:<4}{mark} {exc}",
                            flush=True,
                        )
                        error_shown += 1
                    elif error_shown == 8:
                        print("   … 后续同类错误不再逐条打印，看实时行的 429/5xx", flush=True)
                        error_shown += 1

            def worker(wid: int) -> None:
                conversation = Conversation(
                    wid,
                    system=system_text,
                    user=prompt,
                    cache=hit_cache,
                    session_prefix=initial_session or "llm-bench",
                    followup=followup_text,
                )
                worker_no = wid + 1
                for turn_no in range(1, turns + 1):
                    if stop.is_set():
                        return
                    live = (
                        RoundLiveDisplay(turn_no, worker=worker_no, turn=turn_no)
                        if use_live
                        else None
                    )
                    live_done = False
                    progress = (
                        live.on_progress
                        if live is not None
                        else board.on_progress(worker_no, turn_no)
                    )
                    board.update(
                        worker_no,
                        phase="wait",
                        turn=turn_no,
                        out_tokens=0,
                        tok_s=0.0,
                        started=time.perf_counter(),
                    )
                    if live is not None:
                        live.start()
                    try:
                        while not stop.is_set():
                            if not gate.acquire(stop):
                                return
                            stats.begin()
                            try:
                                if throttle:
                                    time.sleep(throttle)
                                result = _turn_request(
                                    protocol=protocol,
                                    base_url=protocol_base,
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
                                if exc.rate_limited or exc.capacity_limited:
                                    if exc.rate_limited:
                                        stats.note_429()
                                        gate.release(rate_limited=True)
                                        mark = "⚠️429"
                                    else:
                                        stats.note_unavailable()
                                        gate.release(rate_limited=False)
                                        mark = "⚠️5xx"
                                    if live is not None:
                                        live.note(
                                            f"{worker_label(worker_no):<8}{turn_no:<4}{mark} {exc}"
                                        )
                                    else:
                                        emit_error(
                                            turn_no, mark, exc, worker=worker_no, turn=turn_no
                                        )
                                    time.sleep(
                                        exc.retry_after if exc.retry_after is not None else retry_delay
                                    )
                                    continue
                                stats.fail()
                                gate.release(rate_limited=False)
                                emit_error(
                                    turn_no, "❌", exc, worker=worker_no, turn=turn_no
                                )
                                break
                            except Exception as exc:
                                stats.fail()
                                gate.release(rate_limited=False)
                                emit_error(
                                    turn_no, "❌", exc, worker=worker_no, turn=turn_no
                                )
                                break
                            else:
                                conversation.commit(result.get("text") or "")
                                stats.succeed(result)
                                gate.release(rate_limited=False)
                                slim = store(result)
                                slim["worker"] = worker_no
                                slim["turn"] = turn_no
                                row = format_bench_row(turn_no, slim, show_cache=hit_cache)
                                board.update(
                                    worker_no,
                                    phase="idle",
                                    turn=turn_no,
                                    out_tokens=int(slim.get("output_tokens") or 0),
                                    tok_s=float(slim.get("output_tps") or 0),
                                    started=None,
                                )
                                with _print_lock:
                                    rows.append(slim)
                                    if live is not None:
                                        live.finish(row)
                                        live_done = True
                                    else:
                                        print_bench_row(turn_no, slim, show_cache=hit_cache)
                                break
                    finally:
                        if live is not None and not live_done:
                            live.finish()

            reporter = None
            if use_ticker:
                reporter = StressReporter(
                    stats,
                    min(float(report_every), 1.0),
                    limit_provider=lambda: gate.limit,
                    show_cache=hit_cache,
                )
                reporter.board = board
                reporter.start()
            try:
                with ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="llm-bench",
                ) as pool:
                    futures = [pool.submit(worker, i) for i in range(workers)]
                    try:
                        if duration > 0:
                            deadline = time.monotonic() + duration
                            while time.monotonic() < deadline and any(
                                not future.done() for future in futures
                            ):
                                time.sleep(0.2)
                            stop.set()
                        for future in as_completed(futures):
                            future.result()
                    except KeyboardInterrupt:
                        stop.set()
                        print("\n⏹ 收到中断，等待在途请求结束后输出统计", flush=True)
            finally:
                stop.set()
                if reporter is not None:
                    reporter.stop()
                    reporter.emit()

            snap = stats.snapshot()
            snapshots[(format_name, model)] = snap
            print_stress_summary(snap, limit=gate.limit, show_cache=hit_cache)
            if not hit_cache and snap.ok and snap.cache_percent > 5:
                print(
                    f"⚠️ cache_mode=miss 已打散前缀但仍有 {snap.cache_percent:.1f}% 缓存命中，"
                    "上游可能按语义或其它键缓存"
                )
            if not rows:
                print("⚠️ 本模型没有成功样本，不生成延迟统计")
                continue
            print(f"   有效样本: {len(rows)}/{rounds}")
            summary[(format_name, model)] = rows
            print_model_average(rows, show_cache=hit_cache)
            print_token_totals(rows, show_cache=hit_cache)

    print_summary(summary, formats, models, show_cache=hit_cache)
    report_path = write_report(
        PROJECT_DIR / "report.md",
        meta={
            "started_at": started_at,
            "formats": formats,
            "models": models,
            "cache_mode": cache_mode,
            "workers": workers,
            "system": system,
            "base_url": base_urls.get(formats[0], "") if formats else "",
        },
        summary=summary,
        snapshots=snapshots,
        show_cache=hit_cache,
    )
    print(f"📄 报告已写入 {report_path}")
    print()


def cache(
    models=DEFAULT_MODELS,
    formats="responses",
    base_url: str = DEFAULT_BASE_URL,
    chat_base_url: str = DEFAULT_CHAT_BASE_URL,
    responses_base_url: str = DEFAULT_RESPONSES_BASE_URL,
    messages_base_url: str = DEFAULT_MESSAGES_BASE_URL,
    api_key: str = "",
    max_tokens: int = 500000,
    prompt: str = DEFAULT_PROMPT,
    prompt_file: str = "",
    followup: str = "",
    rounds: int = 5,
    throttle: float = 0.3,
    system: str = "long",
    system_prompt: str = "",
    system_file: str = "",
    input_tokens: int = TARGET_INPUT_TOKENS,
    context_file: str = "",
    session_id: str = "",
    retries: int = 2,
    retry_delay: float = 1.0,
):
    """缓存命中率诊断：同一路对话、同一 session，观察冷启和稳态。

    Args:
        参数含义与 bench 相同。默认 --system long，便于看出 R1 冷启 → R2 命中。
    """
    models, formats, base_urls, api_key, active_session = _prepare(
        models,
        formats,
        base_url,
        chat_base_url,
        responses_base_url,
        messages_base_url,
        api_key,
        session_id,
    )
    system_text, prompt, followup_text = _build_prompts(
        system=system,
        system_prompt=system_prompt,
        system_file=system_file,
        prompt=prompt,
        prompt_file=prompt_file,
        followup=followup,
        context_file=context_file,
        input_tokens=input_tokens,
    )
    print(f"\n{'═' * 100}")
    print(f"🚀 LLM Bench · 缓存冷启诊断  rounds={rounds}")
    print(f"   chat base      : {base_urls['chat']}")
    print(f"   responses base : {base_urls['responses']}")
    print(f"   messages base  : {base_urls['messages']}")
    print(f"   session  : {active_session} (header: session_id)")
    print(f"   retries  : {retries} (initial delay: {retry_delay}s)")
    print(
        f"   system   : {system} (~{len(system_text)} 字符 / "
        f"{estimate_tokens(system_text)} token)"
    )
    print(f"   方式     : 同一路对话重复 {rounds} 轮，R1 冷启动，R2+ 应命中")
    print(f"{'═' * 100}")

    for format_name in formats:
        protocol = PROTOCOLS[format_name]
        protocol_base = base_urls[format_name]
        print(f"\n{'█' * 100}")
        print(
            f"📦 格式 [{format_name}] {protocol.name}  —  "
            f"{protocol_base}{protocol.endpoint}"
        )
        print(f"{'█' * 100}")
        for model in models:
            model_session = session.configure(
                session.scoped(active_session, format_name, model)
            )
            conversation = Conversation(
                0,
                system=system_text,
                user=prompt,
                cache=True,
                session_prefix=model_session,
                followup=followup_text,
            )
            conversation.session_id = model_session
            print(f"\n📌 model = {model}")
            print(f"   cache key/session: {model_session}")
            print(
                f"{'轮':<3}{'TTFT':>10}{'tok/s':>9}{'E2E':>10}{'input':>8}"
                f"{'cached':>9}{'命中率':>9}{'out':>7}"
            )
            separator()
            rows = []
            for index in range(1, int(rounds) + 1):
                live = RoundLiveDisplay(index, worker=1, turn=index)
                live.start()
                try:
                    result = _turn_request(
                        protocol=protocol,
                        base_url=protocol_base,
                        api_key=api_key,
                        model=model,
                        conversation=conversation,
                        max_tokens=max_tokens,
                        timeout=180,
                        retries=retries,
                        retry_delay=retry_delay,
                        on_progress=live.on_progress,
                    )
                    conversation.commit(result.get("text") or "")
                    retry_mark = (
                        f"  [retry={result['retry_count']}, session={result['session_id']}]"
                        if result["retry_count"] else ""
                    )
                    live.finish(
                        f"{index:<3}{format_ms(result['ttft_ms'])}"
                        f"{format_tps(result.get('output_tps'))}"
                        f"{format_ms(result['e2e_ms'])}"
                        f"{result['input_tokens']:>8}{result['cached_tokens']:>9}"
                        f"{cache_percent(result):>9}{result['output_tokens']:>7}"
                        f"{retry_mark}"
                    )
                    rows.append(result)
                except Exception as exc:
                    live.finish()
                    print(f"{index:<3}❌ {exc}", flush=True)
                time.sleep(throttle)
            if not rows:
                print("❌ 该模型该格式全部请求失败")
                continue
            separator()
            steady = rows[1:]
            print(
                f"📊 首轮(冷启)={cache_percent(rows[0])}  "
                f"稳态(≥2轮)={aggregate_cache_percent(steady) if steady else '   n/a  '}  "
                f"整体={aggregate_cache_percent(rows)}"
            )
    print()
