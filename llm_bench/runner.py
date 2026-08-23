"""命令行：bench / cache。真正的对话循环在 engine。"""

from __future__ import annotations

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
from .engine import run_pool, turn_request as _turn_request
from .prompts import compose_system, compose_user, estimate_tokens
from .protocols.registry import PROTOCOLS
from .reporting import (
    aggregate_cache_percent,
    cache_percent,
    print_model_average,
    print_summary,
    print_stress_summary,
    print_token_totals,
    write_report,
)
from .transport import configure_pool, raise_fd_limit

# 测试仍从 runner 导入 _turn_request
__all__ = ["bench", "cache", "_turn_request"]


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


def _print_start(
    *,
    hit_cache: bool,
    workers: int,
    rounds,
    duration: float,
    cache_mode: str,
    fd_limit: int,
    base_urls: dict,
    models: list[str],
    formats: list[str],
    initial_session: str,
    retries: int,
    retry_delay: float,
    timeout: int,
    system: str,
    system_text: str,
    prompt: str,
    max_tokens: int,
) -> None:
    total = workers * max(int(rounds), 0)
    if hit_cache:
        title = "LLM Bench · 要缓存"
        session_desc = f"每路钉死一条 session（示例 {initial_session}）"
        cache_flow = "粘 session + 固定前缀；第 1 轮冷启，第 2 轮起应对上 cache"
    else:
        title = "LLM Bench · 不要缓存"
        session_desc = "不带 session；每轮打散前缀"
        cache_flow = "不带 session，并在 prompt 最前面换新盐"
    print(f"\n{'═' * 88}")
    print(
        f"{title}  workers={workers}（并发对话）  "
        f"rounds={rounds}（每路问答轮数）  合计最多 {total} 次  "
        f"duration={duration or 'off'}s  cache_mode={cache_mode}"
    )
    if fd_limit:
        print(f"   fd limit : {fd_limit}")
    print(f"   chat      : {base_urls['chat']}")
    print(f"   responses : {base_urls['responses']}")
    print(f"   messages  : {base_urls['messages']}")
    print(f"   models    : {', '.join(models)}")
    print(f"   formats   : {', '.join(formats)}")
    print(f"   session   : {session_desc}")
    print(f"   cache     : {cache_flow}")
    print(f"   retries   : {retries}  timeout={timeout}s  delay={retry_delay}s")
    prefix_tokens = estimate_tokens(system_text)
    print(
        f"   system    : {system}  {len(system_text)} 字 / {prefix_tokens} token  "
        f"首轮约 {prefix_tokens + estimate_tokens(prompt)} token"
    )
    print(f"   prompt    : {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
    print(f"   max_tokens: {max_tokens}")
    print("   画面      : 像 top 清屏。work=一路对话，--round=这一路当前第几次问答")
    if hit_cache and prefix_tokens < 2048:
        print(
            f"⚠️ cache_mode=hit 但 system 只有 {prefix_tokens} token，"
            "多数服务要 1024+ 才缓存。请用 --system long。"
        )
    print(f"{'═' * 88}")


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
    """workers 路并发对话；每路 rounds 轮「用户输入 → 模型输出」。

      --cache_mode hit   本路 session 一直带着，前缀固定。
      --cache_mode miss  不带 session，每轮打散前缀。
    """
    del report_every, verbose
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
    fd_limit = raise_fd_limit(workers * 8 + 256)
    configure_pool(workers)
    _print_start(
        hit_cache=hit_cache,
        workers=workers,
        rounds=rounds,
        duration=duration,
        cache_mode=cache_mode,
        fd_limit=fd_limit,
        base_urls=base_urls,
        models=models,
        formats=formats,
        initial_session=initial_session,
        retries=retries,
        retry_delay=retry_delay,
        timeout=timeout,
        system=system,
        system_text=system_text,
        prompt=prompt,
        max_tokens=max_tokens,
    )

    summary: dict = {}
    snapshots: dict = {}
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for format_name in formats:
        protocol = PROTOCOLS[format_name]
        protocol_base = base_urls[format_name]
        for model in models:
            print(f"\n▶ {format_name}  {protocol_base}{protocol.endpoint}  model={model}")
            rows, stats, gate = run_pool(
                workers=workers,
                turns=int(rounds),
                duration=duration,
                system=system_text,
                user=prompt,
                followup=followup_text,
                cache=hit_cache,
                session_prefix=initial_session or "llm-bench",
                protocol=protocol,
                base_url=protocol_base,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                timeout=timeout,
                retries=retries,
                retry_delay=retry_delay,
                throttle=throttle,
            )
            snap = stats.snapshot()
            snapshots[(format_name, model)] = snap
            print_stress_summary(snap, limit=gate.limit, show_cache=hit_cache)
            if not hit_cache and snap.ok and snap.cache_percent > 5:
                print(
                    f"⚠️ cache_mode=miss 已打散前缀但仍有 {snap.cache_percent:.1f}% 命中"
                )
            if not rows:
                print("⚠️ 没有成功样本")
                continue
            print(f"   有效问答: {len(rows)}（{workers} 路 × 每路最多 {rounds} 轮）")
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
    """单路缓存诊断：同一对话连问 rounds 轮，看 R1 冷启 → R2 命中。"""
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
    configure_pool(1)
    print(f"\n{'═' * 88}")
    print(f"LLM Bench · 缓存冷启诊断  1 路对话 × {rounds} 轮问答")
    print(f"   responses : {base_urls['responses']}")
    print(f"   session   : {active_session}")
    print(f"{'═' * 88}")
    for format_name in formats:
        protocol = PROTOCOLS[format_name]
        protocol_base = base_urls[format_name]
        for model in models:
            model_session = session.configure(
                session.scoped(active_session, format_name, model)
            )
            print(f"\n▶ {format_name}  {model}  session={model_session}")
            rows, stats, gate = run_pool(
                workers=1,
                turns=int(rounds),
                duration=0,
                system=system_text,
                user=prompt,
                followup=followup_text,
                cache=True,
                session_prefix=model_session,
                protocol=protocol,
                base_url=protocol_base,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                timeout=180,
                retries=retries,
                retry_delay=retry_delay,
                throttle=throttle,
            )
            print_stress_summary(stats.snapshot(), limit=gate.limit, show_cache=True)
            if not rows:
                print("❌ 全部失败")
                continue
            steady = rows[1:]
            print(
                f"📊 第1轮(冷启)={cache_percent(rows[0])}  "
                f"第2轮起={aggregate_cache_percent(steady) if steady else 'n/a'}  "
                f"整体={aggregate_cache_percent(rows)}"
            )
    print()
