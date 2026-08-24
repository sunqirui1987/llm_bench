"""命令行：bench / cache。真正的对话循环在 engine。"""

from __future__ import annotations

from datetime import datetime

from . import session
from .config import (
    DEFAULT_BASE_URL,
    DEFAULT_CHAT_BASE_URL,
    DEFAULT_EFFORT,
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
    parse_reasoning_effort,
    resolve_base_urls,
)
from .drivers.registry import parse_via, resolve_driver
from .engine import run_pool, turn_request as _turn_request
from .games import GAMES
from .prompts import (
    CONTEXT_WINDOW,
    DEFAULT_OUTPUT_TOKENS,
    compose_system,
    compose_user,
    fit_max_input,
    plan_request,
)
from .protocols.registry import PROTOCOLS
from .reporting import (
    aggregate_cache_percent,
    cache_percent,
    first_success_by_worker,
    is_first_request,
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
    require_api_key: bool = True,
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
    key = ensure_api_key(api_key) if require_api_key else (api_key or "")
    return (
        model_list,
        format_list,
        base_urls,
        key,
        session.configure(session_id),
    )


def _build_prompts(
    *,
    system_prompt: str,
    system_file: str,
    prompt: str,
    prompt_file: str,
    followup: str,
    context_file: str,
) -> tuple[str, str, str]:
    return (
        compose_system(
            text=system_prompt,
            file=system_file,
            context_file=context_file,
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
    prompt: str,
    max_input: int,
    context_window: int,
    plan: dict,
    pad: bool,
    custom_prompt: bool,
    custom_system: bool,
    effort: str = "high",
    via: str = "http",
    cmd: str = "",
) -> list[str]:
    total = workers * max(int(rounds), 0)
    if hit_cache:
        title = "LLM Bench · 要缓存"
        session_desc = f"每路钉死一条 session（示例 {initial_session}）"
        cache_flow = "每路第一次成功为预热（不计命中率）；之后同一条命令再发，测 cache"
    else:
        title = "LLM Bench · 不要缓存"
        session_desc = "不带 session"
        cache_flow = "每一次都换新命令：换盐、换场面；游戏仍按 work 分开"
    lines = [
        "═" * 88,
        (
            f"{title}  workers={workers}（同时开工，互不等待）  "
            f"rounds={rounds}（每路自己连发几遍）  合计最多 {total} 次  "
            f"duration={duration or 'off'}s  cache_mode={cache_mode}"
        ),
    ]
    if fd_limit:
        lines.append(f"   fd limit : {fd_limit}")
    lines.append(f"   via      : {via}")
    if via == "http":
        lines.extend(
            [
                f"   chat      : {base_urls['chat']}",
                f"   responses : {base_urls['responses']}",
                f"   messages  : {base_urls['messages']}",
                f"   formats   : {', '.join(formats)}",
            ]
        )
    elif cmd:
        lines.append(f"   cmd      : {cmd}")
    lines.extend(
        [
            f"   models    : {', '.join(models)}",
            f"   session   : {session_desc}",
            f"   cache     : {cache_flow}",
            f"   retries   : {retries}  timeout={timeout}s  delay={retry_delay}s",
            (
                f"   effort    : {effort or '不传（网关默认）'}  "
                f"（low/medium/high/xhigh）"
            ),
            (
                f"   window    : context={context_window}  "
                f"input≈{plan['input_tokens']}  output_cap={plan['max_tokens']}"
            ),
        ]
    )
    lines.append(
        f"   prefix    : {'按每路游戏填到 '+str(plan['input_tokens'])+' token' if pad else '短指令，不填充'}"
    )
    n_games = len(GAMES)
    if workers > n_games:
        lines.append(
            f"   games     : {n_games} 款游戏循环复用（work{n_games + 1} 与 work1 同一款）"
        )
    if custom_system:
        lines.append("   system    : 自定义系统提示（所有 work 共用，叠在各自游戏设定上）")
    if custom_prompt:
        lines.append(
            f"   prompt    : 自定义（覆盖所有 work 的用户命令） {prompt[:50]}{'...' if len(prompt) > 50 else ''}"
        )
    else:
        lines.append("   prompt    : 每路一个 Pulse/Lua 模块（只输出代码，games.py）")
    if hit_cache and max_input < 2048:
        lines.append(
            f"⚠️ cache_mode=hit 但 system 只有 {max_input} token，"
            "多数服务要 1024+ 才缓存。请加大 --input_tokens。"
        )
    lines.append("═" * 88)
    return lines


def bench(
    models=DEFAULT_MODELS,
    formats="responses",
    base_url: str = DEFAULT_BASE_URL,
    chat_base_url: str = DEFAULT_CHAT_BASE_URL,
    responses_base_url: str = DEFAULT_RESPONSES_BASE_URL,
    messages_base_url: str = DEFAULT_MESSAGES_BASE_URL,
    api_key: str = "",
    max_tokens: int = DEFAULT_OUTPUT_TOKENS,
    prompt: str = DEFAULT_PROMPT,
    prompt_file: str = "",
    followup: str = "",
    rounds: int = 2,
    throttle: float = 0.0,
    system: str = "long",
    system_prompt: str = "",
    system_file: str = "",
    input_tokens: int = TARGET_INPUT_TOKENS,
    context_window: int = CONTEXT_WINDOW,
    context_file: str = "",
    session_id: str = "",
    retries: int = 2,
    retry_delay: float = 1.0,
    timeout: int = 7200,
    workers: int = DEFAULT_WORKERS,
    duration: float = 0.0,
    cache_mode: str = "hit",
    effort: str = DEFAULT_EFFORT,
    via: str = "http",
    cmd: str = "",
    grok_bin: str = "grok",
    codex_bin: str = "codex",
):
    """workers 路同时发命令。

    --cache_mode hit   同一条命令原样再发，粘 session。每路第一次成功为冷启动，不计命中率。
    --cache_mode miss  每一次都换新命令，不带 session。界面不标预热/缓存。
    --rounds           每路自己连发几遍，互不等待（默认 2：一次冷、一次热）。
    --effort           推理强度：low / medium / high / xhigh（默认 high，对齐 Grok/sub2api）。
    --via              通道：http（默认，LLM 请求）/ grok / codex / cmd。
    --cmd              --via cmd 时的程序模板，占位 {prompt_file} {model} {effort}。
    --prompt/--prompt_file
                       覆盖所有 work 的用户命令。不传则每路用自己那款游戏的开场。
    --system           long=按游戏填充到 --input_tokens（默认约 30 万）；short=短系统提示，不填充。
    --system_prompt/--system_file
                       叠在每路游戏设定上，所有 work 共用这段。
    """
    via = parse_via(via)
    models, formats, base_urls, api_key, initial_session = _prepare(
        models,
        formats,
        base_url,
        chat_base_url,
        responses_base_url,
        messages_base_url,
        api_key,
        session_id,
        require_api_key=via == "http",
    )
    if via != "http":
        formats = [via]
    if via == "cmd" and not str(cmd or "").strip():
        raise ValueError("--via cmd 需要 --cmd，例如 'my-llm --file {prompt_file}'")
    cache_mode = parse_cache_mode(cache_mode)
    effort = parse_reasoning_effort(effort)
    hit_cache = cache_mode == "hit"
    context_window = max(int(context_window), 1)
    max_tokens = max(int(max_tokens), 1)
    max_input = fit_max_input(input_tokens, max_tokens, context_window)
    plan = plan_request(
        max_input=max_input,
        max_tokens=max_tokens,
        context_window=context_window,
    )
    pad = (system or "long").strip().lower() != "short"
    custom_prompt = bool(str(prompt_file or "").strip()) or (
        bool((prompt or "").strip())
        and (prompt or "").strip() != (DEFAULT_PROMPT or "").strip()
    )
    custom_system = bool(str(system_file or "").strip()) or bool(
        (system_prompt or "").strip()
    )
    system_text, prompt, followup_text = _build_prompts(
        system_prompt=system_prompt,
        system_file=system_file,
        prompt=prompt,
        prompt_file=prompt_file,
        followup=followup,
        context_file=context_file,
    )
    user_for_pool = prompt if custom_prompt else ""
    system_for_pool = system_text if custom_system else ""
    workers = max(int(workers), 1)
    duration = max(float(duration), 0.0)
    fd_limit = raise_fd_limit(workers * 8 + 256)
    configure_pool(workers)
    header = _print_start(
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
        prompt=prompt,
        max_input=max_input,
        context_window=context_window,
        plan=plan,
        pad=pad,
        custom_prompt=custom_prompt,
        custom_system=custom_system,
        effort=effort,
        via=via,
        cmd=cmd,
    )

    summary: dict = {}
    snapshots: dict = {}
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for format_name in formats:
        driver = resolve_driver(
            via,
            format_name=format_name if via == "http" else "responses",
            cmd=cmd,
            grok_bin=grok_bin,
            codex_bin=codex_bin,
        )
        protocol_base = base_urls.get(format_name, "") if via == "http" else ""
        for model in models:
            live_header = [
                *header,
                (
                    f"▶ {via}  {protocol_base}{getattr(driver, 'endpoint', '')}  "
                    f"model={model}"
                ),
            ]
            rows, stats, gate = run_pool(
                workers=workers,
                rounds=int(rounds),
                duration=duration,
                system=system_for_pool,
                user=user_for_pool,
                followup=followup_text,
                cache=hit_cache,
                session_prefix=initial_session or "llm-bench",
                driver=driver,
                base_url=protocol_base,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                timeout=timeout,
                retries=retries,
                retry_delay=retry_delay,
                throttle=throttle,
                max_input=max_input if pad else 0,
                context_window=context_window,
                pad=pad,
                header=live_header,
                reasoning_effort=effort,
            )
            snap = stats.snapshot()
            snapshots[(format_name, model)] = snap
            print_stress_summary(
                snap, limit=gate.limit, show_cache=hit_cache, rows=rows
            )
            warm_hit = aggregate_cache_percent(rows, skip_first=True).strip()
            if not hit_cache and rows:
                try:
                    warm_pct = float(warm_hit.replace("%", "").strip())
                except ValueError:
                    warm_pct = 0.0
                if warm_pct > 5:
                    print(
                        f"⚠️ cache_mode=miss 第2次起仍有 {warm_pct:.1f}% 命中"
                    )
            if not rows:
                print("⚠️ 没有成功样本")
                continue
            planned = int(rounds) * workers
            if len(rows) == planned:
                print(
                    f"   有效请求: {len(rows)}（{workers} 路 × 每路 {int(rounds)} 次）"
                )
            else:
                print(
                    f"   有效请求: {len(rows)} / {planned}"
                    f"（{workers} 路 × 每路最多 {int(rounds)} 次）"
                )
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
            "rounds": rounds,
            "system": system,
            "effort": effort or "omit",
            "via": via,
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
    max_tokens: int = DEFAULT_OUTPUT_TOKENS,
    prompt: str = DEFAULT_PROMPT,
    prompt_file: str = "",
    followup: str = "",
    rounds: int = 2,
    throttle: float = 0.3,
    system: str = "long",
    system_prompt: str = "",
    system_file: str = "",
    input_tokens: int = TARGET_INPUT_TOKENS,
    context_window: int = CONTEXT_WINDOW,
    context_file: str = "",
    session_id: str = "",
    retries: int = 2,
    retry_delay: float = 1.0,
    effort: str = DEFAULT_EFFORT,
    via: str = "http",
    cmd: str = "",
    grok_bin: str = "grok",
    codex_bin: str = "codex",
):
    """单路缓存诊断：同一条命令连发 rounds 次，看第 1 次冷启 → 第 2 次起命中。"""
    via = parse_via(via)
    models, formats, base_urls, api_key, active_session = _prepare(
        models,
        formats,
        base_url,
        chat_base_url,
        responses_base_url,
        messages_base_url,
        api_key,
        session_id,
        require_api_key=via == "http",
    )
    if via != "http":
        formats = [via]
    if via == "cmd" and not str(cmd or "").strip():
        raise ValueError("--via cmd 需要 --cmd")
    context_window = max(int(context_window), 1)
    max_tokens = max(int(max_tokens), 1)
    effort = parse_reasoning_effort(effort)
    max_input = fit_max_input(input_tokens, max_tokens, context_window)
    custom_prompt = bool(str(prompt_file or "").strip()) or (
        bool((prompt or "").strip())
        and (prompt or "").strip() != (DEFAULT_PROMPT or "").strip()
    )
    custom_system = bool(str(system_file or "").strip()) or bool(
        (system_prompt or "").strip()
    )
    system_text, prompt, followup_text = _build_prompts(
        system_prompt=system_prompt,
        system_file=system_file,
        prompt=prompt,
        prompt_file=prompt_file,
        followup=followup,
        context_file=context_file,
    )
    user_for_pool = prompt if custom_prompt else ""
    system_for_pool = system_text if custom_system else ""
    configure_pool(1)
    for format_name in formats:
        driver = resolve_driver(
            via,
            format_name=format_name if via == "http" else "responses",
            cmd=cmd,
            grok_bin=grok_bin,
            codex_bin=codex_bin,
        )
        protocol_base = base_urls.get(format_name, "") if via == "http" else ""
        for model in models:
            model_session = session.configure(
                session.scoped(active_session, format_name, model)
            )
            header = [
                "═" * 88,
                f"LLM Bench · 缓存冷启诊断  1 路把同一条命令连发 {rounds} 次",
                f"   via      : {via}",
                f"   session   : {model_session}",
                f"   effort    : {effort or '不传（网关默认）'}",
                "═" * 88,
                f"▶ {via}  {model}  session={model_session}",
            ]
            rows, stats, gate = run_pool(
                workers=1,
                rounds=int(rounds),
                duration=0,
                system=system_for_pool,
                user=user_for_pool,
                followup=followup_text,
                cache=True,
                session_prefix=model_session,
                driver=driver,
                base_url=protocol_base,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                timeout=7200,
                retries=retries,
                retry_delay=retry_delay,
                throttle=throttle,
                max_input=max_input,
                context_window=context_window,
                pad=True,
                header=header,
                reasoning_effort=effort,
            )
            print_stress_summary(
                stats.snapshot(), limit=gate.limit, show_cache=True, rows=rows
            )
            if not rows:
                print("❌ 全部失败")
                continue
            first_map = first_success_by_worker(rows)
            first = [row for row in rows if is_first_request(row, first_map)]
            later = [row for row in rows if not is_first_request(row, first_map)]
            print(
                f"📊 第1次(冷启)={aggregate_cache_percent(first) if first else cache_percent(rows[0])}  "
                f"第2次起={aggregate_cache_percent(later) if later else 'n/a'}  "
                f"整体={aggregate_cache_percent(rows)}"
            )
    print()
