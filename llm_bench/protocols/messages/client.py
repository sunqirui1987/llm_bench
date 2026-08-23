"""Anthropic Messages 兼容流式协议适配器。"""

from __future__ import annotations

from ...metrics import (
    StreamMeasurement,
    cache_is_measurable,
    int_field,
    max_int_field,
    usage_candidates,
)
from ... import session
from ...transport import ensure_success, iter_sse_lines, parse_data_line, post_stream


NAME = "Anthropic Messages (compat)"
ENDPOINT = "/v1/messages"


def stream(
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 2048,
    timeout: int = 180,
    allow_empty: bool = False,
    messages: list | None = None,
    session_id: str | None = None,
    on_progress=None,
) -> dict:
    """调用 ``/v1/messages`` 并兼容非标准嵌套 usage。"""
    if messages:
        system_text = system or ""
        conv = []
        for item in messages:
            role = item.get("role") or "user"
            content = item.get("content") or ""
            if role == "system":
                system_text = (system_text + chr(10) + content).strip() if system_text else content
            else:
                conv.append({"role": role, "content": content})
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "stream": True,
            "messages": conv or [{"role": "user", "content": user}],
        }
        if system_text:
            payload["system"] = system_text
    else:
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "stream": True,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            payload["system"] = system
    affinity = session.affinity_for(session_id)
    if affinity:
        session.configure(affinity)
    else:
        session.clear()
    measurement = StreamMeasurement(on_progress=on_progress)
    input_tokens = output_tokens = cached_tokens = 0
    input_reported = output_reported = cache_reported = False
    stream_state: dict = {}
    sse_event_type = None

    with post_stream(f"{base_url}{ENDPOINT}", api_key, payload, timeout, session_id=affinity) as response:
        ensure_success(response)
        for raw in iter_sse_lines(
            response,
            allow_incomplete=allow_empty,
            stream_state=stream_state,
        ):
            line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
            if line.startswith("event:"):
                sse_event_type = line[len("event:"):].strip()
                continue
            if not line:
                sse_event_type = None
                continue
            event = parse_data_line(raw)
            if event is None:
                continue
            event_type = event.get("type") or sse_event_type
            if event_type == "content_block_delta":
                delta = event.get("delta") or {}
                delta_type = delta.get("type")
                text = ""
                if delta_type == "text_delta":
                    text = delta.get("text", "")
                elif delta_type == "thinking_delta":
                    text = delta.get("thinking", "")
                measurement.add_delta(text, capture=delta_type == "text_delta")

            for usage in usage_candidates(event):
                value, present = int_field(usage, "input_tokens", "prompt_tokens")
                if present:
                    input_tokens = max(input_tokens, value)
                    input_reported = True
                value, present = int_field(
                    usage, "output_tokens", "completion_tokens"
                )
                if present:
                    output_tokens = max(output_tokens, value)
                    output_reported = True
                details = (
                    usage.get("input_tokens_details")
                    or usage.get("prompt_tokens_details")
                    or {}
                )
                value, present = max_int_field(
                    [usage, details],
                    "cache_read_input_tokens",
                    "cached_tokens",
                    "prompt_cache_hit_tokens",
                )
                if present:
                    cached_tokens = max(cached_tokens, value)
                    cache_reported = True

    # 部分 Messages 兼容层把未缓存输入与 cache-read 分开上报，需合并为总输入。
    if input_reported and cache_reported:
        input_tokens += cached_tokens
    cache_reported = cache_is_measurable(
        input_tokens if input_reported else 0, cache_reported
    )
    result = measurement.finalize(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        cache_reported=cache_reported,
        usage_reported=input_reported or output_reported or cache_reported,
        allow_empty=allow_empty,
        empty_error="未收到任何 content_block_delta，可能请求失败或被截断",
    )
    result.update(stream_state)
    return result
