"""OpenAI Responses 标准流式协议适配器。"""

from __future__ import annotations

from ...config import apply_reasoning_effort
from ...metrics import (
    StreamMeasurement,
    cache_is_measurable,
    int_field,
    max_int_field,
)
from ... import session
from ...transport import ensure_success, iter_sse_lines, parse_data_line, post_stream


NAME = "OpenAI Responses"
ENDPOINT = "/v1/responses"


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
    reasoning_effort: str = "",
) -> dict:
    """调用 ``/v1/responses`` 并解析 response.* SSE 事件。"""
    if messages:
        input_messages = messages
    else:
        input_messages = ([{"role": "system", "content": system}] if system else [])
        input_messages.append({"role": "user", "content": user})
    payload = {
        "model": model,
        "input": input_messages,
        "max_output_tokens": max_tokens,
        "stream": True,
    }
    apply_reasoning_effort(payload, reasoning_effort, kind="responses")
    affinity = session.affinity_for(session_id)
    if affinity:
        session.configure(affinity)
        payload["prompt_cache_key"] = affinity
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
            if event_type in {
                "response.output_text.delta",
                "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta",
            }:
                measurement.add_delta(
                    event.get("delta", ""),
                    capture=event_type == "response.output_text.delta",
                )

            if event_type in {"response.completed", "response.incomplete"}:
                usage = (event.get("response") or {}).get("usage") or {}
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
                    [usage, details], "cached_tokens", "prompt_cache_hit_tokens"
                )
                if present:
                    cached_tokens = max(cached_tokens, value)
                    cache_reported = True

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
        empty_error="未收到任何 delta，可能请求失败或被截断",
    )
    result.update(stream_state)
    return result
