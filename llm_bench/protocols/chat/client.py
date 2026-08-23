"""OpenAI Chat Completions 流式协议适配器。"""

from __future__ import annotations

from ...metrics import (
    StreamMeasurement,
    cache_is_measurable,
    int_field,
    max_int_field,
)
from ... import session
from ...transport import ensure_success, iter_sse_lines, parse_data_line, post_stream


NAME = "OpenAI Chat Completions"
ENDPOINT = "/v1/chat/completions"


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
    """调用 ``/v1/chat/completions`` 并解析 choices/usage 增量。"""
    if messages:
        payload_messages = messages
    else:
        payload_messages = ([{"role": "system", "content": system}] if system else [])
        payload_messages.append({"role": "user", "content": user})
    payload = {
        "model": model,
        "messages": payload_messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    affinity = session.affinity_for(session_id)
    if affinity:
        session.configure(affinity)
        # 请求头负责会话亲和，prompt_cache_key 负责上游 Prompt Cache，两者取同一值。
        payload["prompt_cache_key"] = affinity
    else:
        session.clear()
    measurement = StreamMeasurement(on_progress=on_progress)
    input_tokens = output_tokens = cached_tokens = 0
    input_reported = output_reported = cache_reported = False
    stream_state: dict = {}

    with post_stream(f"{base_url}{ENDPOINT}", api_key, payload, timeout, session_id=affinity) as response:
        ensure_success(response)
        for raw in iter_sse_lines(
            response,
            allow_incomplete=allow_empty,
            stream_state=stream_state,
        ):
            event = parse_data_line(raw)
            if event is None:
                continue
            usage = event.get("usage")
            if isinstance(usage, dict):
                value, present = int_field(usage, "prompt_tokens", "input_tokens")
                if present:
                    input_tokens = max(input_tokens, value)
                    input_reported = True
                value, present = int_field(
                    usage, "completion_tokens", "output_tokens"
                )
                if present:
                    output_tokens = max(output_tokens, value)
                    output_reported = True
                details = (
                    usage.get("prompt_tokens_details")
                    or usage.get("input_tokens_details")
                    or {}
                )
                value, present = max_int_field(
                    [usage, details], "prompt_cache_hit_tokens", "cached_tokens"
                )
                if present:
                    cached_tokens = max(cached_tokens, value)
                    cache_reported = True

            choices = event.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                # 一个 SSE 只记一个时间点；推理内容参与 TTFT/CDL，
                # 但最终 text 只保留回答正文。
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""
                measurement.add_delta(content or reasoning)
                if content:
                    measurement.text_parts.append(content)

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
        empty_error="未收到任何 content token，可能请求失败或被截断",
    )
    result.update(stream_state)
    return result
