"""通过 `codex exec --json` 发请求。事件级，不是 token 增量。"""

from __future__ import annotations

import json
import shutil

from ..metrics import StreamMeasurement, int_field
from .process import iter_process_lines
from .types import StreamRequest, combined_prompt


def codex_argv(*, binary: str, model: str) -> list[str]:
    argv = [binary, "exec", "--json", "--ephemeral"]
    if model:
        argv.extend(["-m", model])
    argv.append("-")
    return argv


def _item_text(event: dict) -> tuple[str, bool]:
    item = event.get("item") or event.get("msg") or {}
    if not isinstance(item, dict):
        return "", False
    kind = str(item.get("type") or event.get("type") or "")
    text = str(item.get("text") or item.get("content") or "")
    if kind in {"agent_message", "AgentMessage", "reasoning"}:
        return text, kind == "reasoning"
    if event.get("type") in {"item.completed", "item.updated"}:
        if kind in {"agent_message", "reasoning"}:
            return text, kind == "reasoning"
    return "", False


def _usage_from(event: dict) -> tuple[int, int, int, bool]:
    usage = event.get("usage") or (event.get("item") or {}).get("usage") or {}
    if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
        usage = event["usage"]
    if not isinstance(usage, dict):
        return 0, 0, 0, False
    inp, has_in = int_field(usage, "input_tokens", "prompt_tokens")
    cached, has_cache = int_field(
        usage, "cached_input_tokens", "cached_tokens", "cache_read_input_tokens"
    )
    out, has_out = int_field(usage, "output_tokens", "completion_tokens")
    if has_in and inp >= cached:
        total_in = inp
    else:
        total_in = inp + cached
    return total_in, out, cached, has_in or has_cache or has_out


class CodexDriver:
    via = "codex"
    name = "Codex CLI"
    endpoint = "codex exec"

    def __init__(self, binary: str = "codex", *, popen=None):
        self.binary = binary or "codex"
        self.popen = popen

    def stream(self, req: StreamRequest) -> dict:
        binary = shutil.which(self.binary) or self.binary
        prompt = combined_prompt(req.messages)
        argv = codex_argv(binary=binary, model=req.model)
        kwargs = {}
        if self.popen is not None:
            kwargs["popen"] = self.popen
        measurement = StreamMeasurement(on_progress=req.on_progress)
        input_tokens = output_tokens = cached_tokens = 0
        usage_reported = False
        seen_messages: set[str] = set()
        for line in iter_process_lines(
            argv, timeout=req.timeout, stdin_text=prompt, **kwargs
        ):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                measurement.add_delta(line, capture=True)
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") in {"error", "turn.failed"}:
                raise RuntimeError(str(event.get("message") or event)[:300])
            text, reasoning = _item_text(event)
            if text:
                key = text[:80]
                if key not in seen_messages:
                    seen_messages.add(key)
                    measurement.add_delta(text, capture=not reasoning)
            inp, out, cached, present = _usage_from(event)
            if present:
                input_tokens = max(input_tokens, inp)
                output_tokens = max(output_tokens, out)
                cached_tokens = max(cached_tokens, cached)
                usage_reported = True
        return measurement.finalize(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cache_reported=usage_reported,
            usage_reported=usage_reported,
            allow_empty=req.allow_empty,
            empty_error="codex 没有产出任何消息",
        )
