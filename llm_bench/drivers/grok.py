"""通过 grok CLI 发请求。关工具、覆盖系统提示，不限制 turn（长输出会超过 1 轮）。"""

from __future__ import annotations

import json
import shutil
import tempfile

from ..metrics import StreamMeasurement, int_field
from ..prompts import DEFAULT_OUTPUT_TOKENS, DEFAULT_SYSTEM
from .process import iter_process_lines, write_prompt_file
from .types import StreamRequest, combined_prompt

GROK_DISALLOWED = (
    "Agent,run_terminal_cmd,web_search,web_fetch,search_replace,"
    "read_file,grep,list_dir"
)

GROK_SYSTEM = (
    f"{DEFAULT_SYSTEM} "
    "Do not use tools. Do not inspect the repository. "
    f"Do not explain. Compact Lua only, at most {DEFAULT_OUTPUT_TOKENS} tokens, then stop."
)


def is_max_turns(text: str) -> bool:
    blob = (text or "").lower().replace("-", " ").replace("_", " ")
    return "max turns" in blob


def grok_argv(
    *,
    binary: str,
    model: str,
    effort: str,
    prompt_file: str,
    max_turns: int = 0,
) -> list[str]:
    argv = [
        binary,
        "--no-alt-screen",
        "--no-auto-update",
        "--no-subagents",
        "--no-plan",
        "--disable-web-search",
        "--verbatim",
        "--disallowed-tools",
        GROK_DISALLOWED,
        "--permission-mode",
        "dontAsk",
        "--system-prompt-override",
        GROK_SYSTEM,
        "--output-format",
        "streaming-json",
        "--prompt-file",
        prompt_file,
    ]
    if max_turns and int(max_turns) > 0:
        argv.extend(["--max-turns", str(int(max_turns))])
    if model:
        argv.extend(["-m", model])
    if effort:
        argv.extend(["--effort", effort])
    return argv


def _usage_from(event: dict) -> tuple[int, int, int, bool]:
    usage = event.get("usage") or {}
    if not isinstance(usage, dict):
        return 0, 0, 0, False
    uncached, has_in = int_field(usage, "input_tokens", "prompt_tokens")
    cached, has_cache = int_field(
        usage, "cache_read_input_tokens", "cached_tokens", "cacheReadInputTokens"
    )
    created, _ = int_field(usage, "cache_creation_input_tokens")
    out, has_out = int_field(usage, "output_tokens", "completion_tokens")
    # grok headless：input_tokens 是未缓存部分。bench 的 cache% 要总分母。
    total_in = uncached + cached + created
    present = has_in or has_cache or has_out
    return total_in, out, cached, present


class GrokDriver:
    via = "grok"
    name = "Grok CLI"
    endpoint = "grok -p"

    def __init__(self, binary: str = "grok", *, popen=None):
        self.binary = binary or "grok"
        self.popen = popen

    def stream(self, req: StreamRequest) -> dict:
        binary = shutil.which(self.binary) or self.binary
        prompt = combined_prompt(req.messages)
        folder = tempfile.mkdtemp(prefix="llm-bench-grok-")
        try:
            path = write_prompt_file(prompt, directory=folder)
            argv = grok_argv(
                binary=binary,
                model=req.model,
                effort=req.reasoning_effort,
                prompt_file=path,
            )
            kwargs = {}
            if self.popen is not None:
                kwargs["popen"] = self.popen
            measurement = StreamMeasurement(on_progress=req.on_progress)
            input_tokens = output_tokens = cached_tokens = 0
            usage_reported = False
            try:
                for line in iter_process_lines(argv, timeout=req.timeout, **kwargs):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        measurement.add_delta(line, capture=True)
                        continue
                    if not isinstance(event, dict):
                        continue
                    typ = event.get("type") or ""
                    if typ in {"text", "thought"}:
                        text = str(event.get("data") or "")
                        measurement.add_delta(text, capture=typ == "text")
                    elif typ in {"error", "max_turns_reached"}:
                        msg = str(event.get("message") or typ)
                        if is_max_turns(msg) or typ == "max_turns_reached":
                            continue
                        raise RuntimeError(msg[:300])
                    inp, out, cached, present = _usage_from(event)
                    if present:
                        input_tokens = max(input_tokens, inp)
                        output_tokens = max(output_tokens, out)
                        cached_tokens = max(cached_tokens, cached)
                        usage_reported = True
            except RuntimeError as exc:
                if not measurement.streamed_parts or not is_max_turns(str(exc)):
                    raise
            return measurement.finalize(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                cache_reported=usage_reported,
                usage_reported=usage_reported,
                allow_empty=req.allow_empty,
                empty_error="grok 没有产出任何 text/thought",
            )
        finally:
            shutil.rmtree(folder, ignore_errors=True)
