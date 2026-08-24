"""自定义程序。长 prompt 只走 {prompt_file}，不要塞进 argv。"""

from __future__ import annotations

import json
import shlex
import shutil
import tempfile

from ..metrics import StreamMeasurement
from .process import iter_process_lines, write_prompt_file
from .types import StreamRequest, combined_prompt


def render_cmd(
    template: str,
    *,
    model: str,
    effort: str,
    max_tokens: int,
    prompt_file: str,
    session_id: str,
) -> list[str]:
    text = (template or "").strip()
    if not text:
        raise ValueError("--via cmd 需要 --cmd 模板，例如 'my-llm --file {prompt_file}'")
    filled = text.format(
        model=model,
        effort=effort,
        max_tokens=max_tokens,
        prompt_file=prompt_file,
        session_id=session_id or "",
    )
    return shlex.split(filled)


class CmdDriver:
    via = "cmd"
    name = "custom command"
    endpoint = "cmd"

    def __init__(self, template: str, *, popen=None):
        self.template = template
        self.popen = popen
        self.endpoint = template or "cmd"

    def stream(self, req: StreamRequest) -> dict:
        prompt = combined_prompt(req.messages)
        folder = tempfile.mkdtemp(prefix="llm-bench-cmd-")
        try:
            path = write_prompt_file(prompt, directory=folder)
            argv = render_cmd(
                self.template,
                model=req.model,
                effort=req.reasoning_effort,
                max_tokens=req.max_tokens,
                prompt_file=path,
                session_id=req.session_id,
            )
            kwargs = {}
            if self.popen is not None:
                kwargs["popen"] = self.popen
            measurement = StreamMeasurement(on_progress=req.on_progress)
            for line in iter_process_lines(argv, timeout=req.timeout, **kwargs):
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    measurement.add_delta(line + "\n", capture=True)
                    continue
                if isinstance(event, dict):
                    text = str(
                        event.get("data")
                        or event.get("text")
                        or event.get("content")
                        or ""
                    )
                    if text:
                        measurement.add_delta(text, capture=True)
                    elif event.get("type") == "error":
                        raise RuntimeError(str(event.get("message") or line)[:300])
                else:
                    measurement.add_delta(line + "\n", capture=True)
            return measurement.finalize(
                input_tokens=0,
                output_tokens=0,
                cached_tokens=0,
                cache_reported=False,
                usage_reported=False,
                allow_empty=req.allow_empty,
                empty_error="自定义命令没有产出任何 stdout",
            )
        finally:
            shutil.rmtree(folder, ignore_errors=True)
