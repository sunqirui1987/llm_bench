"""子进程：写 prompt 文件、按行读 stdout、超时杀进程组。"""

from __future__ import annotations

import io
import os
import select
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from ..transport import HttpStatusError


def write_prompt_file(text: str, *, directory: str | None = None) -> str:
    folder = Path(directory) if directory else Path(tempfile.mkdtemp(prefix="llm-bench-"))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "prompt.txt"
    path.write_text(text or "", encoding="utf-8")
    return str(path)


def kill_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()


def _can_select(stream) -> bool:
    try:
        stream.fileno()
        return True
    except (AttributeError, OSError, io.UnsupportedOperation):
        return False


def _readline(stream, deadline: float):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None, True
    if _can_select(stream):
        ready, _, _ = select.select([stream], [], [], remaining)
        if not ready:
            return None, True
    return stream.readline(), False


def look_rate_limited(text: str) -> bool:
    blob = (text or "").lower()
    return "429" in blob or "rate limit" in blob or "too many" in blob


def iter_process_lines(
    argv: list[str],
    *,
    timeout: int,
    cwd: str | None = None,
    env: dict | None = None,
    stdin_text: str | None = None,
    popen=subprocess.Popen,
) -> Iterator[str]:
    """启动 argv，逐行产出 stdout。结束码非 0 或超时则抛错。"""
    stderr_chunks: list[bytes] = []
    proc = popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    def drain_err() -> None:
        try:
            stderr_chunks.append(proc.stderr.read() or b"")
        except Exception:
            pass

    err_thread = threading.Thread(target=drain_err, name="llm-bench-stderr", daemon=True)
    err_thread.start()
    if stdin_text is not None and proc.stdin is not None:
        proc.stdin.write(stdin_text.encode("utf-8", errors="replace"))
        proc.stdin.close()

    deadline = time.monotonic() + max(int(timeout), 1)
    try:
        while True:
            raw, timed_out = _readline(proc.stdout, deadline)
            if timed_out:
                kill_group(proc)
                raise TimeoutError(f"进程超时 {timeout}s: {' '.join(argv[:6])}")
            if not raw:
                break
            if isinstance(raw, bytes):
                yield raw.decode("utf-8", errors="replace").rstrip("\r\n")
            else:
                yield str(raw).rstrip("\r\n")
        code = proc.wait(timeout=5)
    except TimeoutError:
        kill_group(proc)
        raise
    except subprocess.TimeoutExpired:
        kill_group(proc)
        raise TimeoutError(f"进程结束等待超时: {' '.join(argv[:6])}")
    finally:
        err_thread.join(timeout=1)
        if proc.poll() is None:
            kill_group(proc)

    err = b"".join(stderr_chunks).decode("utf-8", errors="replace")
    if code and code != 0:
        if look_rate_limited(err):
            raise HttpStatusError(429, err[:300])
        raise RuntimeError(
            f"进程退出 {code}: {err.strip()[:300] or ' '.join(argv[:8])}"
        )
