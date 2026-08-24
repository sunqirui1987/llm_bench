"""共享的 HTTP/SSE 传输、断流回收与重试策略。"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from typing import Optional

import requests
from requests.adapters import HTTPAdapter

from . import session

_http: Optional[requests.Session] = None
_http_lock = threading.Lock()
_local = threading.local()
SSE_CHUNK_SIZE = 256


def headers(api_key: str, session_id: str | None = None) -> dict[str, str]:
    """构造流式请求头，并附加可穿过代理的双重会话标识。"""
    result = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Accept-Encoding": "identity",
        "Authorization": f"Bearer {api_key}",
    }
    affinity = session.affinity_for(session_id)
    if affinity:
        # 同时发送下划线头和连字符头：部分代理会丢掉 session_id。
        result["session_id"] = affinity
        result["X-Session-Affinity"] = affinity
    return result


def configure_pool(size: int) -> int:
    """关闭旧的共享 Session。真正的连接按线程隔离。"""
    global _http
    size = max(int(size), 1)
    with _http_lock:
        old = _http
        _http = None
    if old is not None:
        old.close()
    return size


def thread_session() -> requests.Session:
    """每个 worker 线程自己的 HTTP Session，一路一条连接。"""
    http = getattr(_local, "http", None)
    if http is None:
        http = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=2,
            pool_maxsize=2,
            max_retries=0,
            pool_block=False,
        )
        http.mount("http://", adapter)
        http.mount("https://", adapter)
        _local.http = http
    return http


def raise_fd_limit(needed: int) -> int:
    """尽量把进程可打开文件数抬到并发连接所需规模。"""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        unlimited = hard < 0
        try:
            unlimited = unlimited or hard == resource.RLIM_INFINITY
        except Exception:
            pass
        cap = needed if unlimited else hard
        target = min(max(int(needed), soft), cap)
        if target > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        return int(resource.getrlimit(resource.RLIMIT_NOFILE)[0])
    except (ValueError, OSError, AttributeError):
        return 0


def post_stream(
    url: str,
    api_key: str,
    payload: dict,
    timeout: int,
    session_id: str | None = None,
):
    """发起流式 POST；响应的生命周期由协议适配器管理。"""
    http = thread_session()
    return http.post(
        url,
        headers=headers(api_key, session_id),
        json=payload,
        stream=True,
        timeout=timeout,
    )


class HttpStatusError(RuntimeError):
    """带状态码的 HTTP 错误；429/5xx 可由上层决定是否重试。"""

    def __init__(self, status_code: int, body: str = "", retry_after: float | None = None):
        self.status_code = int(status_code)
        self.body = body or ""
        self.retry_after = retry_after
        super().__init__(f"HTTP {self.status_code}: {self.body[:300]}")

    @property
    def rate_limited(self) -> bool:
        return self.status_code == 429

    @property
    def capacity_limited(self) -> bool:
        return self.status_code in {502, 503, 504}

    @property
    def retryable(self) -> bool:
        return self.status_code in {429, 502, 503, 504}


def _retry_after_seconds(response) -> float | None:
    raw = ""
    try:
        raw = (response.headers or {}).get("Retry-After") or ""
    except Exception:
        return None
    try:
        return max(float(raw), 0.0)
    except (TypeError, ValueError):
        return None


def ensure_success(response) -> None:
    """将非 200 响应转换为带状态码的明确异常。"""
    if response.status_code == 200:
        return
    body = ""
    try:
        body = response.text[:300]
    except Exception:
        body = ""
    raise HttpStatusError(response.status_code, body, _retry_after_seconds(response))


def iter_sse_lines(
    response,
    *,
    allow_incomplete: bool = False,
    stream_state: Optional[dict] = None,
) -> Iterator[bytes | str]:
    """及时交付 SSE；允许时可回收已经产生有效进度的断流。"""
    state = stream_state if stream_state is not None else {}
    terminal_received = False
    progress_received = False
    pending_event_type = None
    state.update({"terminal_received": False, "incomplete_stream": False})
    try:
        # 小 chunk 避免 urllib3 默认的 512 字节缓冲：网关提前断流时，短 SSE
        # 可能尚未 yield 就随 ChunkedEncodingError 一起丢失。
        for raw in response.iter_lines(chunk_size=SSE_CHUNK_SIZE, decode_unicode=False):
            terminal_received = terminal_received or is_terminal_line(raw)
            line = decode_line(raw)
            if line.startswith("event:"):
                pending_event_type = line[len("event:"):].strip()
                yield raw
                continue
            event = parse_data_line(raw)
            if event is not None:
                # OpenAI Responses 可能只在 event: 行声明类型，
                # 对应 data: JSON 中没有 type。
                event_type = event.get("type") or pending_event_type or ""
                progress_received = progress_received or bool(
                    event.get("usage")
                    or (event.get("message") or {}).get("usage")
                    or (event.get("response") or {}).get("usage")
                    or event_type.endswith(".delta")
                    or event_type == "content_block_delta"
                    or event.get("choices")
                )
                pending_event_type = None
            yield raw
    except requests.exceptions.ChunkedEncodingError:
        state["terminal_received"] = terminal_received
        if not terminal_received and not (allow_incomplete and progress_received):
            raise
        state["incomplete_stream"] = not terminal_received
    else:
        state["terminal_received"] = terminal_received


def parse_data_line(raw) -> Optional[dict]:
    """解析一行 ``data:`` SSE；控制行、DONE 与坏 JSON 返回 None。"""
    line = decode_line(raw)
    if not line.startswith("data:"):
        return None
    data = line[len("data:"):].strip()
    if not data or data == "[DONE]":
        return None
    try:
        parsed = json.loads(data)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def decode_line(raw) -> str:
    """把 requests 返回的 bytes/str 统一为 UTF-8 字符串。"""
    return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw


def is_terminal_line(raw) -> bool:
    """识别三种协议的正常结束事件。"""
    line = decode_line(raw)
    if not line.startswith("data:"):
        return False
    data = line[len("data:"):].strip()
    if data == "[DONE]":
        return True
    event = parse_data_line(raw) or {}
    return event.get("type") in {
        "response.completed",
        "response.incomplete",
        "message_stop",
    }


def call_with_retries(
    function: Callable,
    args: tuple,
    retries: int,
    retry_delay: float,
    *,
    rotate_session_on_retry: bool = True,
) -> dict:
    """重试连接级异常；调用方决定是否更换缓存亲和 session。"""
    retries = max(int(retries), 0)
    retry_delay = max(float(retry_delay), 0.0)
    retryable = (
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
    )
    for attempt in range(retries + 1):
        try:
            result = function(*args)
            result["retry_count"] = attempt
            result["session_id"] = session.current()
            return result
        except HttpStatusError:
            # 429/5xx 必须立刻交给上层：429 收缩并发，503 释放在途后再退避。
            # 不能占着连接槽在这里重试。
            raise
        except retryable as exc:
            if attempt >= retries:
                raise RuntimeError(
                    f"流传输中断，重试 {retries} 次后仍失败: {exc}"
                ) from exc
            if rotate_session_on_retry:
                session.rotate()
            time.sleep(retry_delay * (2 ** attempt))
    raise AssertionError("unreachable")
