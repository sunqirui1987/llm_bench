"""所有协议适配器共享的缓存亲和 session 状态。"""

from __future__ import annotations

import threading
import uuid
from hashlib import sha256

_local = threading.local()


def configure(session_id: str = "") -> str:
    """开始一个 session；未指定固定值时始终生成全新的 ID。

    只写当前线程，50 路并发时不会把别人的 session 抢过来。
    """
    resolved = session_id.strip()
    if not resolved:
        resolved = _new_id()
    if any(char.isspace() or ord(char) < 32 for char in resolved):
        raise ValueError("session_id 不能包含空白字符或控制字符")
    _local.session_id = resolved
    return resolved


def current() -> str:
    """返回当前线程的亲和 session。"""
    return getattr(_local, "session_id", "") or ""


def clear() -> None:
    """本线程不再携带 session，用于不命中缓存的对话。"""
    _local.session_id = ""


def affinity_for(explicit: str | None) -> str:
    """None 沿用当前线程；空字符串表示不带 session。"""
    if explicit is None:
        return current().strip()
    return explicit.strip()


def rotate() -> str:
    """传输失败时生成新 session，以绕开可能异常的后端副本。"""
    return configure(_new_id())


def scoped(base_session_id: str, protocol: str, model: str) -> str:
    """为每个“协议 × 模型”生成稳定且彼此隔离的亲和 ID。

    使用摘要而不是直接拼接模型名，避免特殊字符进入 HTTP 请求头，也防止
    不同模型或协议复用同一个后端缓存状态。
    """
    digest = sha256(f"{protocol}\0{model}".encode("utf-8")).hexdigest()[:12]
    return f"{base_session_id[:80]}-{digest}"


def _new_id() -> str:
    return f"llm-bench-{uuid.uuid4().hex}"
