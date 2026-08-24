"""HTTP 通道：沿用现有 chat/responses/messages 适配器。"""

from __future__ import annotations

from .types import StreamRequest


class HttpDriver:
    def __init__(self, protocol):
        self.protocol = protocol
        self.name = getattr(protocol, "name", "http")
        self.endpoint = getattr(protocol, "endpoint", "")
        self.via = "http"

    def stream(self, req: StreamRequest) -> dict:
        return self.protocol.stream(
            req.base_url,
            req.api_key,
            req.model,
            "",
            "",
            req.max_tokens,
            req.timeout,
            allow_empty=req.allow_empty,
            messages=req.messages,
            session_id=req.session_id,
            on_progress=req.on_progress,
            reasoning_effort=req.reasoning_effort,
        )


def as_driver(protocol_or_driver):
    """测试仍传 protocol 对象时，包成 HttpDriver。"""
    if protocol_or_driver is None:
        raise RuntimeError("缺少 driver/protocol")
    if isinstance(protocol_or_driver, HttpDriver):
        return protocol_or_driver
    if hasattr(protocol_or_driver, "stream") and not hasattr(
        protocol_or_driver, "via"
    ):
        return HttpDriver(protocol_or_driver)
    return protocol_or_driver
