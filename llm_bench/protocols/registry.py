"""协议适配器注册表；runner 只依赖这里，不直接依赖具体协议目录。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import chat, messages, responses


@dataclass(frozen=True)
class Protocol:
    name: str
    endpoint: str
    stream: Callable


PROTOCOLS = {
    "chat": Protocol(chat.NAME, chat.ENDPOINT, chat.stream),
    "responses": Protocol(responses.NAME, responses.ENDPOINT, responses.stream),
    "messages": Protocol(messages.NAME, messages.ENDPOINT, messages.stream),
}
