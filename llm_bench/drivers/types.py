"""Driver 共用的请求对象。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class StreamRequest:
    model: str
    messages: list[dict]
    max_tokens: int
    timeout: int
    session_id: str = ""
    reasoning_effort: str = ""
    on_progress: Callable | None = None
    allow_empty: bool = False
    base_url: str = ""
    api_key: str = ""


def split_messages(messages: list[dict]) -> tuple[str, str]:
    """把对话拆成 system 前缀和其余正文。"""
    system_parts: list[str] = []
    rest: list[str] = []
    for item in messages or []:
        content = str(item.get("content") or "")
        if (item.get("role") or "user") == "system":
            if content:
                system_parts.append(content)
        elif content:
            rest.append(content)
    return "\n".join(system_parts).strip(), "\n".join(rest).strip()


def combined_prompt(messages: list[dict]) -> str:
    system, user = split_messages(messages)
    if system and user:
        return f"{system}\n\n{user}"
    return system or user
