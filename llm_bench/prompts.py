"""通用压测语料：短延迟、长前缀缓存、自定义对话、多轮追加。"""

from __future__ import annotations

import time
import uuid
from pathlib import Path


TARGET_INPUT_TOKENS = 8_192
HISTORY_TOKEN_BUDGET = 420_000
ASSISTANT_KEEP_CHARS = 12_000
UNIQUE_PREFIX_LINES = 16

DEFAULT_SYSTEM = (
    "You are a precise technical assistant. Answer completely. "
    "Do not ask whether to continue. Write until you finish or hit the output limit."
)

DEFAULT_USER = (
    "Explain a production-grade design for an LLM inference gateway: "
    "routing, streaming, prompt cache, retries, backpressure, and observability. "
    "Include concrete interfaces, failure cases, and tests. Write until the output limit."
)

DEFAULT_FOLLOWUP = (
    "Continue from where you left off. Do not restart or repeat previous text. "
    "Add the next missing section until the output limit."
)

_PAD_BLOCK = (
    "This is stable context padding for prompt-cache and TTFT tests. "
    "It stays identical across turns of the same conversation so a prefix cache can hit. "
    "block={i:06d} salt={salt}\n"
)


def estimate_tokens(text: str) -> int:
    """粗估 token：汉字略大于 1，其它按 4 字符 1 token。"""
    if not text:
        return 0
    cjk = 0
    other = 0
    for char in text:
        code = ord(char)
        if (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0xF900 <= code <= 0xFAFF
            or 0x3000 <= code <= 0x303F
            or 0xFF00 <= code <= 0xFFEF
        ):
            cjk += 1
        elif char.isspace():
            continue
        else:
            other += 1
    return int(cjk * 1.15 + other / 4)


def read_text_file(path: str, *, label: str = "file") -> str:
    """读取 UTF-8 文本；路径为空则返回空字符串。"""
    if not path:
        return ""
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} 不存在: {path}")
    return candidate.read_text(encoding="utf-8")


def load_context(path: str = "") -> str:
    """只在显式传入 --context_file 时读取；不再偷偷加载内置游戏语料。"""
    text = read_text_file(path, label="context_file")
    return text.strip() + "\n" if text.strip() else ""


def unique_salt() -> str:
    return f"{uuid.uuid4().hex}-{time.time_ns()}"


def bust_prefix(body: str, lines: int = UNIQUE_PREFIX_LINES) -> str:
    """在文本最前面插入每轮都不同的盐，从 token 0 打断 prefix cache。

    只靠不带 session_id 不够：多数上游仍按 prompt 前缀缓存。
    """
    salt = unique_salt()
    count = max(int(lines), 1)
    header = "\n".join(
        f"CACHE_BYPASS salt={salt} i={index:04d} rev={salt[::-1]} ns={time.time_ns()}"
        for index in range(count)
    )
    return f"{header}\n{body}\nEND_CACHE_BYPASS {salt}"


def pad_to_tokens(text: str, target: int, *, salt: str = "stable") -> str:
    """把文本填充到目标 token。同一 salt 得到同一前缀，便于缓存命中。"""
    target = max(int(target), 0)
    body = text or ""
    if estimate_tokens(body) >= target:
        return body
    parts = [body.rstrip(), "\n===== CONTEXT PADDING =====\n"]
    current = estimate_tokens("".join(parts))
    sample = _PAD_BLOCK.format(i=0, salt=salt)
    block_tokens = max(estimate_tokens(sample), 1)
    index = 0
    while current < target:
        parts.append(_PAD_BLOCK.format(i=index, salt=salt))
        current += block_tokens
        index += 1
        if index > 1_000_000:
            break
    return "".join(parts)


def compose_system(
    *,
    kind: str = "short",
    text: str = "",
    file: str = "",
    context_file: str = "",
    input_tokens: int = TARGET_INPUT_TOKENS,
    salt: str = "stable",
) -> str:
    """拼出本轮 system。

    - text / file：自定义系统提示，file 在前、text 在后
    - context_file：额外长上下文，叠在最前面
    - kind=long：再填充到 input_tokens，用来测 Prompt Cache
    - kind=short：不填充
    """
    body = (text or "").strip()
    from_file = read_text_file(file, label="system_file").strip()
    if from_file:
        body = f"{from_file}\n{body}".strip() if body else from_file
    if not body:
        body = DEFAULT_SYSTEM
    context = load_context(context_file)
    if context:
        body = context + body
    if (kind or "short").strip().lower() == "long":
        return pad_to_tokens(body, input_tokens, salt=salt)
    return body


def compose_user(prompt: str = "", prompt_file: str = "") -> str:
    body = (prompt or "").strip()
    from_file = read_text_file(prompt_file, label="prompt_file").strip()
    if from_file:
        body = f"{from_file}\n{body}".strip() if body else from_file
    return body or DEFAULT_USER


def build_user_prompt(template: str, *, worker_id: int, uid: str) -> str:
    body = (template or DEFAULT_USER).strip() or DEFAULT_USER
    return f"{body}\n[conversation w{worker_id:03d} {uid}]"


def build_followup_prompt(turn: int, template: str = "") -> str:
    body = (template or DEFAULT_FOLLOWUP).strip() or DEFAULT_FOLLOWUP
    return f"CONTINUE_TURN {turn}\n{body}"


def clip_text(text: str, limit: int = ASSISTANT_KEEP_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    keep = max(limit // 2, 256)
    return text[:keep] + "\n…(truncated so this conversation can continue)…\n" + text[-keep:]


def estimate_messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(str(item.get("content") or "")) + 8 for item in messages)


def trim_messages(messages: list[dict], budget: int = HISTORY_TOKEN_BUDGET) -> list[dict]:
    """保住 system 和当前待发送的最后一条 user，丢掉最早的已完成回合。"""
    while len(messages) > 3 and estimate_messages_tokens(messages) > budget:
        if messages[1].get("role") == "user" and messages[2].get("role") == "assistant":
            del messages[1:3]
        else:
            del messages[1]
    return messages
