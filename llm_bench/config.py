"""配置加载、环境变量读取与命令行参数归一化。"""

from __future__ import annotations

import os
from pathlib import Path

from .prompts import DEFAULT_USER, TARGET_INPUT_TOKENS, compose_system


PROJECT_DIR = Path(__file__).resolve().parent.parent


def load_project_env() -> None:
    """读取 run.py 同目录的 .env，但不覆盖 shell 已设置的变量。"""
    env_path = PROJECT_DIR / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] in "\"'" and value[-1:] == value[0]:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


load_project_env()


def env(*names: str, default: str = "") -> str:
    """按优先级读取环境变量，跳过空值。"""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


DEFAULT_API_KEY = env("LLM_API_KEY")
DEFAULT_BASE_URL = env("LLM_BASE_URL", default="https://api.openai.com")
DEFAULT_CHAT_BASE_URL = env("LLM_CHAT_BASE_URL")
DEFAULT_RESPONSES_BASE_URL = env("LLM_RESPONSES_BASE_URL")
DEFAULT_MESSAGES_BASE_URL = env("LLM_MESSAGES_BASE_URL")
DEFAULT_MODELS = env("LLM_MODELS", default="gpt-4o-mini")
DEFAULT_WORKERS = max(int(env("LLM_WORKERS", default="1") or "1"), 1)
DEFAULT_PROMPT = DEFAULT_USER
SYSTEM_PROMPT = compose_system(kind="short")


def parse_list(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        value = ",".join(value)
    return [item.strip() for item in str(value).split(",") if item.strip()]


CACHE_MODE_ALIASES = {
    "miss": "miss",
    "off": "miss",
    "no": "miss",
    "false": "miss",
    "0": "miss",
    "none": "miss",
    "nocache": "miss",
    "no_cache": "miss",
    "hit": "hit",
    "on": "hit",
    "yes": "hit",
    "true": "hit",
    "1": "hit",
    "sticky": "hit",
    "cache": "hit",
}


def parse_cache_mode(value) -> str:
    """hit：本路 session 一直带着；miss：请求不带 session_id。"""
    text = str(value if value is not None else "hit").strip().lower()
    if text not in CACHE_MODE_ALIASES:
        raise ValueError(
            f"未知 cache_mode: {value}；可选 miss（不带 session）或 hit（session 一直）"
        )
    return CACHE_MODE_ALIASES[text]


def resolve_base_urls(
    base_url: str,
    chat_base_url: str,
    responses_base_url: str,
    messages_base_url: str,
) -> dict[str, str]:
    """解析三个协议的独立接入点；未单独配置时回退到公共地址。"""
    common = (base_url or DEFAULT_BASE_URL).rstrip("/")
    return {
        "chat": (chat_base_url or DEFAULT_CHAT_BASE_URL or common).rstrip("/"),
        "responses": (
            responses_base_url or DEFAULT_RESPONSES_BASE_URL or common
        ).rstrip("/"),
        "messages": (
            messages_base_url or DEFAULT_MESSAGES_BASE_URL or common
        ).rstrip("/"),
    }


def ensure_api_key(api_key: str) -> str:
    resolved = api_key or DEFAULT_API_KEY
    if resolved:
        return resolved
    raise RuntimeError(
        "缺少 api_key：请在 .env 配置 LLM_API_KEY，或用 --api_key 传入。"
    )
