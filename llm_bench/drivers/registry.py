"""解析 --via，构造 driver。"""

from __future__ import annotations

from ..protocols.registry import PROTOCOLS
from .cmd import CmdDriver
from .codex import CodexDriver
from .grok import GrokDriver
from .http import HttpDriver

DRIVERS = ("http", "grok", "codex", "cmd")

VIA_ALIASES = {
    "http": "http",
    "api": "http",
    "llm": "http",
    "request": "http",
    "grok": "grok",
    "grok-cli": "grok",
    "codex": "codex",
    "codex-cli": "codex",
    "cmd": "cmd",
    "command": "cmd",
    "program": "cmd",
    "cli": "cmd",
}


def parse_via(value) -> str:
    text = str(value if value is not None else "http").strip().lower()
    if text not in VIA_ALIASES:
        raise ValueError(
            f"未知 --via: {value}；可选 http / grok / codex / cmd"
        )
    return VIA_ALIASES[text]


def resolve_driver(
    via: str,
    *,
    format_name: str = "responses",
    cmd: str = "",
    grok_bin: str = "grok",
    codex_bin: str = "codex",
):
    via = parse_via(via)
    if via == "http":
        if format_name not in PROTOCOLS:
            raise ValueError(
                f"未知格式: {format_name}；可选: {', '.join(PROTOCOLS)}"
            )
        return HttpDriver(PROTOCOLS[format_name])
    if via == "grok":
        return GrokDriver(grok_bin or "grok")
    if via == "codex":
        return CodexDriver(codex_bin or "codex")
    return CmdDriver(cmd)
