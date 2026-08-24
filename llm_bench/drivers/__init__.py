"""请求通道：HTTP 协议或 grok/codex/自定义进程。"""

from .registry import DRIVERS, resolve_driver

__all__ = ["DRIVERS", "resolve_driver"]
