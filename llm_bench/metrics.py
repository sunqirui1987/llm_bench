"""协议无关的 token 解析与流式延迟指标计算。"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from .prompts import estimate_tokens as _estimate_tokens


def int_field(mapping: dict, *names: str) -> tuple[int, bool]:
    """读取第一个有效整数字段，并区分“缺失”与“明确为 0”。"""
    for name in names:
        if name not in mapping or mapping[name] is None:
            continue
        try:
            return max(int(mapping[name]), 0), True
        except (TypeError, ValueError):
            continue
    return 0, False


def max_int_field(sources: list[dict], *names: str) -> tuple[int, bool]:
    """合并多个 usage 容器，避免外层占位 0 覆盖内层真实值。"""
    values = []
    for source in sources:
        value, present = int_field(source, *names)
        if present:
            values.append(value)
    return (max(values), True) if values else (0, False)


def usage_candidates(event: dict) -> list[dict]:
    """递归查找 usage，兼容网关把 token 字段放在非标准嵌套层级。"""
    token_keys = {
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "cache_read_input_tokens",
        "cached_tokens",
        "prompt_cache_hit_tokens",
    }
    candidates: list[dict] = []
    seen: set[int] = set()

    def visit(value, depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(value, dict):
            if token_keys.intersection(value) and id(value) not in seen:
                candidates.append(value)
                seen.add(id(value))
            for nested in value.values():
                visit(nested, depth + 1)
        elif isinstance(value, list):
            for nested in value:
                visit(nested, depth + 1)

    visit(event)
    return candidates


def cache_is_measurable(input_tokens: int, cache_reported: bool) -> bool:
    # 兼容网关冷未命中时常省略 cached_tokens；只要有总输入就可按 0 计算。
    return cache_reported or input_tokens > 0


@dataclass
class StreamMeasurement:
    """记录流式增量时间点，并在结束时统一计算性能指标。"""
    started_at: float = field(default_factory=time.perf_counter)
    first_at: Optional[float] = None
    last_at: Optional[float] = None
    chunk_times: list[float] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    streamed_parts: list[str] = field(default_factory=list)
    on_progress: Optional[Callable] = field(default=None, repr=False, compare=False)
    _last_progress_at: float = field(default=0.0, repr=False, compare=False)

    def add_delta(self, text: str, *, capture: bool = False) -> None:
        """记录一次文本/推理增量；仅正文增量需要写入最终文本。"""
        if not text:
            return
        self.streamed_parts.append(text)
        if capture:
            self.text_parts.append(text)
        now = time.perf_counter()
        first = self.first_at is None
        if first:
            self.first_at = now
        if self.last_at is not None:
            self.chunk_times.append((now - self.last_at) * 1000)
        self.last_at = now
        self._emit_progress(force=first)

    def live_snapshot(self) -> dict:
        now = time.perf_counter()
        text = "".join(self.streamed_parts) or "".join(self.text_parts)
        out_tokens = max(len(text) // 4, 1) if text else 0
        decode_s = 0.0
        if self.first_at is not None:
            decode_s = max(now - self.first_at, 0.0)
        return {
            "ttft_ms": (
                None if self.first_at is None else (self.first_at - self.started_at) * 1000
            ),
            "elapsed_s": now - self.started_at,
            "chunks": (len(self.chunk_times) + 1) if self.first_at is not None else 0,
            "chars": len(text),
            "out_tokens": out_tokens,
            "tok_s": (out_tokens / decode_s) if decode_s > 0.05 and out_tokens else 0.0,
            "text": text,
        }

    def _emit_progress(self, *, force: bool = False) -> None:
        if self.on_progress is None:
            return
        now = time.perf_counter()
        if not force and now - self._last_progress_at < 0.2:
            return
        self._last_progress_at = now
        self.on_progress(self.live_snapshot())

    def finalize(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        cache_reported: bool,
        usage_reported: bool,
        allow_empty: bool,
        empty_error: str,
    ) -> dict:
        """将采集到的时间点和 usage 归一化为公共结果结构。"""
        ended_at = time.perf_counter()
        if self.first_at is None and not allow_empty:
            raise RuntimeError(empty_error)
        first_at = self.first_at if self.first_at is not None else ended_at
        last_at = self.last_at if self.last_at is not None else first_at
        tpot_ms = None
        output_tps = None
        if output_tokens > 0 and self.last_at is not None:
            decode_s = last_at - first_at
            if output_tokens > 1 and decode_s > 1e-9:
                tpot_ms = decode_s * 1000 / (output_tokens - 1)
                output_tps = (output_tokens - 1) / decode_s
            else:
                span = ended_at - first_at
                if span > 1e-9:
                    output_tps = output_tokens / span
        return {
            "ttft_ms": (first_at - self.started_at) * 1000,
            "tpot_ms": tpot_ms,
            "output_tps": output_tps,
            "e2e_ms": (ended_at - self.started_at) * 1000,
            "cdl_avg": (
                sum(self.chunk_times) / len(self.chunk_times)
                if self.chunk_times else None
            ),
            "cdl_p95": percentile_95(self.chunk_times) if self.chunk_times else None,
            "cdl_max": max(self.chunk_times) if self.chunk_times else None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "cache_reported": cache_reported,
            "usage_reported": usage_reported,
            "text": "".join(self.text_parts),
        }


def percentile_95(values: list[float]) -> float:
    """计算采用 nearest-rank 风格索引的 P95。"""
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
