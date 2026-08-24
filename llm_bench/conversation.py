"""一路线程一次请求。hit：同一条命令原样再发；miss：每次都换一条新命令。"""

from __future__ import annotations

import uuid

from .prompts import (
    CONTEXT_WINDOW,
    DEFAULT_OUTPUT_TOKENS,
    build_hit_user,
    build_miss_user,
    bust_prefix,
    clamp_output_tokens,
    estimate_tokens,
    game_prefix,
    pick_game,
)


class Conversation:
    """一个 worker 在某一波里发出的那一条命令。

    cache=True  冻结 [system, user]，每一波字节完全相同，session 钉死。
    cache=False 每一次 outbound 都换新盐、换一场戏，不带 session。
    """

    def __init__(
        self,
        worker_id: int,
        *,
        system: str,
        user: str,
        cache: bool,
        session_prefix: str = "llm-bench",
        followup: str = "",
        max_input: int = 0,
        max_tokens: int = DEFAULT_OUTPUT_TOKENS,
        context_window: int = CONTEXT_WINDOW,
        pad: bool | None = None,
        full_prefix: str | None = None,
        full_tokens: int | None = None,
        seq: int = 0,
    ):
        self.worker_id = int(worker_id)
        self.cache = bool(cache)
        self.user_template = user or ""
        self.followup = followup or ""
        self.uid = uuid.uuid4().hex[:12]
        self.session_id = (
            f"{session_prefix}-w{self.worker_id:03d}" if self.cache else ""
        )
        self.max_input = max(int(max_input), 0)
        self.max_tokens = max(int(max_tokens), 1)
        self.context_window = max(int(context_window), 1)
        self.seq = max(int(seq), 0)
        self.game = pick_game(self.worker_id)
        self.game_title = self.game["title"]
        should_pad = bool(self.max_input) if pad is None else bool(pad)
        if full_prefix:
            self.full_prefix = full_prefix
        elif should_pad and self.max_input:
            self.full_prefix = game_prefix(
                self.worker_id,
                system or "",
                self.max_input,
                salt=(
                    f"{session_prefix}|{self.game_title}"
                    if self.cache
                    else f"miss-{self.uid}"
                ),
            )
        else:
            self.full_prefix = system or "\n".join(
                part
                for part in (
                    f"游戏《{self.game_title}》",
                    self.game.get("system") or "",
                )
                if part
            )
        self.full_tokens = (
            max(int(full_tokens), 1)
            if full_tokens
            else max(estimate_tokens(self.full_prefix), 1)
        )
        self._frozen: list[dict[str, str]] | None = None
        if self.cache:
            self._frozen = [
                {"role": "system", "content": self.full_prefix},
                {
                    "role": "user",
                    "content": build_hit_user(
                        self.user_template,
                        extra=self.followup,
                        worker_id=self.worker_id,
                    ),
                },
            ]

    def input_tokens_for(self) -> int:
        return max(self.full_tokens, 1)

    def output_tokens_for(self) -> int:
        return clamp_output_tokens(
            self.input_tokens_for(),
            self.max_tokens,
            self.context_window,
        )

    def outbound(self) -> list[dict[str, str]]:
        if self.cache and self._frozen is not None:
            return [
                {"role": item["role"], "content": item["content"]}
                for item in self._frozen
            ]
        self.seq += 1
        return [
            {"role": "system", "content": bust_prefix(self.full_prefix)},
            {
                "role": "user",
                "content": build_miss_user(
                    self.user_template,
                    worker_id=self.worker_id,
                    seq=self.seq,
                    extra=self.followup,
                ),
            },
        ]
