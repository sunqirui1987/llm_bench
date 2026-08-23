"""每个压测线程对应一路普通多轮对话。"""

from __future__ import annotations

import uuid

from .prompts import (
    build_followup_prompt,
    build_user_prompt,
    bust_prefix,
    clip_text,
    trim_messages,
)


class Conversation:
    """一路连续对话：成功后把回复和下一条用户消息追加进去。

    cache=True  钉死 session_id，system 前缀保持不变，让 Prompt Cache 能命中。
    cache=False 不带 session，每次 outbound 都在 system 最前面换新盐，强制不命中。
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
    ):
        self.worker_id = int(worker_id)
        self.cache = bool(cache)
        self.followup = followup or ""
        self.uid = uuid.uuid4().hex[:12]
        self.session_id = (
            f"{session_prefix}-w{self.worker_id:03d}-{self.uid}" if self.cache else ""
        )
        self.turn = 0
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": build_user_prompt(user, worker_id=self.worker_id, uid=self.uid),
            },
        ]

    def outbound(self) -> list[dict[str, str]]:
        messages = [
            {"role": item["role"], "content": item["content"]} for item in self.messages
        ]
        if self.cache or not messages:
            return messages
        if messages[0]["role"] == "system":
            messages[0]["content"] = bust_prefix(messages[0]["content"])
        else:
            messages.insert(0, {"role": "system", "content": bust_prefix("")})
        return messages

    def commit(self, assistant_text: str) -> None:
        self.messages.append(
            {
                "role": "assistant",
                "content": clip_text(assistant_text) or "(empty reply; continue.)",
            }
        )
        self.turn += 1
        self.messages.append(
            {
                "role": "user",
                "content": build_followup_prompt(self.turn, self.followup),
            }
        )
        trim_messages(self.messages)
