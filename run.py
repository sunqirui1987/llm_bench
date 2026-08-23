#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fire>=0.7,<1",
#   "requests>=2.32,<3",
# ]
# ///
"""通用 LLM Bench：N 路连续对话，测 cache / TTFT / token/s。

端点：
  chat       POST {chat_base_url}/v1/chat/completions
  responses  POST {responses_base_url}/v1/responses
  messages   POST {messages_base_url}/v1/messages

示例：
  uv run run.py bench --cache_mode hit --workers 1 --rounds 10
  uv run run.py bench --cache_mode miss --workers 1 --rounds 10
  uv run run.py cache --rounds 5
"""

import fire

from llm_bench.runner import bench, cache


def main() -> None:
    fire.Fire({"bench": bench, "cache": cache})


if __name__ == "__main__":
    main()
