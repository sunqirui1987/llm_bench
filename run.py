#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fire>=0.7,<1",
#   "requests>=2.32,<3",
# ]
# ///
"""LLM Bench：N 路同时发命令。hit=同一条命令再来；miss=每次都换新命令。

示例：
  uv run run.py bench --cache_mode hit --workers 8 --rounds 2
  uv run run.py bench --cache_mode miss --workers 8 --rounds 2
  uv run run.py cache --rounds 2
"""

import fire

from llm_bench.runner import bench, cache


def main() -> None:
    fire.Fire({"bench": bench, "cache": cache})


if __name__ == "__main__":
    main()
