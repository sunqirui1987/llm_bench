# LLM Bench

测 **TTFT**、**token/s**、**Prompt Cache**。默认协议是 **responses**（流式）。

没有 step。每一波，每个 worker 只发 **一条命令**。

- `--cache_mode hit`：每个 worker 一款不同的游戏，**同一条命令原样再发**。粘 session。每路第一次成功是冷启动，不计命中率。
- `--cache_mode miss`：仍是每人一款游戏，但每一次换新场面、换新盐，不带 session。界面不标预热/缓存。

`--rounds`：全量线程把命令再发几遍。hit 时就是「同样的命令再来」；miss 时每一遍都不同。

内置 16 款游戏（`llm_bench/games.py`）。`--workers` 超过 16 会循环复用。`--prompt` / `--prompt_file` 会覆盖**所有** work 的用户命令，不再按游戏开场。`--system_prompt` / `--system_file` 叠在每路游戏设定上，所有 work 共用。

## 配置

```bash
cp .env.example .env
```

```dotenv
LLM_API_KEY=sk-xxxx
LLM_BASE_URL=http://127.0.0.1:8080
LLM_MODELS=grok-4.6
```

## 命令

```bash
uv run run.py bench --cache_mode hit --workers 8 --rounds 2
uv run run.py bench --cache_mode miss --workers 8 --rounds 2
uv run run.py cache --rounds 2
```

- `--workers`：一波同时开几路线程
- `--rounds`：同一批线程把命令再发几遍（默认 2：一次冷、一次热）
- `--input_tokens`：这条命令的输入大小（默认约 43 万）
- `--max_tokens`：输出上限（默认 50 万，按窗口剩余截断）

控制台每次刷新会**清整屏**，再画出启动横幅和每个 work 一栏。正文写在该 work 下面。全文写入 `logs/` 和 `report.md`。
