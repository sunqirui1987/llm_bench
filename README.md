# LLM Bench

测 **TTFT**、**token/s**、**Prompt Cache**。默认协议是 **responses**（流式）。

没有 step。每一波，每个 worker 只发 **一条命令**。

- `--cache_mode hit`：每个 worker 一款不同的游戏，**同一条命令原样再发**。粘 session。第 1 次冷启，第 2 次起应对上 cache。
- `--cache_mode miss`：仍是每人一款游戏，但每一次换新场面、换新盐，不带 session。

`--rounds`：全量线程把命令再发几遍。hit 时就是「同样的命令再来」；miss 时每一遍都不同。

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
