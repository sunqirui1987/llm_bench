# LLM Bench

测 **TTFT**、**token/s**、**Prompt Cache**。默认协议是 **responses**（流式）。

没有 step。所有 worker **同时开工**，各路自己连发，互不等待。

压测任务是给 **Pulse**（Lua 5.4 游戏宿主）写模块：**只输出 Lua 代码**，不写故事。

- `--cache_mode hit`：每个 worker 一款不同的游戏模块，**同一条实现命令原样再发**。粘 session。每路第一次成功是冷启动，不计命中率。
- `--cache_mode miss`：仍是每人一款游戏，但每一次换一个 Lua 子系统、换新盐，不带 session。界面不标预热/缓存。

`--rounds`：每路自己连发几遍。hit 时就是「同样的命令再来」；miss 时每一遍都不同。某路先跑完就先进入下一次，不等别人。

内置 16 个 Pulse 模块（`llm_bench/games.py`），宿主 API 在 `llm_bench/data/pulse_framework.lua`。`--workers` 超过 16 会循环复用。`--prompt` / `--prompt_file` 会覆盖**所有** work 的用户命令。`--system_prompt` / `--system_file` 叠在每路模块设定上，所有 work 共用。

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
uv run run.py bench --cache_mode miss --workers 8 --rounds 2 --effort xhigh
uv run run.py bench --via grok --effort xhigh --workers 4
uv run run.py bench --via cmd --cmd 'my-llm --model {model} --file {prompt_file}'
uv run run.py cache --rounds 2
```

- `--workers`：同时开几路线程，互不等待
- `--rounds`：每路自己连发几遍（默认 2：一次冷、一次热）
- `--input_tokens`：这条命令的输入大小（默认约 30 万，用来把 TPM 打高）
- `--max_tokens`：输出上限（默认 2048，短输出才结束得快）
- `--effort`：推理强度（默认 `high`）。`low` / `medium` / `high` / `xhigh`，和 Grok Build、sub2api 一样。`max`/`ultra` 当作 `xhigh`。空或 `none` 则不传，走网关默认。
- `--via`：发请求的通道。`http`（默认，直接打 LLM API）/ `grok` / `codex` / `cmd`。CLI 默认关工具、1 轮，接近补全而不是 agent。
- `--cmd`：`--via cmd` 时的程序模板。长 prompt 用 `{prompt_file}`，不要把正文塞进 argv。其它占位：`{model}` `{effort}` `{max_tokens}` `{session_id}`。

控制台每次刷新会**清整屏**，再画出启动横幅和每个 work 一栏。正文写在该 work 下面。全文写入 `logs/` 和 `report.md`。
