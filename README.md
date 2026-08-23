# LLM Bench

测 **TTFT**、**token/s**、**Prompt Cache**。默认协议是 **responses**（流式，当前轮会实时刷新）。

`--cache_mode hit`：粘 session，前缀固定。`miss`：不带 session，每轮打散前缀。

## 配置

```bash
cp .env.example .env
```

```dotenv
LLM_API_KEY=sk-xxxx
LLM_BASE_URL=http://127.0.0.1:8080
LLM_MODELS=grok-4.6
```

## 协议

| `--formats` | 路径 |
|-------------|------|
| `responses`（默认） | `/v1/responses` |
| `chat` | `/v1/chat/completions` |
| `messages` | `/v1/messages` |

三个地址可分开：`LLM_CHAT_BASE_URL` / `LLM_RESPONSES_BASE_URL` / `LLM_MESSAGES_BASE_URL`。

## 命令

```bash
uv run run.py bench --cache_mode hit --workers 1 --rounds 10
uv run run.py bench --cache_mode miss --workers 1 --rounds 10
uv run run.py bench --formats chat,responses,messages --workers 2 --rounds 5
uv run run.py cache --rounds 5
```

- `--workers`：同时开几路**对话**（并发）
- `--rounds`：每一路要问答多少轮（输入 → 输出 → 再输入 → 再输出）

屏幕像 `top`：清屏刷新。每个 work 下面只显示**当前这一轮**。跑完写入 `report.md`（按 work 列出每一轮）。

## 常用参数

| 参数 | 默认 |
|------|------|
| `--formats` | `responses` |
| `--cache_mode` | `hit` |
| `--workers` | `1` |
| `--max_tokens` | `500000` |
| `--system` | `long`（约 8k 前缀，测缓存） |
| `--models` | `LLM_MODELS` |
| `--base_url` | `LLM_BASE_URL` |

```bash
uv run run.py bench --help
```
