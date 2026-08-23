"""压测语料：5 步把输入抬到上限，用户任务按 worker/步骤换题。"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from .games import GAMES, pick_game


CONTEXT_WINDOW = 500_000
TOKEN_OVERHEAD = 2_048
DEFAULT_OUTPUT_RESERVE = 65_536
TARGET_INPUT_TOKENS = CONTEXT_WINDOW - DEFAULT_OUTPUT_RESERVE - TOKEN_OVERHEAD
HISTORY_TOKEN_BUDGET = 420_000
ASSISTANT_KEEP_CHARS = 12_000
UNIQUE_PREFIX_LINES = 16

DEFAULT_SYSTEM = (
    "你是《宿命旅途》的场景主持人。只用这份档案里的专属设定："
    "竖屏放置小队卡牌 RPG，队伍最多五人，十五个难度层，每层 23 章、每章 5 关，"
    "初始伙伴只有卡琳、麦琪、琳达。把当前场景写完整，写到输出上限，不要问是否继续。"
)

DEFAULT_USER = (
    "请按系统里的《宿命旅途》档案，把下面这场戏从进界面写到首通结算结束。"
    "不要改设定，不要发明档案里没有的英雄、页签或建筑，不要问是否继续，写到输出上限。\n"
    "\n"
    "1. 开始界面，选择区服「雷鸣区」，记下 firstLoginTime。\n"
    "2. 看完开场情景对话。\n"
    "3. 从卡琳、麦琪、琳达里选定卡琳，initialHeroId 写入该区服档案，编入 team。\n"
    "4. 切到战斗页签，进入难度层 1、第 1 章、第 1 关，battleMode 为首通，客户端做自动战斗逐帧演算。\n"
    "5. 通关申报和切关申报必须分开。申报关卡等于 currentStageId。"
    "首通只奖一次，写入 clearedStages。\n"
    "过程里要用到的字段：currentStageId、clearedStages、battleMode、roster、team、"
    "initialHeroId、firstLoginTime。"
)

DEFAULT_FOLLOWUP = (
    "从刚才停下的地方继续写这一场，不要重开，不要重复。写到输出上限。"
)

WORKER_DOMAINS = tuple(game["title"] for game in GAMES)

_PAD_BLOCKS = (
    (
        "【{domain} · {genre} · 卷 {i:06d}】\n"
        "salt={salt}\n"
        "{lore} 本条编号 {i:06d}。不要把别的游戏规则写进来。"
        "layer={layer} chapter={chapter} stage={stage}。\n"
    ),
    (
        "【{domain} 现场 {i:06d}】\n"
        "salt={salt}\n"
        "{lore} 发生在条目 {i:06d}。时间、工具、限制都按本游戏。\n"
    ),
    (
        "【{domain} 禁则 {i:06d}】\n"
        "salt={salt}\n"
        "{lore} 违反条目 {i:06d} 必须停手并记录。\n"
    ),
    (
        "【{domain} 交接 {i:06d}】\n"
        "salt={salt}\n"
        "{lore} 交班只口头复述本游戏术语。编号 {i:06d}。\n"
    ),
)


def estimate_tokens(text: str) -> int:
    """粗估 token：汉字略大于 1，其它按 4 字符 1 token。"""
    if not text:
        return 0
    cjk = 0
    other = 0
    for char in text:
        code = ord(char)
        if (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0xF900 <= code <= 0xFAFF
            or 0x3000 <= code <= 0x303F
            or 0xFF00 <= code <= 0xFFEF
        ):
            cjk += 1
        elif char.isspace():
            continue
        else:
            other += 1
    return int(cjk * 1.15 + other / 4)


def read_text_file(path: str, *, label: str = "file") -> str:
    """读取 UTF-8 文本；路径为空则返回空字符串。"""
    if not path:
        return ""
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} 不存在: {path}")
    return candidate.read_text(encoding="utf-8")


def load_context(path: str = "") -> str:
    """只在显式传入 --context_file 时读取；不再偷偷加载内置游戏语料。"""
    text = read_text_file(path, label="context_file")
    return text.strip() + "\n" if text.strip() else ""


def unique_salt() -> str:
    return f"{uuid.uuid4().hex}-{time.time_ns()}"


def bust_prefix(body: str, lines: int = UNIQUE_PREFIX_LINES) -> str:
    """在文本最前面插入每轮都不同的盐，从 token 0 打断 prefix cache。"""
    salt = unique_salt()
    count = max(int(lines), 1)
    header = "\n".join(
        f"CACHE_BYPASS salt={salt} i={index:04d} rev={salt[::-1]} ns={time.time_ns()}"
        for index in range(count)
    )
    return f"{header}\n{body}\nEND_CACHE_BYPASS {salt}"


def game_bible() -> str:
    path = Path(__file__).parent / "data" / "destiny_journey_context.md"
    if path.is_file():
        return path.read_text(encoding="utf-8").strip() + "\n"
    return ""


def worker_domain(worker_id: int) -> str:
    return pick_game(worker_id)["title"]


def _window_overhead(context_window: int) -> int:
    window = max(int(context_window), 1)
    return min(TOKEN_OVERHEAD, max(window // 20, 1))


def clamp_output_tokens(input_tokens: int, max_tokens: int, context_window: int) -> int:
    """输出顶满剩余窗口，但不超过 --max_tokens。"""
    room = max(int(context_window) - int(input_tokens) - _window_overhead(context_window), 1)
    return max(1, min(int(max_tokens), room))


def fit_max_input(max_input: int, max_tokens: int, context_window: int) -> int:
    """给输出留位置；输入上限不超过窗口。"""
    window = max(int(context_window), 1)
    min_out = max(min(int(max_tokens), DEFAULT_OUTPUT_RESERVE, max(window // 5, 1)), 1)
    cap = max(window - _window_overhead(window) - min_out, 1)
    return max(1, min(int(max_input), cap))


def plan_request(
    *,
    max_input: int,
    max_tokens: int,
    context_window: int,
) -> dict[str, int]:
    inp = fit_max_input(max_input, max_tokens, context_window)
    out = clamp_output_tokens(inp, max_tokens, context_window)
    return {"input_tokens": inp, "max_tokens": out}


def slice_to_tokens(text: str, target: int) -> str:
    """按字符比例切前缀，保证 step k 是 step k+1 的字节前缀。"""
    if not text:
        return ""
    target = max(int(target), 1)
    total = estimate_tokens(text)
    if total <= target:
        return text
    n = max(1, int(len(text) * target / total))
    return text[:n]


def pad_to_tokens(
    text: str,
    target: int,
    *,
    salt: str = "stable",
    domain: str = "",
    genre: str = "",
    lore: str = "",
) -> str:
    """把文本填充到目标 token。同一 salt 得到同一前缀，便于缓存命中。"""
    target = max(int(target), 0)
    body = text or ""
    if estimate_tokens(body) >= target:
        return body
    topic = domain or "未命名游戏"
    flavor = lore or topic
    kind = genre or "游戏"
    if topic == "宿命旅途":
        bible = game_bible()
        if bible and "宿命旅途" not in body:
            body = bible + body
            if estimate_tokens(body) >= target:
                return body
    parts = [body.rstrip(), "\n===== CONTEXT PADDING =====\n"]
    current = estimate_tokens("".join(parts))

    def render(index: int) -> str:
        template = _PAD_BLOCKS[index % len(_PAD_BLOCKS)]
        return template.format(
            i=index,
            salt=salt,
            domain=topic,
            genre=kind,
            lore=flavor,
            layer=(index % 15) + 1,
            chapter=(index % 23) + 1,
            stage=(index % 5) + 1,
        )

    block_tokens = [
        max(estimate_tokens(render(slot)), 1) for slot in range(len(_PAD_BLOCKS))
    ]
    index = 0
    while current < target:
        slot = index % len(_PAD_BLOCKS)
        parts.append(render(index))
        current += block_tokens[slot]
        index += 1
        if index > 1_000_000:
            break
    return "".join(parts)


def compose_system(
    *,
    kind: str = "short",
    text: str = "",
    file: str = "",
    context_file: str = "",
    input_tokens: int = TARGET_INPUT_TOKENS,
    salt: str = "stable",
) -> str:
    """拼出系统提示。kind=long 时填到 input_tokens。"""
    body = (text or "").strip()
    from_file = read_text_file(file, label="system_file").strip()
    if from_file:
        body = f"{from_file}\n{body}".strip() if body else from_file
    if not body:
        body = DEFAULT_SYSTEM
    context = load_context(context_file)
    if context:
        body = context + body
    if (kind or "short").strip().lower() == "long":
        return pad_to_tokens(body, input_tokens, salt=salt, domain="宿命旅途", genre="竖屏放置卡牌")
    return body


def compose_user(prompt: str = "", prompt_file: str = "") -> str:
    body = (prompt or "").strip()
    from_file = read_text_file(prompt_file, label="prompt_file").strip()
    if from_file:
        body = f"{from_file}\n{body}".strip() if body else from_file
    return body or DEFAULT_USER


def game_prefix(worker_id: int, base_system: str, max_input: int, salt: str) -> str:
    """按 worker 生成该游戏自己的长前缀，不和其他 work 共用。"""
    game = pick_game(worker_id)
    body = "\n".join(part for part in (game["system"], (base_system or "").strip()) if part)
    if game["title"] == "宿命旅途":
        bible = game_bible()
        if bible:
            body = bible + body
    return pad_to_tokens(
        body,
        max_input,
        salt=salt,
        domain=game["title"],
        genre=game["genre"],
        lore=game["lore"],
    )


def build_hit_user(template: str = "", extra: str = "", worker_id: int = 0) -> str:
    """命中缓存：这个 worker 自己那款游戏的命令，每一波字节都一样。"""
    custom = (template or "").strip()
    if custom and custom != DEFAULT_USER:
        body = custom
    else:
        body = pick_game(worker_id)["command"]
    extra = (extra or "").strip()
    if extra:
        return f"{body}\n{extra}"
    return body


def build_miss_user(
    template: str = "",
    *,
    worker_id: int = 0,
    seq: int = 1,
    extra: str = "",
) -> str:
    """不走缓存：仍是这个 worker 的游戏，但每次换一场新命令。"""
    game = pick_game(worker_id)
    beats = game["miss"]
    beat = beats[(int(seq) - 1) % len(beats)]
    salt = unique_salt()
    parts = [
        f"本场命令编号 {salt}",
        f"游戏《{game['title']}》（{game['genre']}）。",
        f"这一场要演：{beat}。",
        game["system"],
        "不要问是否继续，写到输出上限。不要写成别的游戏。",
        f"seq={seq} worker={worker_id:03d}",
    ]
    custom = (template or "").strip()
    if custom and custom != DEFAULT_USER:
        parts.append(custom)
    extra = (extra or "").strip()
    if extra:
        parts.append(extra)
    return "\n".join(parts)


def build_followup_prompt(turn: int, template: str = "") -> str:
    body = (template or DEFAULT_FOLLOWUP).strip() or DEFAULT_FOLLOWUP
    return f"CONTINUE_TURN {turn}\n{body}"


def clip_text(text: str, limit: int = ASSISTANT_KEEP_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    keep = max(limit // 2, 256)
    return text[:keep] + "\n…(truncated so this conversation can continue)…\n" + text[-keep:]


def estimate_messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(str(item.get("content") or "")) + 8 for item in messages)


def trim_messages(messages: list[dict], budget: int = HISTORY_TOKEN_BUDGET) -> list[dict]:
    """保住 system 和当前待发送的最后一条 user，丢掉最早的已完成回合。"""
    while len(messages) > 3 and estimate_messages_tokens(messages) > budget:
        if messages[1].get("role") == "user" and messages[2].get("role") == "assistant":
            del messages[1:3]
        else:
            del messages[1]
    return messages
