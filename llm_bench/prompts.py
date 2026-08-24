"""压测语料：按 worker 选游戏。默认大输入、短输出，用来把 TPM 拉高。"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from .games import GAMES, pick_game


CONTEXT_WINDOW = 500_000
TOKEN_OVERHEAD = 2_048
DEFAULT_OUTPUT_TOKENS = 2_048
DEFAULT_OUTPUT_RESERVE = DEFAULT_OUTPUT_TOKENS
# 网关当前消息预算实测 475424。Lua 注释填充时粗估 41 万会涨到 54 万。
GATEWAY_INPUT_BUDGET = 475_424
MAX_PADDED_INPUT = 300_000
# 默认垫满输入：一次请求计入大量 input token，短输出很快结束，TPM 才高。
DEFAULT_PADDED_INPUT = MAX_PADDED_INPUT
INPUT_FRAME_RESERVE = 2_048
UNIQUE_PREFIX_LINES = 16


def _window_overhead(context_window: int) -> int:
    """给网关采样预算留空。500k 窗大约要 5%（实测预算 475424）。"""
    window = max(int(context_window), 1)
    percent = max(window // 20, 1)
    floor = min(TOKEN_OVERHEAD, max(window // 4, 1))
    return max(percent, floor)


TARGET_INPUT_TOKENS = min(
    DEFAULT_PADDED_INPUT,
    CONTEXT_WINDOW - DEFAULT_OUTPUT_RESERVE - _window_overhead(CONTEXT_WINDOW),
    MAX_PADDED_INPUT,
)

DEFAULT_SYSTEM = (
    "You are a Pulse Lua 5.4 code generator. Output ONLY Lua source. "
    "No markdown, no fences, no story prose. Implement a compact Game module "
    "against pulse.* host APIs. Stop as soon as boot/tick/serialize work."
)

OUTPUT_FILL = (
    "Keep it compact. Only boot, tick, on_input, serialize, and return Game. "
    "No extra tables, no test_*, no padding. Stop at the output cap."
)

DEFAULT_USER = GAMES[0]["command"]

_PAD_BLOCKS = (
    (
        "-- [{domain} :: {genre} :: chunk {i:06d}]\n"
        "-- salt={salt}\n"
        "-- {lore}\n"
        "-- pulse.world layer={layer} chapter={chapter} stage={stage}\n"
        "-- do not mix other game modules into this file\n"
    ),
    (
        "-- [{domain} api {i:06d}] salt={salt}\n"
        "-- {lore}\n"
        "-- local function _f{i:06d}(world) assert(world) end\n"
    ),
    (
        "-- [{domain} guard {i:06d}] salt={salt}\n"
        "-- {lore}\n"
        "-- if false then pulse.log.error('cross-game {i:06d}') end\n"
    ),
    (
        "-- [{domain} handoff {i:06d}] salt={salt}\n"
        "-- {lore}\n"
        "-- -- verbal handoff uses this module's identifiers only #{i:06d}\n"
    ),
)


def estimate_tokens(text: str) -> int:
    """粗估 token。宁多勿少：Grok 对这份中文填充大约比 1.15 贵 11%，低估会 413。"""
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
    # Lua 注释/标识符比普通英文更碎，/3.5 会低估。
    return max(int(cjk * 1.4 + other / 2.5), 1)


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


def pulse_spec() -> str:
    path = Path(__file__).parent / "data" / "pulse_framework.lua"
    if path.is_file():
        return path.read_text(encoding="utf-8").strip() + "\n"
    return ""


def clamp_output_tokens(input_tokens: int, max_tokens: int, context_window: int) -> int:
    """输出顶满剩余窗口，但不超过 --max_tokens。"""
    room = max(int(context_window) - int(input_tokens) - _window_overhead(context_window), 1)
    return max(1, min(int(max_tokens), room))


def fit_max_input(max_input: int, max_tokens: int, context_window: int) -> int:
    """给输出留位置；输入上限不超过窗口，也不超过网关消息预算。"""
    window = max(int(context_window), 1)
    min_out = max(min(int(max_tokens), DEFAULT_OUTPUT_RESERVE, max(window // 5, 1)), 1)
    cap = max(window - _window_overhead(window) - min_out, 1)
    cap = min(cap, MAX_PADDED_INPUT, GATEWAY_INPUT_BUDGET)
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
    parts = [body.rstrip(), "\n-- ===== CONTEXT PADDING =====\n"]
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
    while True:
        slot = index % len(_PAD_BLOCKS)
        nxt = block_tokens[slot]
        if current >= target or current + nxt > target:
            break
        parts.append(render(index))
        current += nxt
        index += 1
        if index > 1_000_000:
            break
    text = "".join(parts)
    while len(parts) > 2 and estimate_tokens(text) > target:
        parts.pop()
        text = "".join(parts)
    return text


def compose_system(
    *,
    text: str = "",
    file: str = "",
    context_file: str = "",
) -> str:
    """拼出自定义系统提示。未传则空，避免把默认游戏塞进每一路。"""
    body = (text or "").strip()
    from_file = read_text_file(file, label="system_file").strip()
    if from_file:
        body = f"{from_file}\n{body}".strip() if body else from_file
    context = load_context(context_file)
    if context:
        body = context + body
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
    body = "\n".join(
        part
        for part in (
            DEFAULT_SYSTEM,
            f"游戏《{game['title']}》模块 {game.get('module') or ''}",
            game["system"],
            (base_system or "").strip(),
        )
        if part
    )
    spec = pulse_spec()
    if spec:
        body = spec + body
    target = max(int(max_input) - INPUT_FRAME_RESERVE, 1)
    return pad_to_tokens(
        body,
        target,
        salt=salt,
        domain=game["title"],
        genre=game["genre"],
        lore=game["lore"],
    )


def build_hit_user(template: str = "", extra: str = "", worker_id: int = 0) -> str:
    """命中缓存：这个 worker 自己那款游戏的命令，每一波字节都一样。"""
    custom = (template or "").strip()
    catalog = not custom or custom == DEFAULT_USER
    body = pick_game(worker_id)["command"] if catalog else custom
    extra = (extra or "").strip()
    if extra:
        body = f"{body}\n{extra}"
    if catalog:
        body = f"{body}\n{OUTPUT_FILL}"
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
        "只输出 Lua，不要文章，不要问是否继续。模块写完就停。不要写成别的游戏。",
        OUTPUT_FILL,
        f"seq={seq} worker={worker_id:03d}",
    ]
    custom = (template or "").strip()
    if custom and custom != DEFAULT_USER:
        parts.append(custom)
    extra = (extra or "").strip()
    if extra:
        parts.append(extra)
    return "\n".join(parts)
