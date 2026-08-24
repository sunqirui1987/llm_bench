"""每个 worker 一款不同的 Pulse/Lua 游戏模块。hit 时这条实现命令原样再发。"""

from __future__ import annotations

GAMES = (
    {
        "title": "宿命旅途",
        "genre": "竖屏放置卡牌",
        "module": "destiny_journey",
        "system": (
            "实现 Pulse 模块 destiny_journey。"
            "实体 Hero/Stage/Team；页签 roster,log,battle,town,dungeon。"
            "队伍最多 5 人，15 难度层。初始英雄只有 karin 卡琳、maggie 麦琪、linda 琳达。"
            "只输出 Lua，不要文章。"
        ),
        "command": (
            "写出完整 Lua：return Game。"
            "boot 里选区服 thunder、initialHeroId=karin、team={karin}。"
            "tick 里打难度1第1章第1关，battleMode=first_clear，自动战斗逐帧。"
            "通关申报 clear_stage 与切关申报 advance_stage 必须分开；"
            "申报关卡等于 currentStageId；首通只奖一次，写入 clearedStages。"
            "字段：currentStageId,clearedStages,battleMode,roster,team,"
            "initialHeroId,firstLoginTime。只输出 Lua，写到输出上限。"
        ),
        "lore": (
            "-- destiny_journey: layer/chapter/stage. temple_shift. "
            "idle and offline share drop table. server rolls loot."
        ),
        "miss": (
            "实现 frost_server.lua：霜原区选 maggie 打第3层，只输出 Lua",
            "实现 temple_shift.lua：暮港终焉神殿换挡状态机，只输出 Lua",
            "实现 smith_reroll.lua：城镇铁匠铺洗练，只输出 Lua",
        ),
    },
    {
        "title": "潮汐港务",
        "genre": "海港调度模拟",
        "module": "tide_harbor",
        "system": (
            "实现 Pulse 模块 tide_harbor。实体 Berth/Ship/Crane/Pilot。"
            "泊位分 deep/shallow；引水员执照按航道。只输出 Lua，不要文章。"
        ),
        "command": (
            "写出完整 Lua：return Game。"
            "boot 三艘船同时到港，潮位 falling，只有两个深水泊位。"
            "tick 里派泊位、吊机、引水员；误派要写 accident 表。"
            "潮汐表是权威表，禁止口头改。只输出 Lua，写到输出上限。"
        ),
        "lore": "-- tide_harbor: berth depth, pilot license, accident report, tide table is source of truth.",
        "miss": (
            "实现 fog_north.lua：雾天只开北航道调度，只输出 Lua",
            "实现 fish_oil.lua：渔汛和油轮抢泊位，只输出 Lua",
            "实现 crane_wire.lua：吊机钢丝报废停机，只输出 Lua",
        ),
    },
    {
        "title": "星轨快递",
        "genre": "轨道物流",
        "module": "star_rail_post",
        "system": (
            "实现 Pulse 模块 star_rail_post。实体 Parcel/Relay/Window。"
            "轨道窗口 90 秒。只输出 Lua，不要文章。"
        ),
        "command": (
            "写出完整 Lua：return Game。"
            "冷藏舱报警，中继站7号窗口还剩 40 秒。"
            "实现分拣、抛货、地面站按舱单哈希签收。只输出 Lua，写到输出上限。"
        ),
        "lore": "-- star_rail_post: 90s window, cold-hold, manifest hash, opening parcel is accident.",
        "miss": (
            "实现 storm_reroute.lua：太阳风暴改轨，只输出 Lua",
            "实现 hash_collision.lua：两件同哈希撞单，只输出 Lua",
            "实现 door_jam.lua：舱门卡死改用备份臂，只输出 Lua",
        ),
    },
    {
        "title": "夜市烟火",
        "genre": "摊位经营",
        "module": "night_market",
        "system": (
            "实现 Pulse 模块 night_market。实体 Stall/Grid/Inspector。"
            "电箱分区限流。只输出 Lua，不要文章。"
        ),
        "command": (
            "写出完整 Lua：return Game。"
            "电箱跳闸、排队 20 人、巡视 10 分钟后到。"
            "实现备料、改菜单、收摊状态机。只输出 Lua，写到输出上限。"
        ),
        "lore": "-- night_market: grid limit, stall_id, oil temp, neighbor loan ledger.",
        "miss": (
            "实现 rain_cold.lua：暴雨改做冷盘，只输出 Lua",
            "实现 supply_late.lua：供应商迟到，只输出 Lua",
            "实现 stall_taken.lua：摊位号被占，只输出 Lua",
        ),
    },
    {
        "title": "地下一层",
        "genre": "地城探索",
        "module": "dungeon_one",
        "system": (
            "实现 Pulse 模块 dungeon_one。实体 Room/Torch/Party。"
            "房间每晚重组。只输出 Lua，不要文章。"
        ),
        "command": (
            "写出完整 Lua：return Game。"
            "三人小队进门，昨夜地图作废，火把只够 7 个房间。"
            "实现寻路、假记号、撤退。只输出 Lua，写到输出上限。"
        ),
        "lore": "-- dungeon_one: nightly shuffle, torch currency, fake marks, retreat > clear.",
        "miss": (
            "实现 two_torches.lua：只剩两根火把，只输出 Lua",
            "实现 carry_poison.lua：队友中毒要抬，只输出 Lua",
            "实现 hear_breath.lua：听见对面呼吸，只输出 Lua",
        ),
    },
    {
        "title": "田间纪事",
        "genre": "农事日历",
        "module": "field_almanac",
        "system": (
            "实现 Pulse 模块 field_almanac。实体 Plot/Canal/Labor。"
            "节气不可跳。只输出 Lua，不要文章。"
        ),
        "command": (
            "写出完整 Lua：return Game。"
            "谷雨：水渠决口，天黑前补秧，换工只帮半天。"
            "实现田簿 ledger 和分工。只输出 Lua，写到输出上限。"
        ),
        "lore": "-- field_almanac: solar term lock, labor debt, public canal, ledger is only record.",
        "miss": (
            "实现 locust.lua：蝗虫过境，只输出 Lua",
            "实现 dry_well.lua：旱井见底，只输出 Lua",
            "实现 tool_loan.lua：农具被借走没还，只输出 Lua",
        ),
    },
    {
        "title": "法庭速记",
        "genre": "庭审记录",
        "module": "court_steno",
        "system": (
            "实现 Pulse 模块 court_steno。实体 Transcript/Utterance/Appendix。"
            "不能润色证言。只输出 Lua，不要文章。"
        ),
        "command": (
            "写出完整 Lua：return Game。"
            "证人改口，律师要求删除上一句，法官要求原样记录并打时间戳。"
            "删除请求进 appendix，正文不动。只输出 Lua，写到输出上限。"
        ),
        "lore": "-- court_steno: no polish, timestamp recant, delete->appendix, recess bell stops write.",
        "miss": (
            "实现 translator_lag.lua：翻译耳机延迟，只输出 Lua",
            "实现 gallery_noise.lua：旁听席喧哗，只输出 Lua",
            "实现 exhibit_id.lua：物证编号对不上，只输出 Lua",
        ),
    },
    {
        "title": "极地电台",
        "genre": "短波值守",
        "module": "polar_radio",
        "system": (
            "实现 Pulse 模块 polar_radio。实体 Radio/Callsign/Generator。"
            "呼号格式固定。只输出 Lua，不要文章。"
        ),
        "command": (
            "写出完整 Lua：return Game。"
            "暴风雪夜班：呼号断续，发电机油料 4 小时，弱信号求救。"
            "实现值机日志与复述核对。只输出 Lua，写到输出上限。"
        ),
        "lore": "-- polar_radio: callsign grammar, fuel hard cap, noise is not command, repeat-back SOS.",
        "miss": (
            "实现 ice_antenna.lua：天线结冰，只输出 Lua",
            "实现 swollen_cell.lua：备用电池鼓包，只输出 Lua",
            "实现 two_calls.lua：两路呼号重叠，只输出 Lua",
        ),
    },
    {
        "title": "车队夜奔",
        "genre": "长途货运",
        "module": "night_convoy",
        "system": (
            "实现 Pulse 模块 night_convoy。实体 Truck/Convoy/Layby。"
            "编队不许散。只输出 Lua，不要文章。"
        ),
        "command": (
            "写出完整 Lua：return Game。"
            "头车爆胎，后车超车判定，山路只有一个避让点。"
            "实现口令协议和强制休息。只输出 Lua，写到输出上限。"
        ),
        "lore": "-- night_convoy: no scatter, overtake report, one truck per layby, fatigue lock.",
        "miss": (
            "实现 fog_lamp.lua：雾灯坏了，只输出 Lua",
            "实现 bridge_weight.lua：桥梁限重，只输出 Lua",
            "实现 lost_tail.lua：后车失联两分钟，只输出 Lua",
        ),
    },
    {
        "title": "古楼修缮",
        "genre": "古建修复",
        "module": "old_tower",
        "system": (
            "实现 Pulse 模块 old_tower。实体 Beam/Scaffold/WorkOrder。"
            "工序不能颠倒。只输出 Lua，不要文章。"
        ),
        "command": (
            "写出完整 Lua：return Game。"
            "梁上新裂纹，雨季 3 天后到，脚手架未验收。"
            "实现勘察绘图和临时加固。只输出 Lua，写到输出上限。"
        ),
        "lore": "-- old_tower: process order, hidden steel only, no people before sign-off, crack drawing.",
        "miss": (
            "实现 no_tiler.lua：瓦匠没来，只输出 Lua",
            "实现 damp_paint.lua：彩画遇潮，只输出 Lua",
            "实现 settle.lua：基础沉降加剧，只输出 Lua",
        ),
    },
    {
        "title": "深海勘探",
        "genre": "潜器作业",
        "module": "deep_dive",
        "system": (
            "实现 Pulse 模块 deep_dive。实体 Sub/Arm/Tank。"
            "深度锁死。只输出 Lua，不要文章。"
        ),
        "command": (
            "写出完整 Lua：return Game。"
            "预定深度发现未知壳体，氧气 37 分钟，海缆在抖。"
            "实现取样、过热停臂、减压上浮。只输出 Lua，写到输出上限。"
        ),
        "lore": "-- deep_dive: depth lock, O2 authority, arm overheat, decompression stops.",
        "miss": (
            "实现 blackout.lua：能见度骤降，只输出 Lua",
            "实现 tank_leak.lua：取样罐泄漏，只输出 Lua",
            "实现 beacon_drift.lua：定位信标漂移，只输出 Lua",
        ),
    },
    {
        "title": "校园广播",
        "genre": "广播室值班",
        "module": "campus_radio",
        "system": (
            "实现 Pulse 模块 campus_radio。实体 Playlist/Mic/Window。"
            "课间窗口 8 分钟。只输出 Lua，不要文章。"
        ),
        "command": (
            "写出完整 Lua：return Game。"
            "稿件最后一分钟被改，U盘读不出，操场等广播操。"
            "实现过审、备用麦、口播留底。只输出 Lua，写到输出上限。"
        ),
        "lore": "-- campus_radio: 8min window, registered songs, temp speech logged, failover mic.",
        "miss": (
            "实现 battery_cut.lua：停电切蓄电池，只输出 Lua",
            "实现 two_classes.lua：两个班级抢播，只输出 Lua",
            "实现 principal_add.lua：校长加通知，只输出 Lua",
        ),
    },
    {
        "title": "武馆晨课",
        "genre": "馆内日常",
        "module": "dojo_morning",
        "system": (
            "实现 Pulse 模块 dojo_morning。实体 Student/Drill/Injury。"
            "伤停是硬规则。只输出 Lua，不要文章。"
        ),
        "command": (
            "写出完整 Lua：return Game。"
            "新人马步站塌，老学员要散手，先处理膝盖伤。"
            "实现课表、护具检查、禁止夜课。只输出 Lua，写到输出上限。"
        ),
        "lore": "-- dojo_morning: injury stop, spar needs gear, novice no spar, no private night class.",
        "miss": (
            "实现 rain_indoor.lua：雨天改室内，只输出 Lua",
            "实现 gear_short.lua：护具不够，只输出 Lua",
            "实现 fever.lua：有人发烧还来了，只输出 Lua",
        ),
    },
    {
        "title": "荒星温室",
        "genre": "殖民地农学",
        "module": "dome_farm",
        "system": (
            "实现 Pulse 模块 dome_farm。实体 Dome/Filter/Crop。"
            "水循环闭环。只输出 Lua，不要文章。"
        ),
        "command": (
            "写出完整 Lua：return Game。"
            "三号棚湿度掉了，备用滤芯一套，补给 36 小时后。"
            "实现漏气优先、保苗、作物编号对舱单。只输出 Lua，写到输出上限。"
        ),
        "lore": "-- dome_farm: closed loop water, leak first, do not split filter, crop id == manifest.",
        "miss": (
            "实现 sand_glass.lua：沙尘磨花玻璃，只输出 Lua",
            "实现 seed_damp.lua：种子库受潮，只输出 Lua",
            "实现 bee_jam.lua：工蜂机器人卡轨，只输出 Lua",
        ),
    },
    {
        "title": "河灯渡口",
        "genre": "摆渡与民俗",
        "module": "lantern_ferry",
        "system": (
            "实现 Pulse 模块 lantern_ferry。实体 Ferry/Wind/Lantern。"
            "载重线不能超。只输出 Lua，不要文章。"
        ),
        "command": (
            "写出完整 Lua：return Game。"
            "黄昏超载香客要过河放灯，风速在涨，对岸有人接船。"
            "实现载重检查、开灯时辰、劝返记账。只输出 Lua，写到输出上限。"
        ),
        "lore": "-- lantern_ferry: load line, drum hour, night lamp, refusal still logged.",
        "miss": (
            "实现 rope_fray.lua：缆绳磨断，只输出 Lua",
            "实现 overboard.lua：有人落水，只输出 Lua",
            "实现 dark_mark.lua：对岸灯标灭了，只输出 Lua",
        ),
    },
    {
        "title": "机库夜班",
        "genre": "机甲维护",
        "module": "hangar_night",
        "system": (
            "实现 Pulse 模块 hangar_night。实体 Mech/WorkOrder/Bay。"
            "液压和火控互锁。只输出 Lua，不要文章。"
        ),
        "command": (
            "写出完整 Lua：return Game。"
            "三号机液压渗漏，明早出巡，火控舱还在校准。"
            "实现工单隔离、序列号归还、口头交班表。只输出 Lua，写到输出上限。"
        ),
        "lore": "-- hangar_night: open work order blocks launch, hydro/fire lock, serial return, verbal handoff.",
        "miss": (
            "实现 missing_torque.lua：扭力扳手丢了，只输出 Lua",
            "实现 coolant_sku.lua：冷却液标号错，只输出 Lua",
            "实现 nv_rollback.lua：夜视仪固件回滚，只输出 Lua",
        ),
    },
)


def pick_game(worker_id: int) -> dict:
    """work1 起按目录轮转；超过 16 路会循环复用同一款游戏。"""
    return GAMES[int(worker_id) % len(GAMES)]
