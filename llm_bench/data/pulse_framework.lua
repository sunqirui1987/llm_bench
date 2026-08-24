-- Pulse 0.4 · Lua 5.4 game host
-- One process loads exactly one Game module. Host owns the loop; Game never
-- prints, never blocks, never talks to the player in natural language.
-- Output of a bench run MUST be Lua source only.

--[[
Game module contract (return this table):

  Game.id            string   stable module id, snake_case
  Game.title         string
  Game.boot(world)            called once; seed entities, bind input
  Game.tick(world, dt)        dt in seconds, fixed 1/20
  Game.on_input(world, ev)    ev = {type=string, ...}
  Game.serialize(world)       return a pure table for pulse.save

Host globals (read-only unless documented):

  pulse.time.now() -> number
  pulse.time.tick  integer
  pulse.rng.seed(n)
  pulse.rng.int(a, b)  pulse.rng.float()
  pulse.log.debug / info / warn / error (fmt, ...)
  pulse.bus.emit(topic, payload)  pulse.bus.on(topic, fn)
  pulse.save.write(name, tbl)     pulse.save.read(name) -> tbl|nil
  pulse.input.queue               FIFO of ev
  pulse.world                     the live World

World:

  world:spawn(kind, data) -> id
  world:get(id) -> entity|nil
  world:kill(id)
  world:query(kind) -> iterator
  world:set(id, key, value)

Entity fields used by all games: id, kind, alive, x, y, tags (string list).
Do not invent host APIs. Missing API = pulse.log.error and skip.
]]

pulse = pulse or {}
pulse.time = pulse.time or { tick = 0, _t = 0 }
pulse.rng = pulse.rng or {}
pulse.log = pulse.log or {}
pulse.bus = pulse.bus or { _h = {} }
pulse.save = pulse.save or { _s = {} }
pulse.input = pulse.input or { queue = {} }
pulse.world = pulse.world or { _e = {}, _n = 0 }

function pulse.time.now()
  return pulse.time._t
end

function pulse.rng.seed(n)
  math.randomseed(math.floor(tonumber(n) or 1))
end

function pulse.rng.int(a, b)
  a, b = math.floor(a), math.floor(b)
  if b < a then a, b = b, a end
  return math.random(a, b)
end

function pulse.rng.float()
  return math.random()
end

local function _log(level)
  return function(fmt, ...)
    -- host captures; Game must not io.write
    local msg = fmt
    if select("#", ...) > 0 then
      msg = string.format(fmt, ...)
    end
    rawset(pulse.log, "_last", { level = level, msg = msg, t = pulse.time._t })
  end
end
pulse.log.debug = _log("debug")
pulse.log.info  = _log("info")
pulse.log.warn  = _log("warn")
pulse.log.error = _log("error")

function pulse.bus.on(topic, fn)
  local h = pulse.bus._h[topic]
  if not h then
    h = {}
    pulse.bus._h[topic] = h
  end
  h[#h + 1] = fn
end

function pulse.bus.emit(topic, payload)
  local h = pulse.bus._h[topic]
  if not h then return end
  for i = 1, #h do
    h[i](payload)
  end
end

function pulse.save.write(name, tbl)
  pulse.save._s[name] = tbl
end

function pulse.save.read(name)
  return pulse.save._s[name]
end

function pulse.world:spawn(kind, data)
  self._n = self._n + 1
  local id = self._n
  local e = { id = id, kind = kind, alive = true, x = 0, y = 0, tags = {} }
  if type(data) == "table" then
    for k, v in pairs(data) do
      e[k] = v
    end
  end
  self._e[id] = e
  return id
end

function pulse.world:get(id)
  return self._e[id]
end

function pulse.world:kill(id)
  local e = self._e[id]
  if e then e.alive = false end
end

function pulse.world:query(kind)
  return coroutine.wrap(function()
    for _, e in pairs(self._e) do
      if e.alive and (kind == nil or e.kind == kind) then
        coroutine.yield(e)
      end
    end
  end)
end

function pulse.world:set(id, key, value)
  local e = self._e[id]
  if not e then return end
  e[key] = value
end

-- Host loop. Game authors do not call this.
function pulse.run(Game, steps)
  assert(type(Game) == "table", "Game module missing")
  assert(type(Game.boot) == "function", "Game.boot required")
  assert(type(Game.tick) == "function", "Game.tick required")
  local world = pulse.world
  Game.boot(world)
  steps = math.floor(tonumber(steps) or 1)
  local dt = 0.05
  for _ = 1, steps do
    pulse.time.tick = pulse.time.tick + 1
    pulse.time._t = pulse.time._t + dt
    local q = pulse.input.queue
    while q[1] do
      local ev = table.remove(q, 1)
      if Game.on_input then Game.on_input(world, ev) end
    end
    Game.tick(world, dt)
  end
  if Game.serialize then
    pulse.save.write("slot0", Game.serialize(world))
  end
end

-- Coding rules for generated source:
-- 1. Only Lua 5.4. No markdown, no ``` fences, no player-facing prose.
-- 2. Comments in English or terse Chinese, never narrative scenes.
-- 3. One file = one module. Last line returns Game.
-- 4. Deterministic: all randomness through pulse.rng.
-- 5. Do not require(), do not io.*, do not os.execute.
-- 6. Never stop early. Fill to the output cap: complete systems, tables,
--    helpers, then test_* functions that call Game.boot/tick.
