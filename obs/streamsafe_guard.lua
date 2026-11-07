--[[
StreamSafe guard for OBS Studio (Tools -> Scripts -> +). Lua: works in every OBS install, no Python needed.

What it does, 10 times per second:
  * reads the StreamSafe state file (one line: "<alive> <seq> <muted> <full>", see streamsafe/io/obs_link.py)
  * mutes the game audio source(s) while StreamSafe says "muted" (explicit scene on screen)
  * FAIL CLOSED: if StreamSafe is not running, stopped, or frozen (seq not moving for "stale" ms), the game
    audio is muted AND the slate source is shown over everything, until StreamSafe is healthy again.
It never touches the stream key, the microphone or any other source.
]]

obs = obslua

local state_path = ""
local audio_names = {}
local slate_name = "StreamSafe Slate"
local stale_ms = 500
local TICK_MS = 100

local last_seq = nil
local frozen_ms = 0
local applied_mute = nil   -- what this script last set (nil = never), so a manual mute is not undone every tick

local function split(s)
  local out = {}
  for raw in string.gmatch(s or "", "([^,]+)") do
    local name = raw:match("^%s*(.-)%s*$")
    if name ~= "" then table.insert(out, name) end
  end
  return out
end

local function read_state()
  local f = io.open(state_path, "r")
  if f == nil then return nil end
  local line = f:read("*l")
  f:close()
  if line == nil then return nil end
  local alive, seq, muted, full = line:match("^(%d) (%-?%d+) (%d) (%d)")
  if alive == nil then return nil end
  return { alive = alive == "1", seq = tonumber(seq), muted = muted == "1", full = full == "1" }
end

local function set_mute(on)
  if applied_mute == on then return end
  for _, name in ipairs(audio_names) do
    local src = obs.obs_get_source_by_name(name)
    if src ~= nil then
      obs.obs_source_set_muted(src, on)
      obs.obs_source_release(src)
    end
  end
  applied_mute = on
end

local function set_slate(on)  -- checked every tick: also right after the streamer switches scenes
  local scene_src = obs.obs_frontend_get_current_scene()
  if scene_src == nil then return end
  local scene = obs.obs_scene_from_source(scene_src)
  local item = obs.obs_scene_find_source_recursive(scene, slate_name)
  if item ~= nil and obs.obs_sceneitem_visible(item) ~= on then obs.obs_sceneitem_set_visible(item, on) end
  obs.obs_source_release(scene_src)
end

local function tick()
  local st = read_state()
  local healthy = false
  if st ~= nil and st.alive then
    if st.seq == last_seq then frozen_ms = frozen_ms + TICK_MS else frozen_ms = 0 end
    last_seq = st.seq
    healthy = frozen_ms < stale_ms
  end
  if healthy then
    set_slate(false)
    set_mute(st.muted)
  else
    set_slate(true)
    set_mute(true)
  end
end

function script_description()
  return "StreamSafe guard: mutes the game audio with StreamSafe's full-screen cover and fails closed " ..
         "(mute + slate) whenever StreamSafe is not running."
end

function script_properties()
  local p = obs.obs_properties_create()
  obs.obs_properties_add_path(p, "state_path", "StreamSafe state file", obs.OBS_PATH_FILE, "*.txt", nil)
  obs.obs_properties_add_text(p, "audio_names", "Game audio source(s), comma separated", obs.OBS_TEXT_DEFAULT)
  obs.obs_properties_add_text(p, "slate_name", "Slate source name (top of the stream scene)", obs.OBS_TEXT_DEFAULT)
  obs.obs_properties_add_int(p, "stale_ms", "Fail closed after StreamSafe is frozen for (ms)", 200, 5000, 100)
  return p
end

function script_defaults(settings)
  obs.obs_data_set_default_string(settings, "slate_name", "StreamSafe Slate")
  obs.obs_data_set_default_int(settings, "stale_ms", 500)
end

function script_update(settings)
  state_path = obs.obs_data_get_string(settings, "state_path")
  audio_names = split(obs.obs_data_get_string(settings, "audio_names"))
  slate_name = obs.obs_data_get_string(settings, "slate_name")
  stale_ms = obs.obs_data_get_int(settings, "stale_ms")
  applied_mute = nil  -- re-apply with the new names
end

function script_load(settings)
  obs.timer_add(tick, TICK_MS)
end

function script_unload()
  obs.timer_remove(tick)
end
