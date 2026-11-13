--[[
StreamSafe guard for OBS Studio (Tools -> Scripts -> +). Lua: works in every OBS install, no Python needed.

What it does, 10 times per second:
  * reads the StreamSafe state file (one line: "<alive> <seq> <muted> <full>", see streamsafe/io/obs_link.py)
  * mutes the game audio source(s) while StreamSafe says "muted" (explicit scene on screen)
  * FAIL CLOSED: the game audio is muted AND the slate source is shown over everything while
      - StreamSafe is not running, stopped, or frozen (seq not moving for "stale" ms),
      - the captured StreamSafe window is empty (closed, minimised, capture lost),
      - a game audio source named in the settings does not exist (its mute would silently do nothing),
      - the streamer pressed the PANIC hotkey (Settings -> Hotkeys -> "StreamSafe: PANIC"; press again to end),
  * the raw game capture must never be on air: if it is visible in the current scene, OBS is switched back
    to the protected scene (and the slate shown).
It never touches the stream key, the microphone or any other source.
The state file path should be plain ASCII: Lua's io.open cannot open non-ASCII paths on Windows - it would read as
"not running" (fails closed, but stuck). The installed app shows the right path in its Settings tab (normally
%LOCALAPPDATA%\StreamSafe\obs\state.txt; for a user name with non-ASCII letters its 8.3 short form).
]]

obs = obslua

local state_path = ""
local audio_names = {}
local slate_name = "StreamSafe Slate"
local output_name = "StreamSafe Output"
local raw_name = ""
local protected_scene = ""
local stale_ms = 500
local TICK_MS = 100

local last_seq = nil
local frozen_ms = 0
local applied_mute = nil   -- what this script last set (nil = never), so a manual mute is not undone every tick
local panic = false
local panic_hotkey = nil

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

local function audio_ok()
  if #audio_names == 0 then return false end
  for _, name in ipairs(audio_names) do
    local src = obs.obs_get_source_by_name(name)
    if src == nil then return false end
    obs.obs_source_release(src)
  end
  return true
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

local function visible_in(scene, name)  -- the scene item, and whether it is visible (nil when absent)
  if name == "" then return nil, false end
  local item = obs.obs_scene_find_source_recursive(scene, name)
  if item == nil then return nil, false end
  return item, obs.obs_sceneitem_visible(item)
end

local function output_empty(scene)  -- the window capture of StreamSafe's output shows nothing
  local item, shown = visible_in(scene, output_name)
  if item == nil or not shown then return false end  -- not in this scene (BRB / chatting scene): nothing to check
  local src = obs.obs_get_source_by_name(output_name)
  if src == nil then return true end
  local w = obs.obs_source_get_width(src)
  obs.obs_source_release(src)
  return w == 0
end

local function switch_to_protected()
  if protected_scene == "" then return end
  local target = obs.obs_get_source_by_name(protected_scene)
  if target ~= nil then
    obs.obs_frontend_set_current_scene(target)
    obs.obs_source_release(target)
  end
end

local function tick()
  local scene_src = obs.obs_frontend_get_current_scene()
  local scene = scene_src ~= nil and obs.obs_scene_from_source(scene_src) or nil
  local st = read_state()
  local healthy = false
  if st ~= nil and st.alive then
    if st.seq == last_seq then frozen_ms = frozen_ms + TICK_MS else frozen_ms = 0 end
    last_seq = st.seq
    healthy = frozen_ms < stale_ms
  end
  local raw_on_air = false
  if scene ~= nil then
    local _, raw_shown = visible_in(scene, raw_name)
    raw_on_air = raw_shown
    if output_empty(scene) then healthy = false end
  end
  if panic or not audio_ok() or raw_on_air then healthy = false end
  if scene ~= nil then  -- slate checked every tick: also right after the streamer switches scenes
    local item, shown = visible_in(scene, slate_name)
    if item ~= nil and shown == healthy then obs.obs_sceneitem_set_visible(item, not healthy) end
  end
  if healthy then set_mute(st.muted) else set_mute(true) end
  if scene_src ~= nil then obs.obs_source_release(scene_src) end
  if raw_on_air then switch_to_protected() end
end

local function on_panic(pressed)
  if pressed then panic = not panic end
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
  obs.obs_properties_add_text(p, "output_name", "StreamSafe window capture source name", obs.OBS_TEXT_DEFAULT)
  obs.obs_properties_add_text(p, "raw_name", "Raw game video source (must never be on air)", obs.OBS_TEXT_DEFAULT)
  obs.obs_properties_add_text(p, "protected_scene", "Protected stream scene (switched back to)", obs.OBS_TEXT_DEFAULT)
  obs.obs_properties_add_int(p, "stale_ms", "Fail closed after StreamSafe is frozen for (ms)", 200, 5000, 100)
  return p
end

function script_defaults(settings)
  obs.obs_data_set_default_string(settings, "slate_name", "StreamSafe Slate")
  obs.obs_data_set_default_string(settings, "output_name", "StreamSafe Output")
  obs.obs_data_set_default_int(settings, "stale_ms", 500)
end

function script_update(settings)
  state_path = obs.obs_data_get_string(settings, "state_path")
  audio_names = split(obs.obs_data_get_string(settings, "audio_names"))
  slate_name = obs.obs_data_get_string(settings, "slate_name")
  output_name = obs.obs_data_get_string(settings, "output_name")
  raw_name = obs.obs_data_get_string(settings, "raw_name")
  protected_scene = obs.obs_data_get_string(settings, "protected_scene")
  stale_ms = obs.obs_data_get_int(settings, "stale_ms")
  applied_mute = nil  -- re-apply with the new names
end

function script_load(settings)
  panic_hotkey = obs.obs_hotkey_register_frontend("streamsafe.panic", "StreamSafe: PANIC (cover everything)", on_panic)
  local saved = obs.obs_data_get_array(settings, "streamsafe.panic")
  obs.obs_hotkey_load(panic_hotkey, saved)
  obs.obs_data_array_release(saved)
  obs.timer_add(tick, TICK_MS)
end

function script_save(settings)
  local saved = obs.obs_hotkey_save(panic_hotkey)
  obs.obs_data_set_array(settings, "streamsafe.panic", saved)
  obs.obs_data_array_release(saved)
end

function script_unload()
  obs.timer_remove(tick)
end
