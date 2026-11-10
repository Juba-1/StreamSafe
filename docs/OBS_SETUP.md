# StreamSafe + OBS: console streamer setup (PlayStation / Xbox → capture card → PC)

The game runs on the console. The streamer's PC only runs OBS and StreamSafe, so its GPU is free for our
models (the game does not load it).

```
console ─HDMI─> capture card ─┬─> TV / monitor (the streamer plays live, no delay)
                              └─> PC: OBS "Game Raw" source (hidden scene)
                                       │  OBS Virtual Camera (output type: Source = Game Raw)
                                       ▼
                                  StreamSafe (delay D, covers, scene card)
                                       │  window "StreamSafe Output"
                                       ▼
                       OBS stream scene: Window Capture "StreamSafe Output"
                                         + camera / mic delayed by the same D
                                         + "StreamSafe Slate" image (top, hidden)
                                         + guard script (mutes game audio, fails closed)
                                       ▼
                                  Twitch / Kick / YouTube (OBS keeps the stream key)
```

Why this shape:
- OBS stays the streaming app the streamer knows; StreamSafe never touches the stream key.
- The raw game is never in the stream scene. If StreamSafe stops, its window disappears and the guard
  script shows the slate and mutes the game audio within 0.5 s (fail closed).

## Steps (one time)

1. **Hidden raw scene.** Create a scene `Game Raw` with the capture-card video source (name it `Game Raw`
   too). Never add this video source to the stream scene: the guard switches OBS back to the stream scene
   and shows the slate if it ever appears on air.
2. **Game audio as its own source.** Add the capture card's sound as an **Audio Input Capture** source
   named `Game Audio` (audio only), so the stream scene never contains the raw video.
3. **Virtual camera.** OBS -> Start Virtual Camera -> settings: output type **Source**, source **Game Raw**.
   Never "Program": the engine would see its own output (a loop).
4. **Start StreamSafe:**
   `streamsafe --input "device:OBS Virtual Camera" --output window --delay 2`
   with `OBS_STATE=C:\ProgramData\StreamSafe\state.txt` (an ASCII path: OBS's Lua cannot open paths with
   Arabic or other non-ASCII letters on Windows).
   (`HUD_OCR=1` turns on reading the game's prompts once GTA 6 positions are set.)
5. **Stream scene** (name it `StreamSafe Stream`):
   - Window Capture -> window `StreamSafe Output` (full canvas). Its size is fixed; if it is minimised the
     engine restores it, and while the capture is empty the guard shows the slate.
   - Image source `StreamSafe Slate` (full canvas, top of the list, eye OFF). The script switches it.
   - `Game Audio`: **Advanced Audio Properties -> Sync Offset = D** (2000 ms for D = 2 s), so the sound matches
     the delayed picture.
   - Mic: Sync Offset = D too, so the streamer's voice matches the game the viewers see.
   - Camera: filter **Video Delay (Async)** = D (up to 20 s, DOC-CLAIM: OBS forum). Render Delay stops at
     500 ms. The real picture path adds the virtual camera and window capture on top of D: set the final
     values with the clap test (test 1).
6. **Guard script:** Tools -> Scripts -> + -> `obs/streamsafe_guard.lua`. Fill in: the state file path,
   game audio `Game Audio`, slate `StreamSafe Slate`, output `StreamSafe Output`, raw video `Game Raw`,
   protected scene `StreamSafe Stream`, stale time 500 ms.
7. **Panic key:** Settings -> Hotkeys -> "StreamSafe: PANIC (cover everything)". One press = slate + game
   muted at once (works even if the engine has stopped); press again to end.

## Tests before a real stream (the "soak", 2 hours)

| # | Test | Pass |
|---|---|---|
| 1 | Sync: clap on camera + a beep in the game, recorded from the stream scene | clap, beep and voice together; adjust Video Delay / Sync Offsets |
| 2 | Lowest safe delay: run 10 min at D = 0.5, 1.0, 2.0 | 0 slates from lateness (`failclosed_frames` in the stats) |
| 3 | Kill StreamSafe (Task Manager) during a stream test | slate + game muted in ≤ 0.5 s, OBS keeps streaming |
| 4 | Restart StreamSafe | slate goes away on its own, game audio back |
| 4b | Panic key; minimise the output window; rename the game audio source; show `Game Raw` in the stream scene | slate (+ mute) every time; raw scene switched back |
| 5 | Strip-club / dance scene (GTA V clip played through the capture path) | scene card + game muted, mic still on |
| 6 | 2 hours of mixed gameplay to a private / unlisted stream | no crash; memory flat; no late-frame bursts; no frame without protection |

Server-side, the same long run is automated: `python -m bench.soak OUT 120 clips...` (live UDP feed in real
time → engine → H.264 out, state file watched like the OBS script, memory sampled every 10 s).
