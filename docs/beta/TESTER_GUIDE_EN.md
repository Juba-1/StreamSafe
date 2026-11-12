# StreamSafe beta: tester guide (English)

Arabic version: `TESTER_GUIDE_AR.md`. Thank you for testing. This build is a beta: read `KNOWN_LIMITS.md` first.

## What StreamSafe does

StreamSafe sits between your game and your stream. It holds the picture back for a short delay (2 seconds by
default, set with `--delay`) and checks every frame during that time.

- A character that must be covered gets an **opaque cover** shaped to their pose. It moves with them.
- An explicit scene gets a **full-screen cover**, and the game audio is **muted** while it is on.
- If anything goes wrong (an error, a slow PC, StreamSafe crashing), viewers see a full-screen slate. They never see an unchecked frame.
- There is **no blur and no pixelation**. Covers are solid.

## 1. Requirements

| | |
|---|---|
| System | Windows 10 or 11, 64-bit |
| Graphics card | NVIDIA RTX 3060 or better: recommended (fastest, uses TensorRT). AMD or Intel (any DirectX 12 card): supported, but slower. No graphics card: it will not start. |
| Video memory | 6 GB or more is suggested (the check warns below this) |
| RAM / disk | 8 GB RAM, 2 GB free disk (the check warns below this) |
| Other | OBS Studio, FFmpeg (comes with StreamSafe), a Twitch / Kick / YouTube account |

Always run the check first (step 3).

## 2. Install

The installer is **coming in the installer**: it will install StreamSafe, FFmpeg, the models and the OBS script. Until it
ships, we send you the `StreamSafe` folder and a model folder; the commands below use `StreamSafe.exe` from
a Command Prompt opened in that folder. Where a step says "the model folder", use the folder we sent you
(`<model folder>` below).

## 3. First run

**3.1 Check your PC**

```
StreamSafe.exe check --models "<model folder>"
```

Add `--lang ar` for Arabic hints, `--engines` on an NVIDIA card to see if the fast engines are ready.

Each line is `ok`, `warn` or `fail`. The last line is the verdict:

- `ok`: go on.
- `warn`: it works but something is weak (low memory, OBS not found, slower mode). Read the hint.
- `fail`: StreamSafe will refuse to start. Fix it first (see the table at the end).

**3.2 Models.** The model files are checked every time (they must match their fingerprints). Downloading them for you
is **coming in the installer**; for now they come in the folder we send.

**3.3 NVIDIA only: one-time preparation (about 4 minutes).** TensorRT builds the fast engines for your exact card and
driver. Do it once, before your first stream. It is redone after a graphics driver update.

```
StreamSafe.exe prepare --models "<model folder>"
```

It prints `preparing 1/4 ...` and ends with `PREPARE_OK`. If it ends with `PREPARE_FAILED`, send us the text.
(If you skip it, StreamSafe still runs on the slower mode. AMD and Intel cards skip this step.)

## 4. Two OBS setups

Use **A** if you can: it is simple and your stream key never leaves the one command you type. Use **B** if you
need StreamSafe's picture inside an OBS scene.

> **Sound in setup A:** StreamSafe passes on the sound OBS sends (game + mic, as one mix), delayed by the same
> amount as the picture, so they stay in sync without any OBS sync offsets. While a full-screen cover is on, the
> whole mix is silent in A, **your mic too** (OBS sends one mixed track; StreamSafe cannot tell the game from your
> voice). If your mic must stay on during covers, use B. Sound in A is new: please do the clap test on a private
> stream and tell us in the feedback form.

### Setup A: OBS sends to StreamSafe (SRT), StreamSafe sends to the platform

```
OBS ──SRT──> StreamSafe (delay + cover) ──RTMPS──> Twitch / Kick / YouTube
```

1. In OBS, build your scene with the game as usual.
2. Get your platform's ingest **server URL** and **stream key** (Twitch: Creator Dashboard; Kick: Stream settings; YouTube: Studio, Go live). Join them as `server/key`.
   Example shape: `rtmps://<ingest-server>/app/<your-key>`. Use the `rtmps://` address your platform gives.
3. Open Command Prompt in the StreamSafe folder and start StreamSafe first. It waits for OBS:
   ```
   StreamSafe.exe run --input "srt://127.0.0.1:9000?mode=listener" --output "rtmps://<ingest-server>/app/<your-key>" --delay 2
   ```
4. In OBS: **Settings -> Stream**, Service **Custom**, Server `srt://127.0.0.1:9000`, leave the Stream Key empty. Use H.264 as the encoder, as usual. Click OK.
5. Click **Start Streaming** in OBS. Now OBS is sending to StreamSafe, **not** to the platform.
6. Open your stream page on the platform. After the delay you should see your picture. OBS's own preview has **no** delay: you play on the live picture, viewers see the protected one.
7. To stop: click Stop Streaming in OBS and let StreamSafe end by itself (see "Ending a session" below).

Why A is safe: OBS never holds your stream key, and if StreamSafe stops, nothing goes to the platform. The raw game never goes out.
Note: the key is typed in the command, so do not film or share your Command Prompt, and do not paste that command to anyone.

### Setup B: StreamSafe shows its picture in a window, OBS captures it

This setup needs more OBS work, but it carries your sound and lets OBS keep the stream key.

```
Game capture (hidden "Game Raw")  ->  OBS Virtual Camera  ->  StreamSafe  ->  window "StreamSafe Output"
                                   ->  OBS stream scene captures that window  ->  platform
```

1. **Hidden scene.** Create a scene called `Game Raw` with your game capture, named `Game Raw`. Never add this video to your stream scene.
2. **Game sound as its own source.** Add an **Audio Input Capture** (or Audio Output Capture) named `Game Audio`.
3. **Virtual camera.** Click **Start Virtual Camera** (in Controls, the gear next to it): Output Type **Source**, Source **Game Raw**. Not "Program".
4. **Start StreamSafe** (set the state file first; the path must use English letters only):
   ```
   mkdir C:\ProgramData\StreamSafe
   set OBS_STATE=C:\ProgramData\StreamSafe\state.txt
   StreamSafe.exe run --input "device:OBS Virtual Camera" --output window --delay 2
   ```
   A window named `StreamSafe Output` opens. Do not close or minimise it (if you minimise it, StreamSafe restores it).
5. **Stream scene** called `StreamSafe Stream`:
   - **Window Capture** of `StreamSafe Output`, full canvas.
   - An image source named `StreamSafe Slate`, on top, eye **off** (the script turns it on when needed).
   - `Game Audio`: **Advanced Audio Properties -> Sync Offset** = 2000 ms (same as the delay).
   - Your mic: Sync Offset = 2000 ms too. Camera: filter **Video Delay (Async)** = 2000 ms.
   - These numbers are a start. Do the clap test (clap on camera, play a beep in the game, record, check the three line up) and adjust.
6. **Guard script.** OBS -> **Tools -> Scripts -> +** -> choose `streamsafe_guard.lua` (location of the file: **coming in the installer**; for now we send it). Fill in: state file `C:\ProgramData\StreamSafe\state.txt`; game audio `Game Audio`; slate `StreamSafe Slate`; output `StreamSafe Output`; raw video `Game Raw`; protected scene `StreamSafe Stream`; stale time `500`.
   The script mutes the game and shows the slate within 0.5 s if StreamSafe stops, freezes, or the window goes empty.
7. **Panic key.** OBS -> Settings -> Hotkeys -> "StreamSafe: PANIC (cover everything)": set a key. One press = slate and mute; press again to end.
8. Stream from OBS to the platform as you normally do, with the `StreamSafe Stream` scene. **Test with a private or unlisted stream first.**
9. Quick test before going live: kill StreamSafe in Task Manager. The slate must appear and the game must be muted within a second.

## 5. The delay

- Viewers see you about 2 seconds later than real time (plus your platform's own delay). That is how StreamSafe has time to check each frame.
- **Chat will feel late.** You see chat's reply to what the viewer saw 2+ seconds ago. Warn your viewers, or react to "late" chat.
- In setup B, your mic and camera must be delayed by the same amount (step 5), or your voice will be ahead of the game.
- You can change it with `--delay <seconds>`. A shorter delay gives the PC less time and can cause more full covers (slow PC). Please test the default first.

## 6. Full-screen cover and muted sound are normal

When StreamSafe sees an explicit scene, it covers the **whole screen** with a card and **mutes the game**. That is the
intended behaviour, not a bug. It also shows the full-screen slate for a moment when it starts, and when something is wrong or slow.
In setup B your mic stays on; in setup A the mic is silent too while the cover is on (see the note in section 4).
When the scene is over, picture and sound come back by themselves.

## 7. If it covers too much or too little

There is no setting for you to change in the beta. Do not try to tune it. Instead:

1. **Too much** (covers a character or screen that was fine) or **too little / late** (something showed that should have been covered): note the **time** (stream clock or your clock) and what was on screen. **If a real revealing frame reached viewers, tell us right away, do not wait for the form.**
2. If you recorded the stream locally, keep that recording; we may ask for the short clip around that time.
3. Mention it in the feedback form with the time. It is the most useful thing you can send.

## 8. Ending a session and sending the report

At a normal end, StreamSafe writes a short report in:

```
%LOCALAPPDATA%\StreamSafe\reports
```

(Type that in the File Explorer address bar.) It prints `SESSION_REPORT <path>` when it is done. You get two files with the same name: `session_YYYYMMDD_HHMMSS.txt` (read it, Arabic and English) and `.json`.

**Important:** if you stop StreamSafe with **Ctrl+C**, no report is written in this beta. Let it end on its own: end the stream or the input
(in setup A, click Stop Streaming in OBS; StreamSafe should end when OBS stops sending. If it does not end after a minute, tell us, then use Ctrl+C and say that no report was written).

To send: attach the `.txt` and `.json` of each session to the message you send us with the feedback form (`FEEDBACK_FORM.md`).
The verdict line says `ok` or `check`; send it either way.

**Privacy.** The report holds only counts and times: your GPU name, Windows version, duration, how many seconds of cover and mute, and error counts.
It has **no pictures, no stream URL or key, and no file paths**. The command you typed is not in it.
Open the `.txt` and read it before you send it.

## 9. Troubleshooting (from `StreamSafe.exe check`)

| Line in the check | Meaning | What to do |
|---|---|---|
| `gpu` FAIL | No graphics card found | Install the latest driver (NVIDIA / AMD / Intel), run check again |
| `gpu` WARN | Under 6 GB video memory | Close other programs while streaming; tell us your card |
| `runtime` FAIL | StreamSafe cannot use your GPU, or files are missing | Update your graphics driver and Windows; if it stays, reinstall |
| `runtime` WARN (DirectML on NVIDIA) | Works, but slower than the NVIDIA mode | Use the NVIDIA edition and run `prepare` |
| `ffmpeg` FAIL | FFmpeg is missing | Reinstall StreamSafe |
| `models` FAIL | Model files are missing or damaged | Re-copy / reinstall the model folder |
| `python` FAIL | Internal runtime too old | Reinstall StreamSafe with the official build |
| `os` WARN | Not Windows | Live streaming is supported on Windows only |
| `obs` WARN | OBS not found in its usual folder | Install OBS or ignore if it is installed elsewhere |
| `ram` / `disk` WARN | Low memory or free space | Close other programs / free some space |
| `engines` WARN | NVIDIA fast engines not ready | Run `StreamSafe.exe prepare --models "<model folder>"` |

Other problems:

| Problem | What to do |
|---|---|
| Stream shows the slate all the time | Look at the Command Prompt for errors. A slow PC causes this: close other programs and tell us your card |
| Nothing arrives in setup A | Start StreamSafe **before** OBS. Check `srt://127.0.0.1:9000` in OBS (no key) and that the command has `?mode=listener` |
| Nothing arrives on the platform | Check the server/key, and that your platform accepts `rtmps://` |
| No sound in setup A | Check that OBS itself has sound (the mixer bars move) and that you run the current StreamSafe build. Silence during a full-screen cover, and for about half a second after it, is intended |
| Window `StreamSafe Output` not found in OBS | StreamSafe must be running first; use "Window Capture" and pick it from the list |
| Slate stuck after restarting StreamSafe (B) | Check the state file path is `C:\ProgramData\StreamSafe\state.txt` and English letters only |

Exit codes of `check`: 0 = ok, 1 = warnings, 2 = a failure.
