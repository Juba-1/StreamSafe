# StreamSafe beta: tester guide (English)

Arabic version: `TESTER_GUIDE_AR.md`. Thank you for testing. This build is a beta: read `KNOWN_LIMITS.md` first.

## What StreamSafe does

StreamSafe sits between your game and your stream. It holds the picture back for a short delay (2 seconds by
default) and checks every frame during that time.

- A character that must be covered gets an **opaque cover** shaped to their pose. It moves with them.
- An explicit scene gets a **full-screen cover**, and the **sound is muted** while it is on.
- If anything goes wrong (an error, a slow PC, StreamSafe crashing), viewers see a full-screen slate. They never see an unchecked frame.
- There is **no blur and no pixelation**. Covers are solid.

## 1. Requirements

| | |
|---|---|
| System | Windows 10 or 11, 64-bit |
| Graphics card | NVIDIA RTX 3060 or better: recommended (fast mode). AMD or Intel (any DirectX 12 card): supported, but slower. No graphics card: it will not start. |
| Video memory | 6 GB or more is suggested |
| RAM / disk | 8 GB RAM. Free disk: about 5 GB for the models, 12 GB if you enable the fast NVIDIA mode |
| Other | OBS Studio, a Twitch / Kick / YouTube account. FFmpeg comes with StreamSafe. |
| Rights | No administrator rights needed |

## 2. Install

1. Download `StreamSafe-Beta-Setup.exe` from the link we sent you.
2. Double-click it. A blue box **"Windows protected your PC"** appears.
   This is expected: the beta is **not code-signed yet** (we have no publisher certificate), so Windows does not know the publisher.
   Click **More info**, then **Run anyway**.
   If your antivirus blocks it, tell us instead of turning the antivirus off.
3. Click through the setup (Next). It asks for no administrator rights and installs to `%LOCALAPPDATA%\Programs\StreamSafe`.
4. At the end leave "Open StreamSafe" ticked and finish. Later you find it in the Start menu as **StreamSafe**.
5. Uninstall: Windows Settings > Apps > StreamSafe > Uninstall. It asks whether to also delete the models and reports (several GB).

## 3. First run

Open StreamSafe. The window starts in Arabic; press the language button at the top right (**عربي**) to switch. It has three tabs: **Home**, **Settings**, **Check**.

1. **Check.** It runs by itself when the window opens (**Check** tab). Each line is green (fine), orange (remark) or red (must be fixed).
   With a red line you cannot start. Press **Check again** after fixing. (Fix table: section 10.)
2. **Models (once, about 3.4 GB).** On **Home** a **Setup** box appears. Press **Download models**.
   They come from Hugging Face and the app verifies every file. The progress bar shows where it is; **Cancel** changes nothing.
   - If it says it cannot open the download (not public yet), press **Choose zip file...** or **Choose models folder...** and pick the file we sent you.
   - When it ends you see "The models are installed" and the check runs again.
3. **NVIDIA cards only: fast mode (once).** After the models, the **Check** tab shows a box: **Enable fast NVIDIA mode**.
   - It downloads NVIDIA libraries (about 3.5 GB), then prepares your card (about 4 minutes). All once.
   - Do it before your first stream. Without it the app still works, only slower.
   - **Not now** hides the box and it does not ask again. Press it only if you mean it.
   - AMD and Intel cards skip this step.
4. When the check is green or orange and **Start protection** is enabled, you are ready.

## 4. Two ways to connect OBS

Use **A** if you can: simpler, and the stream key stays inside StreamSafe. Use **B** if the mic must stay on during a cover, or you want the StreamSafe picture inside an OBS scene.

> **Sound.** In A, StreamSafe passes on the sound OBS sends (game and mic mixed in one track) and delays it exactly like the picture,
> so they stay in sync with no Sync Offset. During a full-screen cover the whole track is muted, **mic included**. If the mic must stay on, use B.

### Setup A: OBS sends to StreamSafe (SRT), StreamSafe sends to the platform

```
OBS ──SRT──> StreamSafe (delay + cover) ──RTMPS──> Twitch / Kick / YouTube
```

1. Build your scene in OBS as usual.
2. From your platform take the **server URL** and the **stream key** (Twitch: Creator Dashboard. Kick: stream settings. YouTube: Studio, then Go live). Keep them separate; do not join them.
3. In StreamSafe open **Settings**:
   - **Input:** choose **OBS sends over SRT**. Port `9000` (keep it unless it is taken).
   - **Output:** choose **Stream to the platform (RTMP/RTMPS)**.
   - **Stream address (without the key):** paste the server URL. It must start with `rtmps://` or `rtmp://`.
   - **Stream key:** paste the key (shown as dots). To keep it, tick **Remember the key on this PC**: it is stored protected by your Windows account. Otherwise paste it each time.
   - **Delay:** `2`.
4. Go to Home and press **Start protection**. The status reads "Starting... screen is covered" while it waits for OBS. **Start StreamSafe before OBS.**
5. In OBS: **Settings > Stream**, Service **Custom**, Server `srt://127.0.0.1:9000`, leave Stream Key **empty**. Encoder H.264 as usual. Press OK.
   (The same line is shown under the Port field in StreamSafe's Settings and follows the port number.)
6. Press **Start Streaming** in OBS. OBS now sends to StreamSafe, **not** to the platform.
7. The status turns **Protected** (green). Open your stream page on the platform: after the delay you see your picture **and sound**.
   Your OBS preview has no delay; viewers see the protected picture.
8. To end: press Stop Streaming in OBS, then **Stop** in StreamSafe (section 8).

Why A is safer: OBS never holds the stream key, and if StreamSafe stops nothing reaches the platform. The raw game never goes out.
The key is never on a command line, never in the report, and is blanked out of every line the window shows.

### Setup B: StreamSafe shows the picture in a window, OBS captures it

More OBS work, but the mic stays on during a cover, and OBS keeps the stream key.

```
game capture (hidden scene "Game Raw") -> OBS Virtual Camera -> StreamSafe -> window "StreamSafe Output"
                                       -> OBS stream scene captures the window -> platform
```

1. **Hidden scene.** Make a scene `Game Raw` with your game capture, and name the source `Game Raw`. Never add this video to the stream scene.
2. **Game sound as its own source.** Add an **Audio Input Capture** (or Audio Output Capture) named `Game Audio`.
3. **Virtual camera.** Press **Start Virtual Camera** (gear beside it): Output Type **Source**, source **Game Raw**. Not "Program".
4. **StreamSafe settings.** In **Settings:** Input **OBS Virtual Camera** (name `OBS Virtual Camera`), Output **A window that OBS captures**, Delay `2`.
5. Press **Start protection**. A window named `StreamSafe Output` opens. Do not close or minimise it (if you minimise it, StreamSafe restores it).
6. **Stream scene** named `StreamSafe Stream`:
   - **Window Capture** of the `StreamSafe Output` window, full canvas.
   - An image source named `StreamSafe Slate` on top of everything, eye **off** (the script turns it on when needed).
   - `Game Audio`: **Advanced Audio Properties > Sync Offset** = 2000 ms (the same as the delay).
   - Mic: Sync Offset = 2000 ms too. Camera: filter **Video Delay (Async)** = 2000 ms.
   - These numbers are a start. Do the clap test (clap in front of the camera, play a beep in the game, record, check all three line up) and adjust.
7. **Guard script.** In StreamSafe, **Settings**, section **OBS setup**, there are two paths, each with a **Copy** button:
   - **Guard script** (`...\Programs\StreamSafe\obs\streamsafe_guard.lua`): copy it, then in OBS **Tools > Scripts > +** and paste it into the file name box.
   - **State file for the OBS script** (`...\StreamSafe\obs\state.txt`): copy it and paste it into the script's state file field.
   - Other fields: game audio `Game Audio`; slate `StreamSafe Slate`; output `StreamSafe Output`; raw video `Game Raw`; protected scene `StreamSafe Stream`; stale time `500`.
   The script mutes the game and shows the slate within half a second if StreamSafe stops, freezes, or the window empties.
8. **Panic key.** OBS > Settings > Hotkeys > "StreamSafe: PANIC (cover everything)": assign a key. One press = cover and mute; press again to release.
9. Stream from OBS to the platform as usual with the `StreamSafe Stream` scene. **Try a private or unlisted stream first.**
10. Quick test before a real stream: kill `StreamSafe` in Task Manager. The slate must appear and the game must mute within a second.

## 5. Delay

- Viewers see you about 2 seconds behind real time (on top of the platform's own delay). That is how StreamSafe has time to check every frame.
- **Chat will feel late.** Tell your viewers.
- In B the mic and camera must be delayed by the same amount (step 6), or your voice will run ahead of the game.
- You can change it in **Settings > Delay** (0.5 to 10 seconds). A shorter delay gives the PC less time and can cause more full covers. Try the default first.

## 6. Full-screen cover and mute are normal

When StreamSafe sees an explicit scene it covers the **whole screen** and **mutes the sound**. That is the intended behaviour, not a fault.
The slate also shows for a moment at start, and whenever something is wrong or slow. The window status then reads **Fully covered**. When the scene ends, picture and sound come back by themselves.
In A the mic is muted with the game; in B the mic stays on.

## 7. Too much cover, or too little

There is no setting to change in the beta. Instead:

1. **Too much cover** (a normal character or screen covered) or **too little / late** (something that should have been covered showed): note the **time** and what was on screen. **If a really explicit frame reached viewers, tell us right away; do not wait for the form.**
2. If you record your stream locally, keep the recording; we may ask for a short clip around that time.
3. Write it in the feedback form with the time. This is the most useful thing you can send.

## 8. Ending a session and sending the report

1. Press **Stop** on **Home** (in A stop OBS first). You see "Stopping and writing the report..."; wait until it clears.
   If you close the window while protection is running, it asks and stops the same way.
2. Press **Open last session report**: it opens the `.txt` (Arabic and English). Read it.
3. The files are in `%LOCALAPPDATA%\StreamSafe\reports` (paste that into the Explorer address bar): `session_YYYYMMDD_HHMMSS.txt` and `.json`.
4. Send both files for every session with the feedback form (`FEEDBACK_FORM.md`). The result line says `ok` or `check`; send it either way.

If the program crashed or you killed it in Task Manager, there may be no report. Tell us in the form.

**Privacy.** The report has numbers and times only: graphics card name, Windows version, stream length, seconds of cover and mute, error counts.
**No images, no stream URL or key, no file paths.**

## 9. Beta expiry

- At the top of **Home** an orange line reads "Beta version: ends in N days (date)".
- After that date **Start protection** is disabled and the status reads **Not ready**. A stream that is already running is not cut, but you cannot start a new one.
- Then download the new version from us and install it over the old one (close StreamSafe first). Your models and settings stay.

## 10. Troubleshooting

Lines in the **Check** tab:

| Line | Meaning | What to do |
|---|---|---|
| Graphics card (red) | No graphics card found | Install the latest driver (NVIDIA / AMD / Intel), press **Check again** |
| Graphics card (orange) | Under 6 GB video memory | Close other programs while streaming, and tell us your card |
| GPU runtime (red) | The app cannot use the card | Update the driver and Windows; if it stays, reinstall |
| GPU runtime (orange, DirectML on NVIDIA) | Works but slower | Enable the fast NVIDIA mode (section 3) |
| FFmpeg or Decision files (red) | A program file is missing | Reinstall |
| Model files (red) | Not installed or damaged | On **Home** press **Download models** (or choose the file we sent) |
| NVIDIA preparation (orange) | Fast mode not ready | Enable it from the same tab |
| OBS (orange) | OBS not in its usual place | Install OBS, or ignore if it is installed elsewhere |
| Memory / Disk space (orange) | Low RAM or space | Close programs / free space |

Other problems:

| Problem | What to do |
|---|---|
| **Start protection** refuses and says "Cannot start now" | Read the reason it shows (models, red check, missing key, bad stream address, beta ended) |
| The whole stream is the slate | Look at the **Check** tab. A slow PC causes this: close other programs and tell us your card |
| Status "Engine not answering: screen covered" or "Stopped by an error" | Stop and start again. If it repeats, send us the report |
| Nothing arrives in A | Start StreamSafe **before** OBS. Check `srt://127.0.0.1:<port>` in OBS (no key) and that the port matches Settings |
| Nothing reaches the platform in A | Check the stream address and key, and that the address is `rtmps://` or `rtmp://` |
| No sound in A | Check that OBS's own mixer bars move. Silence during a full cover and about half a second after is intended |
| `StreamSafe Output` window not in OBS (B) | StreamSafe must be running first (status not Stopped); use Window Capture and pick it from the list |
| Slate stuck in B after starting StreamSafe | Make sure the **State file** path in the OBS script settings is exactly what the **Copy** button gave you |
| Cannot find StreamSafe after install | Start menu, type StreamSafe |

## 11. Advanced appendix: command line (optional)

Not needed for normal testing. The installed program also has the console version `StreamSafe.exe` next to `StreamSafeApp.exe` in `%LOCALAPPDATA%\Programs\StreamSafe`.
Open a Command Prompt there:

```
StreamSafe.exe check               the check (add --lang ar for Arabic)
StreamSafe.exe prepare             NVIDIA preparation (after enabling fast mode in the window)
StreamSafe.exe about               version, beta end date, folders
StreamSafe.exe run --input "srt://127.0.0.1:9000?mode=listener" --output window --delay 2
```

- Models and reports live in `%LOCALAPPDATA%\StreamSafe` (no `--models` needed).
- For a live stream from the command line do not type the key in the command: put the full address in an environment variable and use `--output env:NAME`.
- **Ctrl+C** ends the session and still writes the report.
- The window and the command line cannot use the same port at the same time.
