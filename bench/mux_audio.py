"""Review copy WITH sound: the protected video + the original game audio, silenced where the engine muted it.

The engine's output frame v is input frame v (the delay only shifts when it is shown), so a mute interval
[a, b] in frames is [a / fps, (b + 1) / fps] in the clip's own time. Clips downloaded video-only get a quiet
test tone instead, so the mute can still be heard.

usage: python bench/mux_audio.py PROTECTED.mp4 ORIGINAL_CLIP STATS.json OUT.mp4
"""

import json
import subprocess
import sys

import imageio_ffmpeg

FPS = 30.0


def has_audio(ff: str, path: str) -> bool:
    probe = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, text=True)
    return "Audio:" in probe.stderr


def main(protected: str, original: str, stats: str, out: str) -> None:
    with open(stats) as f:
        intervals = json.load(f).get("mute_intervals", [])
    parts = [f"between(t,{a / FPS:.3f},{(b + 1) / FPS:.3f})" for a, b in intervals]
    afilter = f"volume=enable='{'+'.join(parts)}':volume=0" if parts else "anull"
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    game = has_audio(ff, original)
    audio_in = ["-i", original] if game else ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    if not game:
        afilter = "volume=0.3," + afilter
    cmd = [ff, "-loglevel", "error", "-y", "-i", protected, *audio_in, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"]
    cmd += ["-af", afilter, "-c:a", "aac", "-b:a", "160k", "-shortest", out]
    subprocess.run(cmd, check=True)
    print("MUX", out, "muted_intervals", len(parts), "game_audio" if game else "tone", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
