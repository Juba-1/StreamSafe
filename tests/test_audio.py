"""Sound in the live path (OBS -> SRT -> StreamSafe -> RTMP): passed through with its picture, delayed alike,
silent while the gate mutes, and still with its picture after dropped frames.

Test material comes from FFmpeg's lavfi: a grey picture with a white corner box on known frames, a steady 440 Hz
tone, and a loud 1 kHz beep that starts on the first frame of each box (the "clap test")."""

from __future__ import annotations

import itertools
import shutil
import socket
import subprocess
import time

import numpy as np
import pytest

from streamsafe.config import Config
from streamsafe.io.avpipe import AUDIO_RATE, AudioBuffer, sample_bounds
from streamsafe.io.sinks import FFmpegSink, MultiSink, ThreadedSink
from streamsafe.io.sources import FFmpegSource
from streamsafe.runtime.runner import run
from tests.stubs import stub_analyzer, stub_scene_analyzer

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

SIZE = (640, 360)
FPS = 30.0
PER_FRAME = AUDIO_RATE // 30  # samples
TONE, BEEP = 0.1, 0.5  # amplitudes (full scale 1.0)
MARKS = (30, 250)  # frames where a box + beep start
SCENE = (90, 150)  # frames with the "explicit scene" background of the stub scene model


def _ffmpeg(*args: str) -> bytes:
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True, capture_output=True).stdout


def make_av(
    path: str, n: int = 300, scene: tuple[int, int] | None = SCENE, marks: tuple[int, ...] = MARKS, audio_codec: str = "aac"
) -> None:
    w, h = SIZE
    video = f"color=c=0x8C785A:s={w}x{h}:r=30:d={n / 30}"
    if scene:
        video += f",drawbox=x=0:y=0:w=iw:h=ih:color=0xC828C8:t=fill:enable='between(n,{scene[0]},{scene[1] - 1})'"
    for m in marks:
        video += f",drawbox=x=0:y=0:w=120:h=120:color=white:t=fill:enable='between(n,{m},{m + 2})'"
    beeps = "+".join(f"between(t,{m / 30:.5f},{m / 30 + 0.1:.5f})" for m in marks)
    sound = f"aevalsrc=exprs='{TONE}*sin(2*PI*440*t)+{BEEP}*sin(2*PI*1000*t)*({beeps})':s={AUDIO_RATE}:d={n / 30}"
    _ffmpeg("-f", "lavfi", "-i", video, "-f", "lavfi", "-i", sound, "-c:v", "libx264", "-preset", "ultrafast", "-g", "30",
            "-pix_fmt", "yuv420p", "-c:a", audio_codec, "-ac", "2", path)  # fmt: skip


def read_sound(path: str) -> np.ndarray:
    """The file's sound, mono, full scale 1.0."""
    return np.frombuffer(_ffmpeg("-i", path, "-vn", "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "s16le", "-"), np.int16) / 32768.0


def read_frames(path: str) -> list[np.ndarray]:
    src = FFmpegSource(path, SIZE, FPS, live=False, audio=False)
    frames = list(src.frames())
    src.close()
    return frames


def codecs(path: str) -> list[str]:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0", path]
    return sorted(set(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.split()))  # .ts lists them twice


def has_box(frame: np.ndarray) -> bool:
    return float(frame[:100, :100].mean()) > 200


def loudness(sound: np.ndarray, frame: int) -> float:
    part = sound[frame * PER_FRAME : (frame + 1) * PER_FRAME]
    return float(np.sqrt((part.astype(np.float64) ** 2).mean())) if len(part) else 0.0


def beep_starts(sound: np.ndarray) -> list[float]:
    """Start of every beep, in frames. Only a beep is louder than 0.35."""
    loud = np.nonzero(np.abs(sound) > 0.35)[0]
    starts = loud[np.diff(loud, prepend=-(10**9)) > AUDIO_RATE // 2]
    return [float(s) / PER_FRAME for s in starts]


class Recorder:
    """A sink that takes sound: keeps, per written frame, the time, whether it shows the box, and its sound."""

    takes_audio = True

    def __init__(self) -> None:
        self.rows: list[tuple[float, np.ndarray, np.ndarray | None]] = []

    def write(self, frame: np.ndarray, audio: np.ndarray | None = None) -> None:
        self.rows.append((time.perf_counter(), frame[:100, :100].copy(), audio))

    def close(self) -> None:
        pass


# ----------------------------------------------------------------------------- source / sink plumbing
def test_source_gives_each_frame_its_own_sound():
    """The sound is a ramp (value = time / 10), so every chunk shows which instant it is from."""
    graph = f"testsrc=size=640x360:rate=30:duration=3[out0];aevalsrc=exprs=t/10|t/10:s={AUDIO_RATE}:d=3[out1]"
    src = FFmpegSource(graph, SIZE, FPS, input_args=("-f", "lavfi"), live=False)
    src.audio_keep(60)
    seen = []
    for frame in src.frames():
        assert frame.shape == (360, 640, 3)
        seen.append(src.frame_index)
    assert seen == list(range(90)) and src.has_audio
    for k in (0, 1, 29, 45, 89):
        chunk = src.audio_chunk(k)
        assert chunk is not None and chunk.shape == (PER_FRAME, 2) and chunk.dtype == np.int16
        assert abs(chunk.mean() / 32768 * 10 - (k + 0.5) / FPS) < 0.002, f"frame {k} got the sound of another instant"
    assert src.audio_chunk(90) is None, "no sound past the end"
    src.close()


def test_source_picture_is_the_same_with_and_without_sound():
    graph = f"testsrc=size=640x360:rate=30:duration=1[out0];anullsrc=r={AUDIO_RATE}:cl=stereo,atrim=0:1[out1]"
    out = []
    for audio in (True, False):
        src = FFmpegSource(graph, SIZE, FPS, input_args=("-f", "lavfi"), live=False, audio=audio)
        out.append(list(src.frames()))
        src.close()
    assert len(out[0]) == len(out[1]) == 30
    assert all(np.array_equal(a, b) for a, b in zip(*out, strict=True))


def test_source_without_a_sound_track_still_gives_the_picture():
    src = FFmpegSource("testsrc=size=640x360:rate=30:duration=1", SIZE, FPS, input_args=("-f", "lavfi"), live=False)
    assert sum(1 for _ in src.frames()) == 30
    assert not src.has_audio and src.audio_chunk(3) is None
    src.close()


def test_source_sound_follows_timestamps_not_sample_counts(tmp_path):
    """Half a second of sound is missing from the input (lost packets, a sender hiccup). The sound after the hole
    must still sit on its picture: it is placed by its timestamps, the hole becomes silence."""
    whole, holed = str(tmp_path / "whole.mkv"), str(tmp_path / "holed.mkv")
    make_av(whole, n=150, scene=None, marks=(30, 110), audio_codec="pcm_s16le")
    _ffmpeg("-i", whole, "-c:v", "copy", "-af", "aselect='not(between(t,2,2.5))'", "-c:a", "pcm_s16le", holed)
    if len(read_sound(holed)) >= len(read_sound(whole)) - 0.4 * AUDIO_RATE:
        # some FFmpeg builds (Ubuntu's apt package) re-time the cut audio, so the file has no hole to test with
        pytest.skip("this FFmpeg build cannot make a test file with a sound hole")
    src = FFmpegSource(holed, SIZE, FPS, live=False)
    src.audio_keep(60)
    boxes = [src.frame_index for frame in src.frames() if has_box(frame)]
    sound = np.concatenate([c for k in range(150) if (c := src.audio_chunk(k)) is not None])[:, 0] / 32768.0
    src.close()
    assert boxes[0] == 30 and 110 in boxes
    starts = beep_starts(sound)
    assert len(starts) == 2 and abs(starts[0] - 30) < 0.5 and abs(starts[1] - 110) < 0.5, starts
    assert loudness(sound, 66) < 0.001 and loudness(sound, 58) > 0.05 and loudness(sound, 78) > 0.05, "the hole is silence"


def test_audio_buffer_forgets_old_sound_and_survives_odd_pieces():
    buf = AudioBuffer(FPS, keep_s=2.0)
    pcm = (np.arange(10 * AUDIO_RATE) // PER_FRAME).astype("<i2").repeat(2).tobytes()  # sample value = frame number
    for i in range(0, len(pcm), 4097):  # pieces that split samples
        buf.append(pcm[i : i + 4097])
    assert buf.chunk(0) is None and buf.chunk(200) is None, "older than keep_s: gone"
    for k in (245, 299):
        chunk = buf.chunk(k)
        assert chunk is not None and chunk.shape == (PER_FRAME, 2) and (chunk == k).all()
    assert buf.chunk(300) is None, "not arrived"


def test_sample_bounds_tile_the_stream_at_any_frame_rate():
    for fps in (30.0, 29.97, 60.0, 25.0, 23.976):
        edges = [sample_bounds(k, fps) for k in range(2000)]
        assert all(a[1] == b[0] for a, b in itertools.pairwise(edges))
        assert abs(edges[-1][1] - 2000 * AUDIO_RATE / fps) <= 0.5, "no drift"


@pytest.mark.parametrize("ext", ["mp4", "flv", "ts"])  # file, RTMP's container, SRT's container
def test_sink_writes_sound_with_its_frame(tmp_path, ext):
    out = str(tmp_path / f"out.{ext}")
    sink = ThreadedSink(FFmpegSink(out, SIZE, FPS))
    assert sink.takes_audio
    t = np.arange(PER_FRAME) / AUDIO_RATE
    for i in range(90):
        frame = np.zeros((360, 640, 3), np.uint8)
        frame[:180] = (0, 0, 255)  # top half red: the picture must not come out upside down
        sound = None  # None = silence
        if 45 <= i < 48:
            frame[:120, :120] = 255
            sound = np.repeat((np.sin(2 * np.pi * 1000 * t) * 0.5 * 32767).astype(np.int16)[:, None], 2, axis=1)
        sink.write(frame, sound)
    sink.close()
    frames = read_frames(out)
    assert len(frames) in (90, 91)  # FLV's millisecond timestamps can make the reader repeat one frame
    assert frames[10][:170, 200:].mean(axis=(0, 1))[2] > 200 and frames[10][190:].mean() < 20
    boxes = [i for i, f in enumerate(frames) if has_box(f)]
    assert len(boxes) == 3 and boxes[0] in (45, 46)
    assert codecs(out) == ["aac", "h264"]
    sound = read_sound(out)
    assert abs(len(sound) - 90 * PER_FRAME) <= 2048, "the sound track is as long as the picture"
    (start,) = beep_starts(sound)
    assert abs(start - 45) < 1.0, "the beep starts with its frame"
    assert loudness(sound, 20) < 0.001 and loudness(sound, 70) < 0.001
    stamps = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time", "-of", "csv=p=0", out],
        capture_output=True, text=True, check=True,
    ).stdout.split()  # fmt: skip
    first = {kind: float(t) for kind, t in (line.split(",") for line in stamps)}
    assert abs(first["video"] - first["audio"]) < 0.03, "picture and sound start together in the container"


def test_sink_without_sound_given_still_has_a_silent_track(tmp_path):
    """Twitch / YouTube ingest need a sound track even when there is nothing to hear."""
    out = str(tmp_path / "out.flv")
    multi = MultiSink([ThreadedSink(FFmpegSink(out, SIZE, FPS))])
    for _ in range(30):
        multi.write(np.zeros((360, 640, 3), np.uint8))
    multi.close()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0", out], capture_output=True, text=True
    )
    assert sorted(probe.stdout.split()) == ["aac", "h264"]
    assert np.abs(read_sound(out)).max() < 0.001


def test_sink_picture_only_when_sound_is_off(tmp_path):
    out = str(tmp_path / "out.mp4")
    sink = FFmpegSink(out, SIZE, FPS, audio=False)
    assert not sink.takes_audio
    for _ in range(10):
        sink.write(np.zeros((360, 640, 3), np.uint8))
    sink.close()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0", out], capture_output=True, text=True
    )
    assert probe.stdout.split() == ["h264"]


# ------------------------------------------------------------------------------------ the live path
def _shown(report: dict) -> list[int]:
    """Source frame number of every output frame (dropped frames are never written)."""
    s = report["summary"]
    assert s["dropped_frames"] == len(report.get("dropped_frames", [])), "too many drops to list"
    dropped = set(report.get("dropped_frames", []))
    return [v for v in range(s["frames_in"]) if v not in dropped]


def _muted(report: dict) -> set[int]:
    return {v for a, b in report["mute_intervals"] for v in range(a, b + 1)}


def test_realtime_sound_is_delayed_with_the_picture_and_silent_while_muted(tmp_path):
    """File with box + beep marks and an explicit scene -> realtime engine (stub scene model) -> H.264 + AAC file."""
    src_path, out = str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4")
    make_av(src_path)
    delay = 1.0
    source = FFmpegSource(src_path, SIZE, FPS, live=False)
    recorder = Recorder()
    sink = MultiSink([recorder, ThreadedSink(FFmpegSink(out, SIZE, FPS))])
    cfg = Config.from_env(env={}, mode="realtime", delay=delay, scene_hold_s=0.5, scene_every_s=0.2)
    started = time.perf_counter()
    try:
        report = run(cfg, source, sink, factory=stub_scene_analyzer)
    finally:
        source.close()
        sink.close()
    s = report["summary"]
    shown, muted = _shown(report), _muted(report)
    assert s["analyzer_errors"] == 0 and s["frames_in"] == 300 and len(shown) >= 280
    assert set(range(*SCENE)) <= muted, "the stub's explicit scene is muted"
    assert all(m in shown and m not in muted for m in MARKS), "the marks are outside the mute (test setup)"

    # 1. what the engine handed to the sink, frame by frame (exact)
    assert len(recorder.rows) == len(shown)
    for v, (_, _, sound) in zip(shown, recorder.rows, strict=True):
        assert (sound is None) == (v in muted), f"frame {v}: sound must be silent exactly while the gate mutes"
        assert sound is None or sound.shape == (PER_FRAME, 2)
    assert s["audio_frames"] == len(shown) - len(muted & set(shown)) and s["audio_missing_frames"] == 0
    for m in MARKS:  # the frame that shows the box carries the start of the beep: one delay for both
        t, corner, sound = recorder.rows[shown.index(m)]
        before = recorder.rows[shown.index(m) - 1]
        assert has_box(corner) and not has_box(before[1])
        assert np.abs(sound).max() > 0.35 * 32768 and np.abs(before[2]).max() < 0.2 * 32768
        # frame m was read from the file at m / fps or later, and is written `delay` after that
        assert t - started >= m / FPS + delay - 0.05, "sound and picture left before the delay was over"

    # 2. the output file viewers get
    frames, sound = read_frames(out), read_sound(out)
    assert len(frames) == len(shown) and abs(len(sound) - len(frames) * PER_FRAME) <= 2048
    box_at = [i for i, f in enumerate(frames) if has_box(f) and not has_box(frames[i - 1])]
    assert box_at == [shown.index(m) for m in MARKS]
    starts = beep_starts(sound)
    assert len(starts) == len(MARKS)
    for box, beep in zip(box_at, starts, strict=True):
        assert abs(beep - box) <= 1.0, f"beep at frame {beep:.2f}, box at frame {box}: lip-sync is off"
    for i, v in enumerate(shown):  # silent exactly in the mute intervals, +-1 frame (AAC smears an edge a little)
        if {v - 1, v, v + 1} <= muted:
            assert loudness(sound, i) < 0.002, f"frame {v} is muted but can be heard"
        elif not {v - 1, v, v + 1} & muted and i < len(shown) - 2:
            assert loudness(sound, i) > 0.5 * TONE / np.sqrt(2), f"frame {v} is not muted but silent"


class CodedSource:
    """Frame k is a flat picture of grey level k and its sound is the constant k. ``lost`` frames never leave the
    source (as when a live feed backs up and the oldest frames are dropped)."""

    live = False
    size = SIZE
    has_audio = True

    def __init__(self, n: int, lost: tuple[int, ...] = ()) -> None:
        self.n, self.lost, self.frame_index = n, lost, -1

    def frames(self):
        for k in range(self.n):
            if k not in self.lost:
                self.frame_index = k
                yield np.full((360, 640, 3), k, np.uint8)

    def audio_keep(self, seconds: float) -> None:
        self.keep_s = seconds

    def audio_chunk(self, index: int) -> np.ndarray:
        return np.full((PER_FRAME, 2), index, np.int16)

    def close(self) -> None:
        pass


class StallingSink:
    """Takes sound; blocks once for a while so the frames behind it are late and get dropped."""

    takes_audio = True

    def __init__(self, stall_at: int, stall_s: float) -> None:
        self.stall_at, self.stall_s = stall_at, stall_s
        self.pairs: list[tuple[int, int | None]] = []

    def write(self, frame: np.ndarray, audio: np.ndarray | None = None) -> None:
        if audio is not None:
            assert (audio == audio[0, 0]).all()
        self.pairs.append((int(frame[200, 300, 0]), None if audio is None else int(audio[0, 0])))
        if len(self.pairs) == self.stall_at:
            time.sleep(self.stall_s)

    def close(self) -> None:
        pass


def test_realtime_dropped_frames_do_not_shift_the_sound():
    """Frames lost in the source and frames the runner drops as late: every frame that is shown still has
    exactly its own sound, before and after the drops."""
    source = CodedSource(200, lost=(40, 41, 42, 120))
    sink = StallingSink(stall_at=70, stall_s=1.0)
    cfg = Config.from_env(env={}, mode="realtime", delay=0.5, max_late_s=0.3)
    report = run(cfg, source, sink, factory=stub_analyzer)
    s = report["summary"]
    assert s["dropped_frames"] >= 5, "the stall did not make the runner drop frames (test setup)"
    assert source.keep_s >= 0.5 + 0.3
    with_sound = [(pic, snd) for pic, snd in sink.pairs if snd is not None]
    assert len(with_sound) >= 150 and s["audio_frames"] == len(with_sound)
    assert all(pic == snd for pic, snd in with_sound), "a frame was sent with the sound of another frame"
    assert with_sound[-1][0] == 199 and not {40, 41, 42, 120} & {snd for _, snd in with_sound}
    assert len(sink.pairs) == s["frames_out"]


def _has_srt() -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-protocols"], capture_output=True, text=True).stdout
    return " srt\n" in out or "\n  srt\n" in out


@pytest.mark.skipif(shutil.which("ffmpeg") is None or not _has_srt(), reason="ffmpeg without SRT")
def test_realtime_srt_in_with_sound_like_obs(tmp_path):
    """OBS-like sender (H.264 + AAC in MPEG-TS over SRT) -> StreamSafe -> MPEG-TS file: sound arrives, in sync."""
    src_path, out = str(tmp_path / "in.mp4"), str(tmp_path / "out.ts")
    make_av(src_path, n=360, scene=None, marks=(240, 320))  # late marks: FFmpeg hands over its first seconds in a burst
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    source = FFmpegSource(f"srt://127.0.0.1:{port}?mode=listener", SIZE, FPS)
    time.sleep(1.0)
    sender = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-re", "-i", src_path, "-c:v", "libx264", "-preset", "ultrafast",
         "-tune", "zerolatency", "-c:a", "aac", "-f", "mpegts", f"srt://127.0.0.1:{port}?mode=caller"],
    )  # fmt: skip
    sink = ThreadedSink(FFmpegSink(out, SIZE, FPS))
    try:
        report = run(Config.from_env(env={}, mode="realtime", delay=1.0), source, sink, factory=stub_analyzer)
    finally:
        sender.wait(timeout=60)
        source.close()
        sink.close()
    s = report["summary"]
    assert source.has_audio and s["analyzer_errors"] == 0 and s["frames_out"] >= 150
    assert s["audio_frames"] >= 120 and s["audio_missing_frames"] <= 15, "sound that did not arrive within the delay"
    frames, sound = read_frames(out), read_sound(out)
    assert len(frames) == s["frames_out"] and abs(len(sound) - len(frames) * PER_FRAME) <= 2048
    box_at = [i for i, f in enumerate(frames) if has_box(f) and not has_box(frames[i - 1])]
    starts = beep_starts(sound)
    assert len(box_at) == len(starts) == 2, "a mark was lost (dropped frames on a slow machine?)"
    for box, beep in zip(box_at, starts, strict=True):
        assert abs(beep - box) <= 1.0, f"beep at frame {beep:.2f}, box at frame {box}"
