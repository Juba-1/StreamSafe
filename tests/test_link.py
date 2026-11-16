"""The two network links: OBS -> us (SRT) and us -> the platform (RTMP / RTMPS), and what the streamer types.

Real FFmpeg where the behaviour is FFmpeg's (a busy port, a refused connection, a sender that goes and comes
back); a fake FFmpeg process where the timing must be exact (a platform that drops in the middle of a stream)."""

from __future__ import annotations

import io
import queue
import shutil
import socket
import subprocess
import threading
import time

import numpy as np
import pytest

from streamsafe.app import engine, ingest
from streamsafe.io import link
from streamsafe.io.sinks import FFmpegSink, PlatformRefused
from streamsafe.io.sources import FFmpegSource, parse_dshow_devices

KEY = "sk_us-west-2_ABCdef123456_xyz"
HAVE_FFMPEG = shutil.which("ffmpeg") is not None


@pytest.fixture
def events():
    got: list[dict] = []
    link.capture(lambda line: got.append(link.parse(line)))
    yield got
    link.capture(None)


# --- what the streamer pastes -------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "server,key,platform,want_server",
    [
        # Kick shows its server without /app: the field report of 16 Nov 2026
        (
            "rtmps://fa723fc1b171.global-contribute.live-video.net/",
            KEY,
            "kick",
            "rtmps://fa723fc1b171.global-contribute.live-video.net:443/app",
        ),
        (
            "rtmps://fa723fc1b171.global-contribute.live-video.net",
            KEY,
            "custom",
            "rtmps://fa723fc1b171.global-contribute.live-video.net:443/app",
        ),
        (
            "  rtmps://fa723fc1b171.global-contribute.live-video.net:443/app/ \n",
            f" {KEY} ",
            "kick",
            "rtmps://fa723fc1b171.global-contribute.live-video.net:443/app",
        ),
        ("", KEY, "kick", ingest.PRESETS["kick"]),
        ("", KEY, "twitch", "rtmps://live.twitch.tv:443/app"),
        ("", KEY, "youtube", "rtmps://a.rtmps.youtube.com:443/live2"),
        ("rtmp://a.rtmp.youtube.com", KEY, "youtube", "rtmp://a.rtmp.youtube.com/live2"),
        ("rtmps://push-rtmp-l11-sg01.tiktokcdn.com/game", KEY, "tiktok", "rtmps://push-rtmp-l11-sg01.tiktokcdn.com/game"),
    ],
)
def test_server_and_key_are_normalised(server, key, platform, want_server):
    t = ingest.resolve(server, key, platform)
    assert t.problems == () and t.server == want_server and t.key == KEY
    assert t.url == f"{want_server}/{KEY}"


def test_whole_address_pasted_in_either_box_is_split():
    full = f"rtmps://fa723fc1b171.global-contribute.live-video.net/app/{KEY}"
    for server, key in ((full, ""), ("", full), (full, KEY)):
        t = ingest.resolve(server, key, "kick")
        assert t.problems == () and t.key == KEY and t.server.endswith(":443/app"), (server, key, t)
    # Kick's server and key run together without /app (what users copy as one line)
    t = ingest.resolve(f"rtmps://fa723fc1b171.global-contribute.live-video.net/{KEY}", "", "custom")
    assert t.key == KEY and t.server.endswith("/app")


def test_bad_values_say_why():
    assert "bad_rtmp" in ingest.resolve("http://example.com/app", KEY).problems
    assert "bad_rtmp" in ingest.resolve("", KEY, "tiktok").problems  # TikTok has no fixed server: it must be pasted
    assert "bad_rtmp_app" in ingest.resolve("rtmp://my.server.example", KEY).problems
    assert "bad_key" in ingest.resolve("rtmps://live.twitch.tv/app", "").problems
    assert "bad_key" in ingest.resolve("rtmps://live.twitch.tv/app", "a<b>").problems
    assert ingest.resolve("rtmps://live.twitch.tv/app", " ab cd ").key == "abcd"  # stray spaces of a paste go


def test_output_url_of_the_app_uses_the_same_rules():
    assert engine.output_url("rtmps://fa723fc1b171.global-contribute.live-video.net/", KEY, "kick").endswith(f":443/app/{KEY}")
    assert engine.output_url("not an address", KEY) == ""


def test_guess_platform():
    assert ingest.guess_platform("rtmps://x.global-contribute.live-video.net") == "kick"
    assert ingest.guess_platform("rtmp://live.twitch.tv/app") == "twitch"
    assert ingest.guess_platform("rtmp://a.rtmp.youtube.com/live2") == "youtube"
    assert ingest.guess_platform("rtmp://push.tiktokcdn.com/game") == "tiktok"
    assert ingest.guess_platform("rtmp://192.168.1.5/live") == "custom"


# --- the key never shows ------------------------------------------------------------------------------------
def test_secrets_never_leave():
    url = f"rtmps://fa723fc1b171.global-contribute.live-video.net:443/app/{KEY}"
    assert KEY not in link.redact_url(url) and link.redact_url(url).endswith("/app/***")
    text = f"Error opening output {url}: I/O error ({KEY})"
    assert KEY not in link.Secrets(url).clean(text)
    assert "user:pw" not in link.redact_url("rtmp://user:pw@host/app/k1234")


@pytest.mark.parametrize(
    "text,reason",
    [
        ("Error opening input: Address already in use", "port_busy"),
        ("Error opening output rtmp://127.0.0.1:1/app/key: Connection refused", "refused"),
        ("Failed to resolve hostname nowhere.invalid: Name or service not known", "no_route"),
        ("Server error: NetStream.Publish.BadName", "refused_key"),
        ("Connection timed out", "timeout"),
        ("Error in the push function. av_interleaved_write_frame(): Broken pipe", "broken"),
        ("something new", "unknown"),
    ],
)
def test_ffmpeg_words_become_reasons(text, reason):
    assert link.reason_of(text) == reason


def test_free_port_when_busy():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        busy = s.getsockname()[1]
        assert not link.udp_port_free(busy)
        other = link.free_udp_port(busy)
        assert other != busy and link.udp_port_free(other)
    assert link.listener_port("srt://127.0.0.1:9000?mode=listener") == 9000
    assert link.listener_port("srt://127.0.0.1:9000?mode=caller") is None


def test_link_state_follows_the_events():
    st = engine.LinkState()
    for e in (
        {"event": "waiting_obs", "port": 9000},
        {"event": "obs_connected"},
        {"event": "platform_connecting"},
        {"event": "platform_connected"},
    ):
        st.take(e)
    assert (st.input, st.platform, st.port) == ("connected", "connected", 9000)
    st.take({"event": "platform_lost", "reason": "broken", "retry_s": 2})
    st.take({"event": "platform_connecting"})  # a retry keeps saying "lost" until it is up again
    assert st.platform == "lost" and st.reason == "broken" and st.retry_s == 2
    st.take({"event": "obs_disconnected"})
    st.take({"event": "waiting_obs", "port": 9000})
    assert st.input == "disconnected"
    st.take({"event": "stopped", "reason": "user", "detail": ""})
    assert st.stopped == "user"


def test_dshow_device_list_both_formats():
    new = (
        '[dshow @ 0x1] "Elgato HD60 X" (video)\n[dshow @ 0x1]   Alternative name "@device_pnp_x"\n'
        '[dshow @ 0x1] "Game Capture HD60 X Audio" (audio)\n'
    )
    old = (
        '[dshow @ 0x1] DirectShow video devices\n[dshow @ 0x1]  "OBS Virtual Camera"\n'
        '[dshow @ 0x1] DirectShow audio devices\n[dshow @ 0x1]  "Microphone (USB)"\n'
    )
    assert parse_dshow_devices(new) == {"video": ["Elgato HD60 X"], "audio": ["Game Capture HD60 X Audio"]}
    assert parse_dshow_devices(old) == {"video": ["OBS Virtual Camera"], "audio": ["Microphone (USB)"]}


# --- the platform side, with a fake FFmpeg ------------------------------------------------------------------
class FakeFFmpeg:
    """An FFmpeg output process: takes bytes on stdin, reports progress on stderr, dies when told."""

    def __init__(self, behaviour: str) -> None:
        self.behaviour, self.code = behaviour, None
        self.lines: queue.Queue = queue.Queue()
        self.stdin = self
        self.stderr = iter(self.lines.get, None)
        self.total = 0
        if behaviour == "refuse":
            self.lines.put(b"Error opening output: Server error: NetStream.Publish.BadName\n")
            self.die()

    def write(self, data: bytes) -> int:
        if self.code is not None:
            raise BrokenPipeError
        self.total += len(data)
        self.lines.put(f"total_size={self.total}\n".encode())
        return len(data)

    def flush(self) -> None:
        pass

    def die(self, why: bytes = b"") -> None:
        if why:
            self.lines.put(why)
        self.code = 1
        self.lines.put(None)

    def close(self) -> None:
        if self.code is None:
            self.code = 0
            self.lines.put(None)

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        return self.code

    def kill(self) -> None:
        self.code = -9


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _sink(behaviours: list[str], clock: Clock) -> tuple[FFmpegSink, list[FakeFFmpeg]]:
    made: list[FakeFFmpeg] = []

    def spawn(cmd, **kw):
        made.append(FakeFFmpeg(behaviours[min(len(made), len(behaviours) - 1)]))
        return made[-1]

    return FFmpegSink(f"rtmps://live.twitch.tv:443/app/{KEY}", (64, 36), 30.0, encoder="h264_nvenc", spawn=spawn, clock=clock), made


def _frames(sink: FFmpegSink, clock: Clock, n: int, value: int = 200) -> None:
    frame = np.full((36, 64, 3), value, np.uint8)
    for _ in range(n):
        sink.write(frame, np.zeros((1600, 2), np.int16))
        clock.t += 1 / 30
        time.sleep(0.002)


def test_platform_that_never_takes_the_stream_ends_the_run_with_the_reason(events):
    clock = Clock()
    sink, made = _sink(["refuse"], clock)
    with pytest.raises(PlatformRefused) as err:
        _frames(sink, clock, 30 * 20)
    assert err.value.reason == "refused_key" and len(made) == FFmpegSink.GIVE_UP_BEFORE_FIRST
    names = [e["event"] for e in events]
    assert names.count("platform_lost") == FFmpegSink.GIVE_UP_BEFORE_FIRST - 1 and names[-1] == "platform_failed"
    assert all(KEY not in str(e) for e in events)


def test_platform_drop_mid_stream_reconnects_and_never_sends_an_unprotected_frame(events):
    clock = Clock()
    sink, made = _sink(["ok", "ok"], clock)
    _frames(sink, clock, 30)
    time.sleep(0.05)
    assert sink.connected and [e["event"] for e in events][-1] == "platform_connected"
    made[0].die(b"av_interleaved_write_frame(): Connection reset by peer\n")
    time.sleep(0.05)
    _frames(sink, clock, 15)  # the link is down: these frames are not sent at all
    assert sink.unsent >= 1 and any(e["event"] == "platform_lost" and e["reason"] == "broken" for e in events)
    before = made[0].total
    _frames(sink, clock, 60)  # 1 s backoff, then a new connection
    time.sleep(0.05)
    assert len(made) == 2 and sink.connected and sink.reconnects == 1
    assert made[0].total == before  # nothing went into the dead connection
    sink.close()
    names = [e["event"] for e in events]
    assert names.count("platform_connected") == 2


# --- the OBS side, with real FFmpeg ---------------------------------------------------------------------------
def _udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _send(port: int, seconds: float) -> subprocess.Popen:
    return subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-re", "-f", "lavfi", "-i", f"testsrc2=size=320x180:rate=30:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:v", "libx264", "-preset", "ultrafast", "-g", "30",
         "-c:a", "aac", "-f", "mpegts", f"srt://127.0.0.1:{port}?mode=caller"],
    )  # fmt: skip


@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
def test_obs_can_stop_and_connect_again_without_a_restart(events):
    port = _udp_port()
    src = FFmpegSource(f"srt://127.0.0.1:{port}?mode=listener", (320, 180), 30.0, live=False)
    got: list[int] = []

    def read() -> None:
        for _ in src.frames():
            got.append(src.frame_index)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    time.sleep(0.8)
    _send(port, 2).wait(timeout=30)
    time.sleep(1.5)  # OBS is off for a moment; the listener opens again
    _send(port, 2).wait(timeout=30)
    time.sleep(1.5)
    src.close()
    reader.join(timeout=10)
    assert src.connections == 2 and src.error == ""
    assert got == list(range(len(got))) and len(got) >= 60  # one running number over both connections
    names = [e["event"] for e in events]
    assert names.count("obs_connected") == 2 and "obs_disconnected" in names and names[0] == "waiting_obs"


@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
def test_busy_port_fails_fast_with_the_reason(events):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        src = FFmpegSource(f"srt://127.0.0.1:{port}?mode=listener", (320, 180), 30.0, live=False)
        t0 = time.monotonic()
        assert list(src.frames()) == []
        src.close()
    assert src.error == "port_busy" and time.monotonic() - t0 < 10
    assert any(e["event"] == "input_failed" and e["reason"] == "port_busy" for e in events)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
def test_real_refused_platform(events):
    sink = FFmpegSink(f"rtmp://127.0.0.1:1/app/{KEY}", (64, 36), 30.0, encoder="libx264")
    sink.RETRY_S = (0.1, 0.1, 0.1)
    frame = np.zeros((36, 64, 3), np.uint8)
    with pytest.raises(PlatformRefused) as err:
        for _ in range(30 * 30):
            sink.write(frame, None)
            time.sleep(1 / 30)
    assert err.value.reason == "refused" and KEY not in str(err.value)


def test_capture_buffer_restores():
    out = io.StringIO()
    link.capture(lambda line: out.write(line + "\n"))
    link.status("waiting_obs", port=1)
    link.capture(None)
    assert link.parse(out.getvalue().strip()) == {"event": "waiting_obs", "port": 1}
