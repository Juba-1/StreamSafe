"""The two network links of a live run, said out loud: OBS -> us (SRT) and us -> the platform (RTMP / RTMPS).

The engine prints one ``STATUS {...}`` line whenever a link changes; the window turns them into the words the
streamer reads ("waiting for OBS on port 9000", "OBS connected", "connected to the platform", why it stopped).
Events (``event`` field):

  waiting_obs          port          the SRT listener is up and nobody has connected yet
  obs_connected                      OBS (or any sender) is sending picture
  obs_disconnected                   the sender went away; the listener is opened again at once
  input_failed         reason        the input could not be opened at all (port busy, no such device)
  platform_connecting                the first protected frame is ready: connecting to the platform now
  platform_connected                 the platform is taking our stream
  platform_lost        reason, retry_s   the connection broke; a new one is tried after ``retry_s`` seconds
  platform_failed      reason        the platform never accepted the stream (wrong address or key): the run ends
  recording            file          the protected picture is being written to this file
  recorded             file          the finished recording
  stopped              reason, detail    the run is over: user / input_ended / input_failed / platform_failed / error

Nothing here ever holds a stream key: ``redact_url`` is applied to every address and FFmpeg line before it is
printed (``Secrets``), and the window blanks the key once more on its side (``app/engine.py``).
"""

from __future__ import annotations

import contextlib
import json
import re
import socket
import threading
from collections.abc import Callable

PREFIX = "STATUS "
MASK = "***"

_lock = threading.Lock()
_sink: Callable[[str], None] | None = None  # tests: where the lines go instead of standard output


def status(event: str, **fields: object) -> None:
    """Print one status line (flushed at once: the window reads them as they come)."""
    line = PREFIX + json.dumps({"event": event, **fields}, ensure_ascii=False)
    with _lock:
        if _sink is not None:
            _sink(line)
        else:
            with contextlib.suppress(OSError, ValueError):
                print(line, flush=True)


def capture(to: Callable[[str], None] | None) -> None:
    """Tests: send status lines to ``to`` (None: back to standard output)."""
    global _sink
    _sink = to


def parse(line: str) -> dict | None:
    """The event of a ``STATUS`` line, None for any other line."""
    if not line.startswith(PREFIX):
        return None
    try:
        data = json.loads(line[len(PREFIX) :])
    except ValueError:
        return None
    return data if isinstance(data, dict) and isinstance(data.get("event"), str) else None


class Secrets:
    """Blanks an output address and its last part (the stream key) out of any text."""

    def __init__(self, url: str = "") -> None:
        parts: set[str] = set()
        if "://" in url:
            parts.add(url)
            rest = url.split("://", 1)[1]
            path = rest.split("/", 1)[1] if "/" in rest else ""
            key = path.rstrip("/").rsplit("/", 1)[-1]
            for piece in (key, key.split("?", 1)[0], *re.findall(r"[?&][^=&]+=([^&]+)", key)):
                if len(piece) >= 4:
                    parts.add(piece)
        self._parts = sorted(parts, key=len, reverse=True)

    def clean(self, text: str) -> str:
        for part in self._parts:
            text = text.replace(part, MASK)
        return text


def redact_url(url: str) -> str:
    """``rtmps://host/app/<key>`` -> ``rtmps://host/app/***`` (what may be shown or logged)."""
    if "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    host, _, path = rest.partition("/")
    host = host.rsplit("@", 1)[-1]  # user:password@host
    if not path.strip("/"):
        return f"{scheme}://{host}"
    head = path.rstrip("/").rsplit("/", 1)[0] if "/" in path.rstrip("/") else ""
    return f"{scheme}://{host}/{head + '/' if head else ''}{MASK}"


def udp_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Can an SRT listener (UDP) be opened on ``port`` right now?"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((host, port))
        return True
    except OSError:
        return False


def free_udp_port(preferred: int, host: str = "127.0.0.1", tries: int = 50) -> int:
    """``preferred`` when it is free, else the next free port above it, else any free port the system gives."""
    for port in range(preferred, min(preferred + tries, 65536)):
        if port >= 1024 and udp_port_free(port, host):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def listener_port(url: str) -> int | None:
    """The port of an ``srt://...mode=listener`` input, None for anything else."""
    m = re.match(r"^srt://[^/?]*:(\d+)", url)
    return int(m.group(1)) if m and "mode=listener" in url else None


# FFmpeg's own words -> a short reason the window can translate (``app/i18n.py``: ``why_<reason>``)
_REASONS = (
    ("port_busy", ("address already in use", "only one usage of each socket", "error number -10048")),
    ("no_device", ("could not find video device", "could not find audio", "i/o error opening", "cannot open capture device")),
    ("refused_key", ("netstream.publish", "publish.badname", "unauthorized", "authentication", "error number -10053", "403 forbidden")),
    ("no_route", ("failed to resolve", "name or service not known", "no such host", "error number -11001", "network is unreachable")),
    ("refused", ("connection refused", "error number -10061", "cannot open connection")),
    ("timeout", ("timed out", "error number -10060", "error number -138")),
    ("tls", ("tls", "ssl", "certificate", "schannel")),
    ("broken", ("broken pipe", "connection reset", "error number -10054", "end of file", "i/o error", "input/output error")),
)


def reason_of(text: str, default: str = "unknown") -> str:
    low = text.lower()
    return next((name for name, words in _REASONS if any(w in low for w in words)), default)
