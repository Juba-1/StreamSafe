"""Where the stream goes: the platforms' ingest addresses and what a streamer may paste into the two boxes.

A platform shows a "server" and a "stream key". The two are joined as ``<server>/<key>``, and the server must
end in the platform's application name - FFmpeg reads the LAST part of the path as the key and what is before
it as the application. Kick (Amazon IVS) shows its server WITHOUT ``/app``:
``rtmps://<id>.global-contribute.live-video.net`` joined with the key gives the application "<key>" and an empty
stream name, which IVS refuses - the stream never starts (field report, 16 Nov 2026). So every address is
normalised here before it is used:

  * spaces and line breaks around and inside a pasted address or key are removed;
  * a known host without its application gets it (IVS / Kick and Twitch ``/app``, YouTube ``/live2``, Facebook
    ``/rtmp``), and ``rtmps`` to IVS gets its port 443;
  * a whole address with the key at its end, pasted into either box, is split into server and key;
  * anything else that cannot work says why (``problems``) instead of failing later at "go live".

The key is only ever returned to the caller: nothing here logs, prints or stores it.
"""

from __future__ import annotations

from dataclasses import dataclass

KICK, TWITCH, YOUTUBE, TIKTOK, FACEBOOK, X, TROVO, CUSTOM = "kick", "twitch", "youtube", "tiktok", "facebook", "x", "trovo", "custom"
PLATFORMS = (KICK, TWITCH, YOUTUBE, TIKTOK, FACEBOOK, X, TROVO, CUSTOM)

# the server each platform publishes for every streamer ("" = it is different per streamer: paste it).
# Sources: OBS Studio's own service list (plugins/rtmp-services/data/services.json, read 17 Nov 2026) and, for
# Kick, a live test (runs/live_kick_r1). Twitch has no single address in that list (it picks the nearest of its
# Amazon IVS ingests); Frankfurt is the nearest listed one to the Gulf.
PRESETS: dict[str, str] = {
    KICK: "rtmps://fa723fc1b171.global-contribute.live-video.net:443/app",
    TWITCH: "rtmp://euc10.contribute.live-video.net/app",
    YOUTUBE: "rtmps://a.rtmps.youtube.com:443/live2",
    TIKTOK: "",  # LIVE Producer shows a new server and key for every stream
    FACEBOOK: "rtmps://rtmp-api.facebook.com:443/rtmp",
    X: "",  # X (Twitter) Media Studio shows the server
    TROVO: "rtmp://livepush.trovo.live/live",
    CUSTOM: "",
}

# host ending -> the application its ingest needs (when a streamer pastes the server without it)
_KNOWN_APPS = (
    (".live-video.net", "app"),  # Amazon IVS: Kick and Twitch
    ("twitch.tv", "app"),
    ("youtube.com", "live2"),
    ("facebook.com", "rtmp"),
    ("fbcdn.net", "rtmp"),
    ("pscp.tv", "x"),  # X (Twitter)
    ("trovo.live", "live"),
    ("restream.io", "live"),
    ("steamcontent.com", "app"),
    ("nimo.tv", "live"),
    ("bilivideo.com", "live-bvc"),
    ("vimeo.com", "live"),
)

# kbit/s a platform takes for 1080p at 30 / 60 fps (what OBS recommends for it, read 17 Nov 2026; Kick: its own
# page). Other sizes scale from these (bitrate_for).
_KBPS_1080 = {
    KICK: (6000, 8000),
    TWITCH: (6000, 6000),
    YOUTUBE: (10000, 12000),
    TIKTOK: (6000, 6000),
    FACEBOOK: (6000, 9000),
    X: (9000, 12000),
    TROVO: (6000, 9000),
}
_SCHEMES = ("rtmp://", "rtmps://")


@dataclass(frozen=True)
class Target:
    server: str  # the address without the key, normalised ("" when it cannot be used)
    key: str
    problems: tuple[str, ...] = ()  # i18n keys: bad_rtmp / bad_rtmp_app / bad_key

    @property
    def url(self) -> str:
        """The full output address. Holds the key: never log or show it."""
        return f"{self.server}/{self.key}" if self.server and self.key else ""


def squeeze(text: str) -> str:
    """A pasted value without the spaces, tabs, line breaks and invisible marks around and inside it."""
    return "".join(c for c in str(text) if not c.isspace() and c not in "‎‏‪‫‬﻿​")


def _split(url: str) -> tuple[str, str, list[str], str]:
    """(scheme with ://, host[:port], path parts, query of the last part)."""
    scheme = next((s for s in _SCHEMES if url.lower().startswith(s)), "")
    rest = url[len(scheme) :]
    host, _, path = rest.partition("/")
    path, _, query = path.partition("?")
    return scheme, host, [p for p in path.split("/") if p], query


def known_app(host: str) -> str:
    name = host.rsplit("@", 1)[-1].split(":", 1)[0].lower()
    return next((app for end, app in _KNOWN_APPS if name.endswith(end)), "")


def is_ivs(host: str) -> bool:
    return host.split(":", 1)[0].lower().endswith(".live-video.net")


def normalise(server: str) -> str:
    """A server address the way FFmpeg needs it; "" when it is not an rtmp(s) address with a host."""
    url = squeeze(server)
    scheme, host, parts, query = _split(url)
    if not scheme or not host or any(c in url for c in '"<>'):
        return ""
    scheme = scheme.lower()
    app = known_app(host)
    if app and not parts:
        parts = [app]
    if is_ivs(host) and scheme == "rtmps://" and ":" not in host:
        host += ":443"
    out = scheme + host + ("/" + "/".join(parts) if parts else "")
    return out + ("?" + query if query else "")


def resolve(server: str, key: str = "", platform: str = CUSTOM) -> Target:
    """What the two boxes mean. ``server`` may be empty (the platform's preset is used), a server, or a whole
    address ending in the key; ``key`` may be the key or, by mistake, the whole address."""
    server, key = squeeze(server), squeeze(key)
    if key.lower().startswith(_SCHEMES):  # the whole address in the key box
        server, key = key, ""
    if not server:
        server = PRESETS.get(platform, "")
    scheme, host, parts, query = _split(server)
    if not scheme or not host:
        return Target("", key, ("bad_rtmp",))
    app = known_app(host)
    if not key:  # is the key at the end of the address?
        if app and parts and parts[0].lower() != app:  # "rtmps://<ivs host>/<key>": the application is missing
            key, parts = "/".join(parts), []
        elif len(parts) >= 2:
            key, parts = parts[-1], parts[:-1]
        if key and query:
            key, query = f"{key}?{query}", ""
    elif parts and parts[-1] == key:
        parts = parts[:-1]  # the whole address in the server box AND the key in its own box
    clean = normalise(scheme + host + ("/" + "/".join(parts) if parts else "") + ("?" + query if query else ""))
    problems = []
    if not clean:
        problems.append("bad_rtmp")
    elif not _split(clean)[2]:
        problems.append("bad_rtmp_app")  # no application in the address: the platform would refuse it
    if not valid_key(key):
        problems.append("bad_key")
    return Target(clean if "bad_rtmp" not in problems else "", key, tuple(problems))


def valid_key(key: str) -> bool:
    return bool(key) and len(key) <= 1024 and not any(c.isspace() or c in '"<>/\\' for c in key)


def guess_platform(server: str) -> str:
    host = _split(squeeze(server))[1].lower()
    for name, words in ((KICK, ("global-contribute.live-video.net", "kick.com")), (TWITCH, ("contribute.live-video.net", "twitch.tv")),
                        (YOUTUBE, ("youtube.com",)), (TIKTOK, ("tiktok",)), (FACEBOOK, ("facebook.com", "fbcdn.net")),
                        (X, ("pscp.tv", "x.com")), (TROVO, ("trovo",))):  # fmt: skip
        if any(w in host for w in words):
            return name
    return CUSTOM


def bitrate_for(platform: str, height: int, fps: float) -> int:
    """The video bitrate (kbit/s) to send: the platform's own figure for that size and frame rate - higher than
    it takes is refused or re-compressed by the platform, lower wastes quality."""
    from streamsafe.io.sinks import auto_kbps

    if platform not in _KBPS_1080:
        return auto_kbps((height * 16 // 9, height), fps, True)
    k30, k60 = _KBPS_1080[platform]
    base = k60 if fps > 30 else k30
    if height <= 720:
        return round(base * 0.65)
    if height <= 1080 or platform != YOUTUBE:  # only YouTube takes more than 1080p
        return base
    return {1440: 24000 if fps <= 30 else 30000}.get(height, 35000 if fps <= 30 else 51000)
