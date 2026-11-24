"""The web window's local server: token, state, actions, files."""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from streamsafe.app.webui import server as web_server


@pytest.fixture
def web(tmp_path, monkeypatch):
    for var in ("LOCALAPPDATA", "APPDATA", "STREAMSAFE_DATA"):
        monkeypatch.setenv(var, str(tmp_path))
    monkeypatch.setattr(web_server.WebApp, "_first_checks", lambda self: None)
    app = web_server.WebApp()
    srv = web_server.ThreadingHTTPServer(("127.0.0.1", 0), web_server.make_handler(app, "tok", [time.monotonic()]))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield app, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def call(base, path, body=None, token="tok"):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"X-Token": token, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.read()


def test_needs_the_token(web):
    _, base = web
    with pytest.raises(urllib.error.HTTPError) as e:
        call(base, "/api/state", token="wrong")
    assert e.value.code == 403


def test_state_and_live_controls(web):
    _, base = web
    state = json.loads(call(base, "/api/state"))
    assert state["status"]["key"] == "stopped" and state["running"] is False
    call(base, "/api/sens", {"k": 9})  # clamped
    call(base, "/api/everyone", {"on": True})
    call(base, "/api/cover_mode", {"mode": "screen"})
    call(base, "/api/zones", {"zones": [[0.7, 0.6, 0.98, 0.95]]})
    call(base, "/api/panic", {"on": True})
    s = json.loads(call(base, "/api/state"))
    assert s["settings"]["sens"] == web_server.cfg.SENS_MAX and s["settings"]["everyone"] and s["settings"]["cover_mode"] == "screen"
    assert s["settings"]["zones"] == [[0.7, 0.6, 0.98, 0.95]] and s["panic"]
    assert json.loads(call(base, "/api/nothing", {}))["ok"] is False


def test_page_and_files(web):
    _, base = web
    page = call(base, "/?t=tok")
    assert b"__TOKEN__" not in page and b"tok" in page
    assert call(base, "/static/app.js").startswith(b"//")
    with pytest.raises(urllib.error.HTTPError) as e:
        call(base, "/static/..%2fserver.py")
    assert e.value.code == 404
