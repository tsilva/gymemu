import base64
import io
import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import torch
from PIL import Image

from gymemu.player import Player, handle_key
from gymemu.replay import load_replay
from gymemu.web_player import PlaybackSession, make_server


class Spy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, history, action):
        self.calls.append((history.clone(), action.clone()))
        return torch.full((1, 3, 21, 17), len(self.calls) / 10)


def config():
    return {"history": 4, "shape": [3, 21, 17], "action_values": [0, 1, 2]}


def session():
    return PlaybackSession(
        Player(Spy(), config(), torch.device("cpu")),
        "model.pt",
        {"left": 2, "right": 1, "space": 0},
    )


def test_shared_keyboard_fresh_presses_and_held_priority():
    s = session()
    p = s.player
    for key in ("x", "left"):
        handle_key(p, key, s.keymap, repeat=True)
    assert not p.model.calls
    handle_key(p, "left", s.keymap)
    handle_key(p, "left", s.keymap)  # Key remains held: no extra single steps.
    assert len(p.model.calls) == 1 and p.steps == 0
    assert p.model.calls[0][1].item() == 3  # Bootstrap, no game action.
    handle_key(p, "left", s.keymap, down=False)
    handle_key(p, "left", s.keymap)
    assert p.steps == 1 and p.last_action == 2
    s.control({"type": "play"})
    handle_key(p, "right", s.keymap)
    before = len(p.model.calls)
    assert s.tick() and p.last_action == 1
    assert len(p.model.calls) == before + 1
    handle_key(p, "right", s.keymap, down=False)
    s.tick()
    assert p.last_action == 2
    handle_key(p, "left", s.keymap, down=False)
    s.tick()
    assert p.last_action == 0
    for kind in ("pause", "blur", "reset", "next"):
        s.control({"type": "play"})
        s.control({"type": kind})
        assert not p.continuous and not p.held_keys
    s.control({"type": "play"})
    s.tick(now=s.last_heartbeat + 2)
    assert not p.continuous  # Lost browser must not leave playback running.


def test_replay_seek_select_atomic_images_and_history(snapshot):
    player = load_replay(Spy(), config(), torch.device("cpu"), dataset=str(snapshot), split="train")
    s = PlaybackSession(player, "model.pt", {})
    assert not player.model.calls
    s.control({"type": "seek", "position": 2})
    assert player.steps == 2 and player.finished and not player.continuous
    history, action = player.model.calls[-1]
    assert history[0, -1].eq(20 / 255).all() and action.item() == 0
    s.publish()
    current = s.read()
    assert current["step"] == current["history"][-1]["step"] == 2
    pixels = Image.open(io.BytesIO(base64.b64decode(current["images"]["original"].split(",")[1])))
    assert np.all(np.asarray(pixels) == 30)
    s.control({"type": "play"})
    assert not player.continuous
    s.control({"type": "select", "value": 3})
    assert player.steps == 0 and player.episode.episode_id == 3 and not s.history
    s.control({"type": "step", "action": 999})
    assert player.steps == 1  # Uses the recorded action.
    s.control({"type": "seek", "position": 0})
    assert player.frame is None and not s.history
    for command in (
        {"type": "seek", "position": -1},
        {"type": "seek", "position": 2},
        {"type": "select", "value": 999},
        {"type": "key", "key": []},
    ):
        with pytest.raises(ValueError):
            s.control(command)


@pytest.fixture
def running_server(tmp_path):
    s = session().start()
    server, url = make_server(s, workspace_path=tmp_path / "workspace.json")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield s, url
    server.shutdown()
    server.server_close()
    s.close()
    thread.join(timeout=3)


def api(url, path, data=None, **headers):
    root, token = url.split("#token=")
    request = Request(
        root.rstrip("/") + path,
        data=None if data is None else json.dumps(data).encode(),
        headers={"X-Player-Token": token, "Content-Type": "application/json", **headers},
    )
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def test_http_commands_security_assets_and_workspace(running_server):
    s, url = running_server
    first = api(url, "/api/state")
    assert first["step"] == 0 and not first["has_prediction"] and not s.player.model.calls
    api(url, "/api/command", {"type": "step", "action": 2})
    api(url, "/api/command", {"type": "step", "action": 1})
    second = api(url, "/api/state")
    assert second["step"] == 1 and second["action"] == 1 and second["revision"] > first["revision"]
    assert api(url, "/api/workspace") is None
    layout = {"version": 1, "panels": {"prediction": {"title": "My prediction"}}}
    api(url, "/api/workspace", layout)
    assert api(url, "/api/workspace") == layout
    for headers in (
        {"X-Player-Token": "wrong"},
        {"Origin": "https://evil.example"},
        {"Host": "evil.example"},
    ):
        with pytest.raises(HTTPError) as error:
            api(url, "/api/command", {"type": "play"}, **headers)
        assert error.value.code == 403
    with pytest.raises(HTTPError) as error:
        api(url, "/api/command", {"type": "execute_python"})
    assert error.value.code == 400 and not s.player.continuous
    root = url.split("#")[0]
    for path in (
        "",
        "assets/app.js",
        "assets/panels/runtime.js",
        "assets/vendor/gridstack/gridstack-all.js",
    ):
        with urlopen(root + path, timeout=5) as response:
            assert response.status == 200
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    with pytest.raises(HTTPError):
        urlopen(root + "assets/../web_player.py", timeout=5)


def test_worker_pacing_and_disconnect_pause(running_server):
    s, url = running_server
    api(url, "/api/command", {"type": "play"})
    deadline = time.monotonic() + 2
    while len(s.player.model.calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    api(url, "/api/command", {"type": "pause"})
    calls = len(s.player.model.calls)
    assert 2 <= calls < 40
    time.sleep(0.08)
    assert len(s.player.model.calls) == calls


@pytest.mark.parametrize("startup", ["default", "explicit", "empty"])
def test_cli_opens_web_without_inference(monkeypatch, tmp_path, startup):
    import gymemu.web_player as web
    import play

    spy = Spy()
    monkeypatch.setattr(play, "load_model", lambda *_: (spy, config()))
    seen = []
    monkeypatch.setattr(web, "serve", lambda player, **options: seen.append((player, options)))
    argv = [str(tmp_path / "model.pt"), "--device", "cpu", "--no-browser"]
    if startup == "empty":
        argv.append("--empty-start")
    else:
        path = tmp_path / ("start-scene.npz" if startup == "default" else "alternate.npz")
        np.savez_compressed(path, frames=np.full((2, 3, 21, 17), 50, np.uint8))
        if startup == "explicit":
            argv.extend(["--start-scene", str(path)])
    play.main(argv)
    player, options = seen[0]
    assert not spy.calls and not player.continuous
    assert options["port"] == 0 and not options["open_browser"]
    if startup != "empty":
        assert player.frame.eq(50 / 255).all()


def test_modes_reset_context_and_reuse_loaded_datasets(snapshot):
    p = Player(Spy(), config(), torch.device("cpu"))
    calls = []

    def factory(mode):
        calls.append(mode)
        return load_replay(p.model, config(), torch.device("cpu"), dataset=str(snapshot))

    s = PlaybackSession(p, "model.pt", {"left": 2}, mode_factory=factory)
    s.control({"type": "step", "action": 2})
    s.control({"type": "mode", "mode": "teacher-forcing"})
    assert s.replay and s.player.steps == 0 and s.keymap == {"space": None}
    s.control({"type": "step"})
    assert s.player.mse is not None
    s.control({"type": "mode", "mode": "autoregressive"})
    assert s.player is p and not s.replay and not p.continuous and p.frame is None
    assert not s.history and s.keymap == {"left": 2}
    s.control({"type": "mode", "mode": "teacher-forcing"})
    assert calls == ["teacher-forcing"] and s.player.frame is None


def test_slow_mode_loading_keeps_transport_and_lease_responsive(running_server, snapshot):
    s, url = running_server
    entered, release = threading.Event(), threading.Event()

    def factory(mode):
        entered.set()
        assert release.wait(timeout=5)
        return load_replay(Spy(), config(), torch.device("cpu"), dataset=str(snapshot))

    s.mode_factory = factory
    try:
        api(url, "/api/command", {"type": "mode", "mode": "teacher-forcing"})
        assert entered.wait(timeout=1)
        loading = api(url, "/api/state")
        assert loading["loading"] == "teacher-forcing" and not loading["playing"]
        api(url, "/api/command", {"type": "heartbeat"})  # Completes before loading is released.
    finally:
        release.set()
    with s.changed:
        assert s.changed.wait_for(lambda: s.current["mode"] == "teacher-forcing", timeout=3)
    assert s.current["loading"] is None and not s.current["playing"]
