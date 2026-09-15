"""Local browser transport. One worker owns inference; HTTP only exchanges snapshots."""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import queue
import secrets
import threading
import time
import webbrowser
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image

from gymemu.player import PLAYBACK_FPS, handle_key
from gymemu.replay import ReplayPlayer

ASSETS = Path(__file__).with_name("web_assets")


def workspace_revision(value):
    revision = value.get("revision", {}) if isinstance(value, dict) else {}
    if not isinstance(revision, dict):
        return (0, "")
    clock, writer = revision.get("clock"), revision.get("writer")
    if type(clock) is int and clock >= 0 and isinstance(writer, str):
        return (clock, writer)
    return (0, "")


def dashboard_urls(player_url):
    url = urlsplit(player_url)
    return player_url, url._replace(path="/workspace/stats").geturl()


def png(pixels):
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class PlaybackSession:
    """Serialized controls, paced inference, and atomic frame/metadata snapshots."""

    def __init__(self, player, checkpoint, keymap, *, scale=3, mode_factory=None):
        self.player, self.checkpoint = player, str(checkpoint)
        self.replay = isinstance(player, ReplayPlayer)
        self.normal_keymap = keymap
        self.mode_factory = mode_factory
        self.players = {"teacher-forcing" if self.replay else "autoregressive": player}
        self.keymap = {"space": None} if self.replay else keymap
        self.scale = scale
        self.revision = 0
        self.error = None
        self.loading = None
        self.history = {}
        self.history_epoch = 0
        self.last_heartbeat = time.monotonic()
        self.inference_ms = None
        self.commands = queue.Queue(maxsize=64)
        self.changed = threading.Condition()
        self.closed = threading.Event()
        self.current = None
        self.worker = None

    def pause(self):
        self.player.continuous = False
        self.player.held_keys.clear()

    def clear_history(self):
        self.history.clear()
        self.history_epoch += 1

    def record_mse(self):
        if self.replay and self.player.mse is not None:
            self.history[self.player.steps] = {"step": self.player.steps, "mse": self.player.mse}

    def advance(self, action=None):
        if self.replay and self.player.finished:
            self.pause()
            return
        start = time.perf_counter()
        if self.replay:
            self.player.advance()
        else:
            self.player.advance(self.default_action if action is None else action)
        self.inference_ms = (time.perf_counter() - start) * 1000
        self.record_mse()

    @property
    def default_action(self):
        return 0 if 0 in self.player.actions else self.player.actions[0]

    def control(self, command):
        kind = command.get("type")
        if kind == "heartbeat":
            self.last_heartbeat = time.monotonic()
            return False
        if kind == "mode":
            self.pause()
            mode = command.get("mode")
            if mode not in ("teacher-forcing", "autoregressive") or self.mode_factory is None:
                raise ValueError("Playback mode is unavailable")
            if mode not in self.players:
                self.loading = mode
                self.publish()
                try:
                    self.players[mode] = self.mode_factory(mode)
                finally:
                    self.loading = None
            self.player = self.players[mode]
            self.replay = isinstance(self.player, ReplayPlayer)
            self.keymap = {"space": None} if self.replay else self.normal_keymap
            self.player.reset()
            self.clear_history()
            self.inference_ms = None
        elif kind in ("pause", "blur"):
            self.pause()
        elif kind == "play":
            self.player.continuous = not (self.replay and self.player.finished)
            self.last_heartbeat = time.monotonic()
        elif kind == "step":
            self.pause()
            self.advance(command.get("action"))
        elif kind in ("reset", "next"):
            self.player.reset(cycle=kind == "next")
            self.clear_history()
            self.inference_ms = None
        elif kind == "select":
            self.pause()
            value = command.get("value")
            if type(value) is not int:
                raise ValueError("Selection must be an integer")
            if self.replay:
                matches = [
                    i for i, e in enumerate(self.player.windows.episodes) if e.episode_id == value
                ]
                if not matches:
                    raise ValueError("Unknown episode")
                self.player.episode_index = matches[0]
            else:
                if not 0 <= value < len(self.player.start_states):
                    raise ValueError("Unknown starting scene")
                self.player.start_index = value
                self.player._set_initial_history(self.player.start_states[value][1])
            self.player.reset()
            self.clear_history()
            self.inference_ms = None
        elif kind == "seek":
            if not self.replay:
                raise ValueError("Seeking requires teacher-forced dataset replay")
            if "episode_id" in command and command["episode_id"] != self.player.episode.episode_id:
                raise ValueError("The chart episode has changed")
            if "history_epoch" in command and command["history_epoch"] != self.history_epoch:
                raise ValueError("The chart history has changed")
            position = command.get("position")
            if type(position) is not int or not 0 <= position <= len(self.player.episode.actions):
                raise ValueError("Frame position is outside this episode")
            self.pause()
            self.player.reset()
            self.inference_ms = None
            if position:
                self.player.steps = position - 1
                self.advance()
        elif kind == "key":
            key = command.get("key")
            if not isinstance(key, str) or len(key) > 32:
                raise ValueError("Invalid key")
            down, repeat = command.get("down", True), command.get("repeat", False)
            if type(down) is not bool or type(repeat) is not bool:
                raise ValueError("Invalid keyboard event")
            before = self.player.steps
            previous_frame = self.player.frame
            started = time.perf_counter()
            handle_key(self.player, key, self.keymap, down=down, repeat=repeat)
            if key in self.keymap and self.player.frame is not previous_frame:
                self.inference_ms = (time.perf_counter() - started) * 1000
            if key in ("r", "c") and down and not repeat:
                self.clear_history()
                self.inference_ms = None
            if self.replay and self.player.steps != before and self.player.mse is not None:
                self.record_mse()
            self.last_heartbeat = time.monotonic()
        else:
            raise ValueError(f"Unknown playback command: {kind}")
        self.error = None
        return True

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        if not self.player.continuous:
            return False
        if now - self.last_heartbeat > 1.5:
            self.pause()
            return True
        action = (
            self.keymap[self.player.held_keys[-1]] if self.player.held_keys else self.default_action
        )
        self.advance(action)
        return True

    def snapshot(self):
        player = self.player
        pixels = player.pixels()
        images = {}
        if self.replay:
            for name, tile in zip(
                ("prediction", "original", "difference"), np.split(pixels, 3, axis=1)
            ):
                images[name] = png(tile)
        else:
            images["prediction"] = png(pixels)
        stack = player.input_stack.permute(0, 2, 3, 1).mul(255).round().byte().numpy()
        images["history"] = png(np.concatenate(list(stack), axis=1))
        choices = (
            [
                {
                    "value": e.episode_id,
                    "label": f"Episode {e.episode_id}",
                    "length": len(e.actions),
                }
                for e in player.windows.episodes
            ]
            if self.replay
            else [{"value": i, "label": name} for i, (name, _) in enumerate(player.start_states)]
        )
        return {
            "revision": self.revision,
            "mode": "teacher-forcing" if self.replay else "autoregressive",
            "available_modes": ["autoregressive", "teacher-forcing"] if self.mode_factory else [],
            "checkpoint": self.checkpoint,
            "name": player.start_name or "Empty start",
            "playing": player.continuous,
            "step": player.steps,
            "total_steps": len(player.episode.actions) if self.replay else None,
            "selection": player.episode.episode_id if self.replay else player.start_index,
            "choices": choices,
            "has_prediction": player.has_prediction,
            "finished": self.replay and player.finished,
            "action": player.recorded_action if self.replay else (player.last_action),
            "action_values": player.actions,
            "keymap": self.keymap,
            "mse": player.mse if self.replay else None,
            "inference_ms": self.inference_ms,
            "history": [self.history[step] for step in sorted(self.history)],
            "history_epoch": self.history_epoch,
            "history_length": player.config["history"],
            "shape": player.config["shape"],
            "scale": self.scale,
            "action_history": player.config.get("action_history", 1),
            "state_fields": player.config.get("state_fields", []),
            "loading": self.loading,
            "images": images,
            "error": self.error,
        }

    def publish(self):
        self.revision += 1
        snapshot = self.snapshot()
        with self.changed:
            self.current = snapshot
            self.changed.notify_all()

    def start(self):
        self.publish()
        self.worker = threading.Thread(target=self.run, name="gymemu-inference", daemon=True)
        self.worker.start()
        return self

    def run(self):
        deadline = time.monotonic() + 1 / PLAYBACK_FPS
        while not self.closed.is_set():
            pending = None
            try:
                timeout = max(0, deadline - time.monotonic()) if self.player.continuous else 0.2
                command, pending = self.commands.get(timeout=timeout)
                changed = self.control(command)
                if changed:
                    self.publish()
                pending.set_result(None)
            except queue.Empty:
                try:
                    if self.tick():
                        self.publish()
                except Exception as error:
                    self.pause()
                    self.error = str(error)
                    self.publish()
                # Inference never catches up with a burst of missed frames.
                deadline = time.monotonic() + 1 / PLAYBACK_FPS
            except Exception as error:
                self.pause()
                self.error = str(error)
                self.publish()
                if pending is not None:
                    pending.set_exception(error)
            if not self.player.continuous:
                deadline = time.monotonic() + 1 / PLAYBACK_FPS

    def submit(self, command):
        if self.closed.is_set():
            raise ValueError("Playback has stopped")
        if command.get("type") == "heartbeat":
            # A lease refresh must remain responsive while a dataset is loading.
            self.last_heartbeat = time.monotonic()
            return
        future = Future()
        self.commands.put_nowait((command, future))
        if command.get("type") != "mode":
            future.result(timeout=30)
        # Mode loading is acknowledged immediately; progress/errors arrive in snapshots.

    def read(self, after=-1):
        with self.changed:
            self.changed.wait_for(
                lambda: self.current["revision"] > after or self.closed.is_set(), timeout=1
            )
            return self.current

    def close(self):
        self.closed.set()
        if self.worker:
            self.worker.join(timeout=5)
        self.pause()
        with self.changed:
            self.changed.notify_all()


def make_server(session, port=0, *, workspace_path=None):
    token = secrets.token_urlsafe(32)
    workspace_path = workspace_path or Path.home() / ".config/gymemu/player-workspace.json"
    workspace_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, code, body, mime="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data: blob:; "
                "style-src 'self' 'unsafe-inline'; font-src 'self'; "
                "connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def authorized(self):
            expected = f"127.0.0.1:{self.server.server_port}"
            return (
                self.headers.get("Host") == expected
                and secrets.compare_digest(self.headers.get("X-Player-Token", ""), token)
                and self.headers.get("Origin", f"http://{expected}") == f"http://{expected}"
            )

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/api/workspace":
                if not self.authorized():
                    return self.respond(403, b'{"error":"Unauthorized player request"}')
                with workspace_lock:
                    try:
                        value = json.loads(workspace_path.read_text())
                    except (OSError, ValueError):
                        value = None
                return self.respond(200, json.dumps(value).encode())
            if url.path == "/api/state":
                if not self.authorized():
                    return self.respond(403, b'{"error":"Unauthorized player request"}')
                try:
                    after = int(parse_qs(url.query).get("after", [-1])[0])
                except ValueError:
                    return self.respond(400, b'{"error":"Invalid revision"}')
                return self.respond(200, json.dumps(session.read(after), allow_nan=False).encode())
            dashboard = url.path in ("/", "/workspace/stats")
            path = ASSETS / ("index.html" if dashboard else url.path.removeprefix("/assets/"))
            if not dashboard and not url.path.startswith("/assets/"):
                return self.respond(404, b"Not found", "text/plain")
            if not path.resolve().is_relative_to(ASSETS.resolve()) or not path.is_file():
                return self.respond(404, b"Not found", "text/plain")
            return self.respond(
                200, path.read_bytes(), mimetypes.guess_type(path)[0] or "application/octet-stream"
            )

        def do_POST(self):
            if self.path not in ("/api/command", "/api/workspace"):
                return self.respond(404, b'{"error":"Not found"}')
            if not self.authorized():
                return self.respond(403, b'{"error":"Unauthorized player request"}')
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= (65536 if self.path == "/api/workspace" else 4096):
                    raise ValueError("Invalid command size")
                command = json.loads(self.rfile.read(length))
                if not isinstance(command, dict):
                    raise ValueError("Expected a playback command")
                if self.path == "/api/workspace":
                    if command.get("version") not in (1, 2, 3) or not isinstance(
                        command.get("panels"), dict
                    ):
                        raise ValueError("Invalid workspace")
                    if len(command["panels"]) > 40:
                        raise ValueError("Workspace supports at most 40 widgets")
                    with workspace_lock:
                        try:
                            saved = json.loads(workspace_path.read_text())
                        except (OSError, ValueError):
                            saved = None
                        if workspace_revision(command) < workspace_revision(saved):
                            return self.respond(200, b'{"ok":true}')
                        workspace_path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = workspace_path.with_suffix(".tmp")
                        temporary.write_text(json.dumps(command))
                        temporary.replace(workspace_path)
                else:
                    session.submit(command)
            except (ValueError, TypeError, OSError, queue.Full, TimeoutError) as error:
                return self.respond(400, json.dumps({"error": str(error)}).encode())
            return self.respond(200, b'{"ok":true}')

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server, f"http://127.0.0.1:{server.server_port}/#token={token}"


def serve(player, *, checkpoint, keymap, port=0, open_browser=True, scale=3, mode_factory=None):
    session = PlaybackSession(
        player, checkpoint, keymap, scale=scale, mode_factory=mode_factory
    ).start()
    server = None
    try:
        server, url = make_server(session, port)
        player_url, stats_url = dashboard_urls(url)
        print(f"Gymemu player: {url}", flush=True)
        print(f"Gymemu diagnostics: {stats_url}", flush=True)
        print(
            "Space/action keys step; Tab plays; R resets; C selects next. Ctrl+C stops server.",
            flush=True,
        )
        if open_browser:
            webbrowser.open(player_url, new=2)
            webbrowser.open(stats_url, new=2, autoraise=False)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.server_close()
        session.close()
