"""Local browser transport. One worker owns inference; HTTP only exchanges snapshots."""

from __future__ import annotations

import base64
import copy
import io
import json
import mimetypes
import queue
import secrets
import threading
import time
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import torch
from PIL import Image

from gymemu.play_dev_assets import PlayerDevAssets, development_page, source_checkout_root
from gymemu.player import PLAYBACK_FPS, handle_key
from gymemu.replay import ReplayPlayer
from gymemu.research_catalog import ResearchCatalog

ASSETS = Path(__file__).with_name("web_assets")
DESKTOP_ASSETS = Path(__file__).with_name("desktop")


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
        self.players = {getattr(player, "mode", "autoregressive"): player}
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
        self.history_preview = None
        self.history_revision = 0
        self.history_source = None
        self.history_edits = {}
        self.record_mse()

    def sync_history_source(self):
        # Every prediction and reset replaces the input stack, even a seek to the same step.
        if self.history_source is not self.player.input_stack:
            self.history_source = self.player.input_stack
            self.history_preview = None
            self.history_edits = {}
            self.history_revision += 1

    def validate_history_revision(self, command):
        if not getattr(self.player, "history_editable", True):
            raise ValueError("This player uses state inputs; RGB history cannot be edited")
        if getattr(self.player, "mode", None) == "reconstruction":
            raise ValueError(
                "Reconstruction uses the current recorded frame; history edits are unavailable"
            )
        self.sync_history_source()
        if not self.player.has_prediction:
            raise ValueError("Predict a frame before modifying its history")
        if command.get("history_revision") != self.history_revision:
            raise ValueError("The input history has changed; reopen the frame or drag again")

    def reorder_history(self, command):
        """Preview a permutation without writing into recorded or generated trajectory history."""
        self.validate_history_revision(command)
        order = command.get("order")
        count = self.player.config["history"]
        if (
            not isinstance(order, list)
            or len(order) != count
            or any(type(index) is not int for index in order)
            or sorted(order) != list(range(count))
        ):
            raise ValueError("History order must contain each frame exactly once")
        self.predict_history(order, self.history_edits)

    def edit_history(self, command):
        """Apply palette-constrained pixel changes to one original frame identity."""
        self.validate_history_revision(command)
        frame = command.get("frame")
        if type(frame) is not int or not 0 <= frame < self.player.config["history"]:
            raise ValueError("Unknown history frame")
        source = self.player.input_stack[frame]
        _, height, width = source.shape
        pixels = command.get("pixels")
        if not isinstance(pixels, list) or not 0 < len(pixels) <= height * width:
            raise ValueError("Supply at most one edit per frame pixel")
        # Match the RGB bytes shown in the browser, but preserve all untouched float pixels.
        palette = set(
            map(tuple, source.mul(255).round().byte().permute(1, 2, 0).reshape(-1, 3).tolist())
        )
        seen = set()
        for pixel in pixels:
            if (
                not isinstance(pixel, list)
                or len(pixel) != 4
                or any(type(value) is not int for value in pixel)
                or not 0 <= pixel[0] < height * width
                or pixel[0] in seen
            ):
                raise ValueError("Invalid or duplicate pixel position")
            if tuple(pixel[1:]) not in palette:
                raise ValueError("Paint colors must belong to the frame's original palette")
            seen.add(pixel[0])
        edits = dict(self.history_edits)
        edited = edits.get(frame, source).clone()
        changes = torch.tensor(pixels)
        positions = changes[:, 0]
        edited[:, positions // width, positions % width] = changes[:, 1:].T.to(edited.dtype) / 255
        edits[frame] = edited
        order = (
            self.history_preview[1]
            if self.history_preview
            else list(range(len(self.history_source)))
        )
        self.predict_history(order, edits)

    def revert_history(self, command):
        """Restore one painted frame while retaining the other edits and frame order."""
        self.validate_history_revision(command)
        frame = command.get("frame")
        if type(frame) is not int or frame not in self.history_edits:
            raise ValueError("This history frame has no edits to revert")
        edits = dict(self.history_edits)
        del edits[frame]
        self.predict_history(self.history_preview[1], edits)

    @torch.inference_mode()
    def predict_history(self, order, edits):
        self.pause()
        player = copy.copy(self.player)
        stack = torch.stack([edits.get(index, player.input_stack[index]) for index in order])
        started = time.perf_counter()
        inputs = stack.unsqueeze(0).to(player.device)
        if player.input_states is None:
            prediction = player.model(inputs, player.input_tokens)
        else:
            prediction, _ = player.model.predict_step(
                inputs, player.input_tokens, player.input_states
            )
        player.frame = prediction[0].float().cpu()
        elapsed = (time.perf_counter() - started) * 1000
        player.input_stack = stack
        if self.replay:
            player.mse = (player.frame - player.target.float()).square().mean().item()
        self.history_preview = (player, order, elapsed)
        self.history_edits = edits
        self.history_revision += 1

    def pause(self):
        self.player.continuous = False
        self.player.held_keys.clear()

    def clear_history(self):
        self.history.clear()
        self.history_epoch += 1
        self.record_mse()

    def record_mse(self):
        if self.replay and self.player.mse is not None:
            self.history[self.player.steps] = {"step": self.player.steps, "mse": self.player.mse}

    def advance(self, action=None):
        if getattr(self.player, "finished", False):
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
            modes = getattr(
                self.player.model, "playback_modes", ("teacher-forcing", "autoregressive")
            )
            if mode not in modes or self.mode_factory is None:
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
        elif kind == "pause":
            self.pause()
        elif kind == "blur":
            self.player.held_keys.clear()
        elif kind == "play":
            self.player.continuous = not getattr(self.player, "finished", False)
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
        elif kind == "reorder_history":
            self.reorder_history(command)
        elif kind == "edit_history":
            self.edit_history(command)
        elif kind == "revert_history":
            self.revert_history(command)
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
            else:
                self.record_mse()
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
            self.player.held_keys.clear()
        action = (
            self.keymap[self.player.held_keys[-1]] if self.player.held_keys else self.default_action
        )
        self.advance(action)
        return True

    def snapshot(self):
        self.sync_history_source()
        player = self.history_preview[0] if self.history_preview else self.player
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
            "mode": getattr(player, "mode", "autoregressive"),
            "available_modes": list(
                getattr(player.model, "playback_modes", ("autoregressive", "teacher-forcing"))
            )
            if self.mode_factory
            else [],
            "checkpoint": self.checkpoint,
            "name": player.start_name or "Empty start",
            "playing": self.player.continuous,
            "step": player.steps,
            "total_steps": len(player.episode.actions) if self.replay else None,
            "selection": player.episode.episode_id if self.replay else player.start_index,
            "choices": choices,
            "has_prediction": player.has_prediction,
            "finished": getattr(player, "finished", False),
            "action": player.recorded_action if self.replay else (player.last_action),
            "action_values": player.actions,
            "keymap": self.keymap,
            "mse": player.mse if self.replay else None,
            "inference_ms": self.history_preview[2] if self.history_preview else self.inference_ms,
            "history": [self.history[step] for step in sorted(self.history)],
            "history_epoch": self.history_epoch,
            "history_length": player.config["history"],
            "history_order": (
                self.history_preview[1]
                if self.history_preview
                else list(range(player.config["history"]))
            ),
            "history_revision": self.history_revision,
            "history_editable": getattr(player, "history_editable", True)
            and getattr(player, "mode", None) != "reconstruction",
            "history_edited_frames": sorted(self.history_edits),
            "history_reordered": bool(
                self.history_preview
                and self.history_preview[1] != list(range(player.config["history"]))
            ),
            "shape": player.config["shape"],
            "scale": self.scale,
            "action_history": player.config.get("action_history", 1),
            "state_fields": player.config.get("state_fields", []),
            "context_note": getattr(player, "context_note", None),
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
        deadline = time.monotonic() + 1 / getattr(self.player, "playback_fps", PLAYBACK_FPS)
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
                deadline = time.monotonic() + 1 / getattr(self.player, "playback_fps", PLAYBACK_FPS)
            except Exception as error:
                self.pause()
                self.error = str(error)
                self.publish()
                if pending is not None:
                    pending.set_exception(error)
            if not self.player.continuous:
                deadline = time.monotonic() + 1 / getattr(self.player, "playback_fps", PLAYBACK_FPS)

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


class CatalogSession:
    """Own one active player; serialize checkpoint replacement with player requests."""

    def __init__(self, catalog, factory):
        self.catalog, self.factory = catalog, factory
        self.session = None
        self.lock = threading.RLock()
        self.generation = 0
        self.label = None

    def open(self, identifier):
        with self.lock:
            checkpoint = self.catalog.resolve(identifier)
            if self.session:
                self.session.submit({"type": "pause"})
            label = self.catalog.label(identifier)
            candidate = self.factory(checkpoint)
            candidate.revision = self.session.revision if self.session else 0
            candidate.start()
            if self.session:
                self.session.close()
            self.generation += 1
            self.session = candidate
            self.label = label

    def submit(self, command):
        with self.lock:
            if self.session is None:
                raise ValueError("Choose a checkpoint first")
            self.session.submit(command)

    def read(self, after=-1):
        with self.lock:
            if self.session is None:
                return {"catalog": True, "revision": 0}
            return {
                **self.session.read(after),
                "catalog": True,
                "generation": self.generation,
                "checkpoint_label": self.label,
            }

    def close(self):
        with self.lock:
            if self.session:
                self.session.close()


def serve_catalog(catalog, factory, *, port=0, open_browser=True, hot_reload=False):
    checkout_root = source_checkout_root() if hot_reload else None
    if hot_reload and checkout_root is None:
        raise RuntimeError("--hotreload requires a source checkout")
    dev_assets = PlayerDevAssets(checkout_root) if checkout_root else None
    session = CatalogSession(catalog, factory)
    server = None
    browser = None
    try:
        if dev_assets:
            print(f"Player hot reload: {dev_assets.start()}", flush=True)
        if open_browser:
            from gymemu.desktop_browser import PlaybackBrowser

            browser = PlaybackBrowser()
        server, url = make_server(
            session, port, catalog=catalog, desktop_browser=browser, dev_assets=dev_assets
        )
        print(f"Gymemu navigator: {url}", flush=True)
        print(f"Catalog: {catalog.root}", flush=True)
        if browser:
            browser.open(url)
            server.timeout = 0.2
            while browser.any_open():
                server.handle_request()
        else:
            server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.server_close()
        if browser:
            browser.close()
        session.close()
        if dev_assets:
            dev_assets.stop()


def make_server(
    session, port=0, *, workspace_path=None, catalog=None, desktop_browser=None, dev_assets=None
):
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
            vite_url = dev_assets.url if dev_assets else None
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data: blob:; "
                f"style-src 'self' 'unsafe-inline' {vite_url or ''}; "
                f"font-src 'self' {vite_url or ''}; "
                f"script-src 'self' {vite_url or ''}; "
                f"connect-src 'self' {vite_url or ''} "
                f"{'ws://' + vite_url.removeprefix('http://') if vite_url else ''}; "
                "frame-ancestors 'none'",
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
            if url.path == "/api/catalog" and catalog is not None:
                if not self.authorized():
                    return self.respond(403, b'{"error":"Unauthorized player request"}')
                try:
                    with session.lock:
                        query = parse_qs(url.query)
                        result = catalog.snapshot(
                            run_id=query.get("run", [None])[0],
                            refresh=query.get("refresh", [""])[0] == "1",
                            **(
                                {"cursor": query.get("cursor", [None])[0], "page_size": 50}
                                if isinstance(catalog, ResearchCatalog)
                                else {}
                            ),
                        )
                except Exception as error:
                    return self.respond(
                        503, json.dumps({"error": f"Catalog unavailable: {error}"}).encode()
                    )
                return self.respond(200, json.dumps(result, allow_nan=False).encode())
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
            navigator = catalog is not None and url.path in ("/", "/browse")
            dashboard = url.path in ("/", "/player", "/workspace/stats", "/browse")
            page = "catalog.html" if navigator else "index.html"
            icon = {
                "/assets/viewer-player.png": "player.png",
                "/assets/viewer-stats.png": "stats.png",
            }.get(url.path)
            if icon:
                path, allowed_root = DESKTOP_ASSETS / icon, DESKTOP_ASSETS
            else:
                path = ASSETS / (page if dashboard else url.path.removeprefix("/assets/"))
                allowed_root = ASSETS
            if not dashboard and not url.path.startswith("/assets/"):
                return self.respond(404, b"Not found", "text/plain")
            if not path.resolve().is_relative_to(allowed_root.resolve()) or not path.is_file():
                return self.respond(404, b"Not found", "text/plain")
            body = path.read_bytes()
            if dashboard and dev_assets:
                body = development_page(
                    body.decode(), dev_assets.url, catalog=navigator
                ).encode()
            return self.respond(
                200, body, mimetypes.guess_type(path)[0] or "application/octet-stream"
            )

        def do_POST(self):
            if self.path not in ("/api/command", "/api/workspace", "/api/open", "/api/window"):
                return self.respond(404, b'{"error":"Not found"}')
            if not self.authorized():
                return self.respond(403, b'{"error":"Unauthorized player request"}')
            try:
                length = int(self.headers.get("Content-Length", "0"))
                limit = (
                    65536
                    if self.path == "/api/workspace"
                    else 16 * 1024 * 1024
                    if self.path == "/api/command"
                    else 4096
                )
                if not 0 < length <= limit:
                    raise ValueError("Invalid command size")
                command = json.loads(self.rfile.read(length))
                if not isinstance(command, dict):
                    raise ValueError("Expected a playback command")
                if (
                    self.path == "/api/command"
                    and command.get("type") != "edit_history"
                    and length > 4096
                ):
                    raise ValueError("Invalid command size")
                if self.path == "/api/window":
                    if desktop_browser is None or command.get("window") not in ("player", "stats"):
                        raise ValueError("Desktop window is unavailable")
                    path = "/player" if command["window"] == "player" else "/workspace/stats"
                    target = f"http://127.0.0.1:{self.server.server_port}{path}#token={token}"
                    try:
                        desktop_browser.open(target, command["window"])
                    except Exception as error:
                        return self.respond(500, json.dumps({"error": str(error)}).encode())
                elif self.path == "/api/open":
                    if catalog is None:
                        raise ValueError("The checkpoint navigator is unavailable")
                    try:
                        session.open(command.get("checkpoint"))
                    except Exception as error:
                        return self.respond(400, json.dumps({"error": str(error)}).encode())
                elif self.path == "/api/workspace":
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


def serve(
    player, *, checkpoint, keymap, port=0, open_browser=True, scale=3,
    mode_factory=None, hot_reload=False,
):
    checkout_root = source_checkout_root() if hot_reload else None
    if hot_reload and checkout_root is None:
        raise RuntimeError("--hotreload requires a source checkout")
    dev_assets = PlayerDevAssets(checkout_root) if checkout_root else None
    session = PlaybackSession(
        player, checkpoint, keymap, scale=scale, mode_factory=mode_factory
    ).start()
    server = None
    browser = None
    try:
        if dev_assets:
            print(f"Player hot reload: {dev_assets.start()}", flush=True)
        if open_browser:
            from gymemu.desktop_browser import PlaybackBrowser

            browser = PlaybackBrowser()
        server, url = make_server(session, port, desktop_browser=browser, dev_assets=dev_assets)
        player_url, stats_url = dashboard_urls(url)
        print(f"Gymemu player: {url}", flush=True)
        print(f"Gymemu diagnostics: {stats_url}", flush=True)
        print(
            "Space/action keys step; Tab plays; R resets; C selects next. Ctrl+C stops server.",
            flush=True,
        )
        if browser:
            browser.open(player_url)
            server.timeout = 0.2
            while browser.any_open():
                server.handle_request()
        else:
            server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.server_close()
        if browser:
            browser.close()
        session.close()
        if dev_assets:
            dev_assets.stop()
