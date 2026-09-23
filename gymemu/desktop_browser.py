"""Dedicated Neutralino windows for the player and diagnostics workspace."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import aiohttp

from gymemu.desktop_runtime import ASSETS, viewer_executable


class DesktopWindow:
    def __init__(self, executable: Path, url: str, title: str, role: str) -> None:
        self.profile = tempfile.TemporaryDirectory(prefix="gymemu-viewer-")
        self.process: subprocess.Popen | None = None
        self._close_listener: threading.Thread | None = None
        self._close_ready = threading.Event()
        self._close_error: BaseException | None = None
        self._close_lock = threading.Lock()
        root = Path(self.profile.name)
        try:
            (root / "public").mkdir()
            shutil.copyfile(ASSETS / f"{role}.png", root / "icon.png")
            (root / "neutralino.config.json").write_text(
                json.dumps(
                    {
                        "applicationId": f"org.gymemu.viewer.{role}",
                        "version": "1.0.0",
                        "defaultMode": "window",
                        "url": url,
                        "port": 0,
                        "enableServer": True,
                        "documentRoot": "/public",
                        "enableNativeAPI": True,
                        "exportAuthInfo": True,
                        "nativeAllowList": ["window.show", "window.unminimize", "window.focus"],
                        "logging": {"enabled": True, "writeToLogFile": True},
                        "modes": {
                            "window": {
                                "title": title,
                                "icon": "/icon.png",
                                "width": 1440,
                                "height": 960,
                                "minWidth": 800,
                                "minHeight": 600,
                                # Neutralino 6.9.0 traps in its macOS close callback
                                # when it exits the process from that callback.
                                "exitProcessOnClose": False,
                                "useSavedState": False,
                                "injectGlobals": False,
                                "injectClientLibrary": False,
                                "extendUserAgentWith": "GymemuDesktop",
                                "newWindowPolicy": "browser",
                            }
                        },
                    }
                )
            )
            with (root / "process.log").open("wb") as log:
                self.process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("desktop_watchdog.py")),
                        "--cleanup-directory",
                        str(root),
                        str(executable),
                        "--res-mode=directory",
                        f"--path={root}",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
        except BaseException:
            self.close()
            raise

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    async def focus(self) -> None:
        """Focus through Python; web pages cannot call native methods."""
        auth_file = Path(self.profile.name) / ".tmp/auth_info.json"
        async with asyncio.timeout(15):
            while not auth_file.is_file():
                if not self.alive:
                    raise RuntimeError(
                        "Gymemu desktop viewer exited during startup. On Linux, install "
                        "GTK 3 and WebKitGTK 4.1, or use --no-browser."
                    )
                await asyncio.sleep(0.05)
            auth = json.loads(auth_file.read_text())
            if self._close_listener is None:
                self._close_listener = threading.Thread(
                    target=self._run_close_listener, args=(auth,), daemon=True
                )
                self._close_listener.start()
                if not await asyncio.to_thread(self._close_ready.wait, 5):
                    raise RuntimeError("Desktop viewer close listener did not connect")
                if self._close_error is not None:
                    raise RuntimeError(
                        "Desktop viewer close listener failed"
                    ) from self._close_error
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"ws://127.0.0.1:{int(auth['nlPort'])}?connectToken={auth['nlConnectToken']}"
                ) as socket:
                    for method in ("window.show", "window.unminimize", "window.focus"):
                        request_id = uuid4().hex
                        await socket.send_json(
                            {
                                "id": request_id,
                                "method": method,
                                "data": {},
                                "accessToken": auth["nlToken"],
                            }
                        )
                        async for message in socket:
                            if message.type != aiohttp.WSMsgType.TEXT:
                                continue
                            result = json.loads(message.data)
                            if result.get("id") == request_id:
                                if not result.get("data", {}).get("success"):
                                    raise RuntimeError(f"Desktop viewer could not {method}")
                                break
                        else:
                            raise RuntimeError("Desktop viewer disconnected")

    def _run_close_listener(self, auth: dict) -> None:
        asyncio.run(self._listen_for_close(auth))

    async def _listen_for_close(self, auth: dict) -> None:
        address = (
            f"ws://127.0.0.1:{int(auth['nlPort'])}"
            f"?connectToken={auth['nlConnectToken']}"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(address) as socket:
                    self._close_ready.set()
                    async for message in socket:
                        if message.type != aiohttp.WSMsgType.TEXT:
                            continue
                        if json.loads(message.data).get("event") == "windowClose":
                            return
        except BaseException as error:
            self._close_error = error
            self._close_ready.set()
        finally:
            # Closing the watchdog pipe stops Neutralino outside its native
            # close callback. A lost listener also stops an orphaned viewer.
            await asyncio.to_thread(self.close)

    def close(self) -> None:
        with self._close_lock:
            if self.process is not None:
                if self.process.stdin is not None and not self.process.stdin.closed:
                    self.process.stdin.close()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                self.process = None
            self.profile.cleanup()


class PlaybackBrowser:
    def __init__(self) -> None:
        self.windows: dict[str, DesktopWindow] = {}
        self.executables: dict[str, Path] = {}
        self.workspace_id = uuid4().hex
        self.lock = threading.RLock()

    def desktop_url(self, url: str) -> str:
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query))
        query["desktop"] = self.workspace_id
        return urlunsplit(parts._replace(query=urlencode(query)))

    def open(self, url: str, role: str = "player") -> None:
        if role not in {"player", "stats"}:
            raise ValueError("Unknown desktop window")
        with self.lock:
            window = self.windows.get(role)
            if window is not None and not window.alive:
                window.close()
                del self.windows[role]
                window = None
            if window is None:
                if role not in self.executables:
                    self.executables[role] = viewer_executable(role)
                window = DesktopWindow(
                    self.executables[role],
                    self.desktop_url(url),
                    f"Gymemu — {'Player' if role == 'player' else 'Diagnostics'}",
                    role,
                )
                self.windows[role] = window
            try:
                asyncio.run(window.focus())
            except BaseException:
                window.close()
                self.windows.pop(role, None)
                raise

    def any_open(self) -> bool:
        with self.lock:
            return any(window.alive for window in self.windows.values())

    def close(self) -> None:
        with self.lock:
            for window in self.windows.values():
                window.close()
            self.windows.clear()
