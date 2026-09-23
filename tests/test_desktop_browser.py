"""Native viewer close events shut down through the watchdog pipe."""

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import Mock, patch

from aiohttp import web

from gymemu.desktop_browser import DesktopWindow


def test_native_close_does_not_exit_from_neutralino_callback():
    process = Mock()
    process.stdin.closed = False
    with patch("gymemu.desktop_browser.subprocess.Popen", return_value=process):
        window = DesktopWindow(Path("/viewer"), "http://127.0.0.1:1234/", "Gymemu", "player")
        root = Path(window.profile.name)
        config = json.loads((root / "neutralino.config.json").read_text())
        assert config["modes"]["window"]["exitProcessOnClose"] is False
        window.close()
        process.stdin.close.assert_called_once()
        process.wait.assert_called_once()
        assert not root.exists()


def test_native_close_event_stops_viewer_without_reentering_close_callback():
    async def scenario():
        connected = asyncio.Event()
        sockets = {}

        async def websocket(request):
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            sockets["viewer"] = socket
            connected.set()
            async for _message in socket:
                pass
            return socket

        app = web.Application()
        app.router.add_get("/", websocket)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        window = DesktopWindow.__new__(DesktopWindow)
        window._close_ready = threading.Event()
        window._close_error = None
        window.close = Mock()
        listener = threading.Thread(
            target=window._run_close_listener,
            args=({"nlPort": port, "nlConnectToken": "test"},),
            daemon=True,
        )
        listener.start()
        try:
            await asyncio.wait_for(connected.wait(), 2)
            assert await asyncio.to_thread(window._close_ready.wait, 2)
            await sockets["viewer"].send_json({"event": "windowFocus"})
            await asyncio.sleep(0.05)
            window.close.assert_not_called()
            await sockets["viewer"].send_json({"event": "windowClose"})
            await asyncio.to_thread(listener.join, 2)
            assert not listener.is_alive()
            window.close.assert_called_once()
            assert window._close_error is None
        finally:
            await runner.cleanup()

    asyncio.run(scenario())
