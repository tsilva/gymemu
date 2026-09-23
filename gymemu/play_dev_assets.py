"""Serve source-checkout player assets through a short-lived Vite process."""

from __future__ import annotations

import select
import shutil
import subprocess
from pathlib import Path


def source_checkout_root() -> Path | None:
    root = Path(__file__).resolve().parent.parent
    required = ("vite.config.ts", "frontend/main.js", "scripts/player-dev-server.mjs")
    return root if all((root / name).is_file() for name in required) else None


class PlayerDevAssets:
    def __init__(self, root: Path):
        self.root = root
        self.process = None
        self.url = None

    def start(self) -> str:
        node = shutil.which("node")
        if node is None or not (self.root / "node_modules/vite").is_dir():
            raise RuntimeError(
                "Hot reload requires Node and checkout dependencies. "
                "Run pnpm install --frozen-lockfile, or omit --hotreload."
            )
        self.process = subprocess.Popen(
            [node, str(self.root / "scripts/player-dev-server.mjs")],
            cwd=self.root,
            stdout=subprocess.PIPE,
        )
        try:
            assert self.process.stdout is not None
            ready, _, _ = select.select([self.process.stdout], [], [], 20)
            if not ready:
                raise RuntimeError("Vite did not report its development URL within 20 seconds")
            line = self.process.stdout.readline()
            prefix = b"GYMEMU_VITE_URL="
            if not line.startswith(prefix):
                raise RuntimeError("Vite did not report its development URL")
            self.url = line[len(prefix) :].decode().strip()
            return self.url
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        process = self.process
        self.process = None
        self.url = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()


def development_page(markup: str, vite_url: str, *, catalog: bool) -> str:
    entry = "catalog" if catalog else "main"
    bundle = "catalog" if catalog else "app"
    page = markup.replace("<body", '<body data-hot-reload="true"', 1)
    page = page.replace(
        '<link rel="stylesheet" href="/assets/styles.css">',
        f'<script type="module" src="{vite_url}/gymemu/web_assets/styles.css?import"></script>',
    )
    if catalog:
        page = page.replace(
            '<link rel="stylesheet" href="/assets/catalog.css">',
            f'<script type="module" '
            f'src="{vite_url}/gymemu/web_assets/catalog.css?import"></script>',
        )
    return page.replace(
        f'<script type="module" src="/assets/dist/{bundle}.js"></script>',
        f'<script type="module" src="{vite_url}/@vite/client"></script>\n'
        f'  <script type="module" src="{vite_url}/frontend/{entry}.js"></script>',
    )
