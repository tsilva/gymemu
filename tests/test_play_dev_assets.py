"""The source UI keeps its API on the player origin while Vite serves modules."""

import threading
from types import SimpleNamespace
from urllib.request import urlopen

from gymemu.play_dev_assets import development_page, source_checkout_root
from gymemu.web_player import make_server


def test_source_checkout_pages_use_vite_for_both_entries():
    root = source_checkout_root()
    assert root is not None
    vite_url = "http://127.0.0.1:5173"
    for page, entry in (("index.html", "main"), ("catalog.html", "catalog")):
        markup = development_page(
            (root / "gymemu/web_assets" / page).read_text(),
            vite_url,
            catalog=entry == "catalog",
        )
        assert 'data-hot-reload="true"' in markup
        assert f'{vite_url}/frontend/{entry}.js' in markup
        assert f'{vite_url}/gymemu/web_assets/styles.css?import' in markup
        assert "/assets/dist/" not in markup


def test_hot_reload_pages_keep_player_origin_and_allow_vite_assets():
    vite_url = "http://127.0.0.1:5173"
    server, url = make_server(
        object(), catalog=object(), dev_assets=SimpleNamespace(url=vite_url)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        root = url.split("#", 1)[0]
        for path, entry in (("browse", "catalog"), ("player", "main")):
            with urlopen(root + path, timeout=5) as response:
                page = response.read().decode()
                policy = response.headers["Content-Security-Policy"]
                assert response.status == 200
                assert f'{vite_url}/frontend/{entry}.js' in page
                assert vite_url in policy
                assert "ws://127.0.0.1:5173" in policy
        with urlopen(root + "assets/tabler-icons.svg", timeout=5) as response:
            assert response.status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
