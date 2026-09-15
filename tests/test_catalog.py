import json
import threading
from pathlib import Path
from urllib.error import HTTPError

import pytest
import torch

from gymemu.catalog import LocalCatalog, run_directory
from gymemu.web_player import CatalogSession, make_server
from tests.test_web_player import api, session


def make_run(root, name, env_id="ALE/Breakout-v5", *, sidecar=True):
    directory = root / name
    directory.mkdir(parents=True)
    config = {"game": {"env_id": env_id}, "approach": {"kind": "latent"}}
    if sidecar:
        (directory / "config.json").write_text(json.dumps(config))
    torch.save({"config": config, "state_dict": {}}, directory / "best.pt")
    return directory


def test_catalog_nested_runs_stages_refresh_and_unknown_metadata(tmp_path):
    directory = make_run(tmp_path, "nested/first")
    stage = directory / "stages/prediction"
    stage.mkdir(parents=True)
    (stage / "last.pt").write_bytes(b"stage")
    make_run(tmp_path, "copied", sidecar=False)
    make_run(tmp_path, "legacy", env_id=None, sidecar=False)
    pending = tmp_path / "pending"
    pending.mkdir()
    (pending / "config.json").write_text('{"game":{"env_id":"Other-v0"}}')
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "best.pt").write_bytes(b"corrupt")
    catalog = LocalCatalog(tmp_path)
    snapshot = catalog.snapshot()
    groups = {env["id"]: env["runs"] for env in snapshot["environments"]}
    assert set(groups) == {"ALE/Breakout-v5", "Other-v0", "Unknown environment"}
    assert len(groups["ALE/Breakout-v5"]) == 2
    run = next(r for r in groups["ALE/Breakout-v5"] if r["name"] == "nested/first")
    assert {c["name"] for c in run["checkpoints"]} == {"best.pt", "stages/prediction/last.pt"}
    assert not groups["Other-v0"][0]["checkpoints"]
    assert len(snapshot["warnings"]) == 1 and "broken" in snapshot["warnings"][0]
    assert run_directory(stage / "last.pt") == directory
    selected = next(c for c in run["checkpoints"] if c["name"] == "best.pt")
    assert catalog.resolve(selected["id"]) == directory / "best.pt"
    (directory / "best.pt").unlink()
    with pytest.raises(ValueError, match="no longer available"):
        catalog.resolve(selected["id"])
    assert catalog.snapshot()["environments"]
    with pytest.raises(ValueError, match="Unknown checkpoint"):
        catalog.resolve(selected["id"])


def test_catalog_rejects_paths_and_symlink_escape(tmp_path):
    root = tmp_path / "runs"
    directory = make_run(root, "valid")
    outside = make_run(tmp_path, "outside")
    (directory / "escape.pt").symlink_to(outside / "best.pt")
    catalog = LocalCatalog(root)
    snapshot = catalog.snapshot()
    items = snapshot["environments"][0]["runs"][0]["checkpoints"]
    assert [c["name"] for c in items] == ["best.pt"]
    for value in ("../outside/best.pt", str(outside / "best.pt"), None, []):
        with pytest.raises(ValueError):
            catalog.resolve(value)
    checkpoint = directory / "best.pt"
    checkpoint.unlink()
    checkpoint.symlink_to(outside / "best.pt")
    with pytest.raises(ValueError):
        catalog.resolve(items[0]["id"])


def test_catalog_http_selection_replacement_and_recoverable_failure(tmp_path):
    make_run(tmp_path, "one")
    make_run(tmp_path, "two")
    catalog = LocalCatalog(tmp_path)
    created = []

    def factory(checkpoint):
        if checkpoint.parent.name == "two":
            raise ValueError("Recorded starting scene not found")
        result = session()
        result.checkpoint = str(checkpoint)
        created.append(result)
        return result

    host = CatalogSession(catalog, factory)
    server, url = make_server(host, catalog=catalog, workspace_path=tmp_path / "workspace.json")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(HTTPError) as error:
            api(url, "/api/catalog", **{"X-Player-Token": "wrong"})
        assert error.value.code == 403
        snapshot = api(url, "/api/catalog")
        assert not created
        assert api(url, "/api/state") == {"catalog": True, "revision": 0}
        runs = {r["name"]: r for r in snapshot["environments"][0]["runs"]}
        good = runs["one"]["checkpoints"][0]["id"]
        bad = runs["two"]["checkpoints"][0]["id"]
        api(url, "/api/open", {"checkpoint": good})
        first = api(url, "/api/state")
        assert not first["playing"] and first["catalog"]
        assert Path(first["checkpoint"]).name == "best.pt"
        api(url, "/api/command", {"type": "step"})
        with pytest.raises(HTTPError) as error:
            api(url, "/api/open", {"checkpoint": bad})
        assert "Recorded starting scene" in error.value.read().decode()
        assert not host.session.closed.is_set()
        api(url, "/api/open", {"checkpoint": good})
        second = api(
            url,
            "/api/state",
        )
        assert second["generation"] > first["generation"]
        assert second["revision"] > first["revision"]
        assert created[0].closed.is_set() and not created[0].worker.is_alive()
    finally:
        server.shutdown()
        server.server_close()
        host.close()
        thread.join(timeout=3)


def test_cli_without_checkpoint_opens_catalog_without_loading_model(tmp_path, monkeypatch):
    from gymemu.commands import play

    captured = {}

    def serve(catalog, factory, **options):
        captured.update(root=catalog.root, options=options, factory=factory)

    monkeypatch.setattr("gymemu.web_player.serve_catalog", serve)
    monkeypatch.setattr(play, "load_model", lambda *args: pytest.fail("Loaded before selection"))
    play.main(["--runs-dir", str(tmp_path), "--no-browser", "--device", "cpu"])
    assert captured["root"] == tmp_path
    assert captured["options"] == {"port": 0, "open_browser": False}
    for arguments in (["--headless-actions", "0"], ["--list-start-states"]):
        with pytest.raises(SystemExit) as error:
            play.main(arguments)
        assert error.value.code == 2
