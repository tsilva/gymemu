import hashlib
import io
import json
from datetime import datetime, timezone

import pytest

from gymemu.catalog import CheckpointCatalog
from gymemu.remote_catalog import RemoteCatalog
from tests.test_catalog import make_run


class Remote:
    def __init__(self):
        self.objects = {}
        self.entries = {}
        self.reads = []

    def put(self, key, payload, second=0):
        self.objects[key] = payload
        self.entries[key] = {
            "Key": key,
            "ETag": hashlib.sha256(payload).hexdigest(),
            "LastModified": datetime(2026, 9, 15, 12, 0, second, tzinfo=timezone.utc),
        }

    def get_object(self, *, Bucket, Key):
        self.reads.append(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def get_paginator(self, name):
        return self

    def paginate(self, *, Bucket, Prefix):
        entries = [value for key, value in self.entries.items() if key.startswith(Prefix)]
        # Exercise pagination, including empty pages.
        yield {"Contents": entries[:2]}
        yield {}
        yield {"Contents": entries[2:]}

    def publish(self, checkpoint=b"first", *, second=0, scene=b"scene"):
        prefix = "runs/ALE-Breakout-v5/test-run"
        objects = {}
        files = {
            "latest.pt": checkpoint,
            "stages/prediction/latest.pt": checkpoint,
            "start-scene.npz": scene,
            "summary.json": b'{"name":"remote-training","approach":"latent"}',
        }
        for name, payload in files.items():
            digest = hashlib.sha256(payload).hexdigest()
            key = f"{prefix}/objects/{digest}"
            self.put(key, payload, second)
            objects[name] = {"key": key, "sha256": digest, "size_bytes": len(payload)}
        manifest = {
            "version": 1,
            "bucket": "gymemu",
            "prefix": prefix,
            "env_id": "ALE/Breakout-v5",
            "run_id": "test-run",
            "objects": objects,
        }
        payload = json.dumps(manifest).encode()
        digest = hashlib.sha256(payload).hexdigest()
        self.put(f"{prefix}/manifests/{digest}.json", payload, second)
        self.put(f"{prefix}/manifest.json", payload, second)
        return manifest


def test_remote_lazily_lists_history_downloads_paired_scene_and_reuses_cache(tmp_path):
    client = Remote()
    client.publish()
    current = client.publish(b"second", second=1, scene=b"second-scene")
    remote = RemoteCatalog(client=client, cache=tmp_path)
    runs, warnings = remote.snapshot()
    assert not warnings and len(runs) == 1
    assert runs[0]["env_id"] == "ALE/Breakout-v5" and not runs[0]["checkpoints"]
    checkpoint_keys = {r["key"] for n, r in current["objects"].items() if n.endswith(".pt")}
    assert not checkpoint_keys.intersection(client.reads)
    assert not any("/manifests/" in key for key in client.reads)
    runs, _ = remote.snapshot(run_id=runs[0]["id"])
    items = runs[0]["checkpoints"]
    assert len(items) == 4  # Two names, each with two unique versions.
    selected = next(c for c in items if c["name"].startswith("latest.pt ·"))
    path = remote.resolve(selected["id"])
    assert path.read_bytes() == b"first"
    assert (path.parent / "start-scene.npz").read_bytes() == b"scene"
    reads = len(client.reads)
    assert remote.resolve(selected["id"]) == path and len(client.reads) == reads
    path.write_bytes(b"broken cache")
    assert remote.resolve(selected["id"]).read_bytes() == b"first"
    stage = next(c for c in items if c["name"] == "stages/prediction/latest.pt")
    path = remote.resolve(stage["id"])
    assert path.read_bytes() == b"second"
    assert (path.parents[2] / "start-scene.npz").read_bytes() == b"second-scene"


def test_remote_rejects_corrupt_downloads_and_manifest_paths(tmp_path):
    client = Remote()
    manifest = client.publish()
    remote = RemoteCatalog(client=client, cache=tmp_path)
    runs, _ = remote.snapshot()
    runs, _ = remote.snapshot(run_id=runs[0]["id"])
    selected = runs[0]["checkpoints"][0]
    client.objects[manifest["objects"]["latest.pt"]["key"]] = b"corrupt"
    with pytest.raises(ValueError, match="checksum mismatch"):
        remote.resolve(selected["id"])
    assert not list(tmp_path.rglob("*.pt"))
    key = f"{manifest['prefix']}/manifest.json"
    manifest["objects"]["../escape.pt"] = manifest["objects"]["latest.pt"]
    client.put(key, json.dumps(manifest).encode())
    runs, warnings = remote.snapshot(refresh=True)
    assert not runs and "Invalid R2 artifact" in warnings[0]


def test_combined_catalog_keeps_local_runs_when_r2_is_unavailable(tmp_path):
    make_run(tmp_path, "local")

    class Offline:
        def snapshot(self, **kwargs):
            raise OSError("offline")

    catalog = CheckpointCatalog(tmp_path, remote=Offline())
    result = catalog.snapshot()
    assert result["environments"][0]["runs"][0]["source"] == "Local"
    assert "R2 unavailable" in result["warnings"][0]
    assert "offline" in result["warnings"][0]


def test_combined_catalog_groups_local_and_remote_by_exact_env_id(tmp_path):
    make_run(tmp_path, "local")
    client = Remote()
    client.publish()
    catalog = CheckpointCatalog(
        tmp_path, remote=RemoteCatalog(client=client, cache=tmp_path / "cache")
    )
    result = catalog.snapshot()
    assert len(result["environments"]) == 1
    runs = result["environments"][0]["runs"]
    assert {run["source"] for run in runs} == {"Local", "R2"}
    remote_run = next(run for run in runs if run["source"] == "R2")
    result = catalog.snapshot(run_id=remote_run["id"])
    run = next(r for r in result["environments"][0]["runs"] if r["source"] == "R2")
    assert catalog.resolve(run["checkpoints"][0]["id"]).read_bytes() == b"first"
