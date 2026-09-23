"""Published Runs are discovered through the catalog, then opened in Player."""

import hashlib
import json
from datetime import UTC, datetime
from io import BytesIO
from types import SimpleNamespace

import pytest

from gymemu.research_catalog import ResearchCatalog


class FakeR2:
    def __init__(self):
        self.objects = {}
        self.list_prefixes = []
        self.reads = []
        self.page_calls = []
        self.exceptions = SimpleNamespace(NoSuchKey=KeyError)

    def get_object(self, *, Bucket, Key):
        self.reads.append(Key)
        return {"Body": BytesIO(self.objects[Key])}

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return self

    def paginate(self, *, Bucket, Prefix):
        self.list_prefixes.append(Prefix)
        yield {
            "Contents": [
                {"Key": key, "LastModified": datetime.now(UTC)}
                for key in self.objects
                if key.startswith(Prefix)
            ]
        }

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys, ContinuationToken=None):
        self.page_calls.append((Prefix, MaxKeys, ContinuationToken))
        keys = sorted(key for key in self.objects if key.startswith(Prefix))
        start = keys.index(ContinuationToken) + 1 if ContinuationToken else 0
        selected = keys[start : start + MaxKeys]
        return {
            "Contents": [{"Key": key, "LastModified": datetime.now(UTC)} for key in selected],
            "NextContinuationToken": selected[-1] if start + MaxKeys < len(keys) else None,
        }

    def head_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"LastModified": datetime.now(UTC)}


def test_catalog_reads_projection_and_hides_recovery(tmp_path, monkeypatch):
    client = FakeR2()
    primary_id = "a" * 32
    prefix = f"runs/Breakout/{primary_id}"
    objects = {}
    for name in (
        "best.pt",
        "stages/representation/best.pt",
        "resume.pt",
        "start-scene.npz",
        "metrics.jsonl",
    ):
        payload = (
            b'{"stage":"prediction","epoch":1,"train":{"loss":0.2},"validation":{"loss":0.1,"mse":0.01}}\n'
            if name == "metrics.jsonl"
            else name.encode()
        )
        digest = hashlib.sha256(payload).hexdigest()
        key = f"{prefix}/objects/{digest}"
        client.objects[key] = payload
        if (name.endswith(".pt") and name != "resume.pt") or name == "start-scene.npz":
            client.objects[f"inference/{digest}"] = payload
        objects[name] = {"key": key, "sha256": digest, "size_bytes": len(payload)}
    manifest = {
        "version": 1,
        "bucket": "gymemu",
        "prefix": prefix,
        "run_id": primary_id,
        "objects": objects,
        "public_objects": {
            name: {
                "url": f"https://gymemu-assets.example.test/inference/{record['sha256']}",
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            }
            for name, record in objects.items()
            if (name.endswith(".pt") and name != "resume.pt") or name == "start-scene.npz"
        },
    }
    client.objects[f"{prefix}/manifest.json"] = json.dumps(manifest).encode()
    index = {
        "version": 1,
        "id": primary_id,
        "environment": "Breakout",
        "goal": "next-frame",
        "revision": "rev",
        "variant": None,
        "comparability": "cmp",
        "name": "run",
        "approach": "latent",
        "status": "complete",
        "best_mse": 0.01,
        "has_final_prediction": True,
        "manifest_key": f"{prefix}/manifest.json",
    }
    client.objects[f"runs/catalog/runs/{primary_id}.json"] = json.dumps(index).encode()
    for run_id, metric, comparison in (("b" * 32, 0.02, "cmp"), ("c" * 32, 0.001, "other")):
        other = {**index, "id": run_id, "best_mse": metric, "comparability": comparison}
        client.objects[f"runs/catalog/runs/{run_id}.json"] = json.dumps(other).encode()
    monkeypatch.setattr(
        "gymemu.research_catalog._open_public",
        lambda url: BytesIO(client.objects["/".join(url.split("/")[-2:])]),
    )
    catalog = ResearchCatalog(
        client=client, cache=tmp_path, public_base_url="https://gymemu-assets.example.test"
    )
    result = catalog.snapshot(run_id=primary_id)
    run = next(
        run
        for run in result["environments"][0]["goals"][0]["revisions"][0]["runs"]
        if run["id"] == primary_id
    )
    assert run["status"] == "complete" and run["best_mse"] == 0.01
    assert run["rank"] == 1
    assert {item["name"] for item in run["checkpoints"]} == {
        "best.pt",
        "stages/representation/best.pt",
    }
    assert run["recovery"] == ["resume.pt"]
    assert run["stage_history"][0]["validation"]["mse"] == 0.01
    assert client.list_prefixes == ["runs/catalog/runs/"]
    assert catalog.resolve(run["checkpoints"][0]["id"]).read_bytes() in {
        b"best.pt",
        b"stages/representation/best.pt",
    }
    assert all(not key.endswith(objects["best.pt"]["sha256"]) for key in client.reads)
    public_key = f"inference/{objects['best.pt']['sha256']}"
    client.objects[public_key] = b"corrupt"
    catalog.cache = tmp_path / "fresh"
    selected = next(item for item in run["checkpoints"] if item["name"] == "best.pt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        catalog.resolve(selected["id"])
    client.objects[public_key] = b"best.pt"
    manifest["public_objects"]["start-scene.npz"]["sha256"] = "0" * 64
    client.objects[f"{prefix}/manifest.json"] = json.dumps(manifest).encode()
    with pytest.raises(ValueError, match="scene is not published"):
        catalog.snapshot(run_id=primary_id, refresh=True)


def test_catalog_pages_run_projections_and_opens_direct_run(tmp_path):
    client = FakeR2()
    for letter in "abc":
        run_id = letter * 32
        client.objects[f"runs/catalog/runs/{run_id}.json"] = json.dumps(
            {
                "version": 1,
                "id": run_id,
                "environment": "Fixture-v0",
                "goal": "next-frame",
                "revision": "r1",
                "variant": "base",
                "name": run_id,
                "status": "complete",
                "manifest_key": f"runs/Fixture-v0/{run_id}/manifest.json",
            }
        ).encode()
    catalog = ResearchCatalog(client=client, cache=tmp_path)
    first = catalog.snapshot(page_size=1)
    assert first["next_cursor"]
    assert client.page_calls == [("runs/catalog/runs/", 1, None)]
    assert len([key for key in client.reads if "/catalog/runs/" in key]) == 1
    second = catalog.snapshot(page_size=1, cursor=first["next_cursor"])
    assert second["next_cursor"]
    third = catalog.snapshot(page_size=1, cursor=second["next_cursor"])
    assert third["next_cursor"] is None
    assert {
        run["id"]
        for page in (first, second, third)
        for env in page["environments"]
        for goal in env["goals"]
        for revision in goal["revisions"]
        for run in revision["runs"]
    } == {letter * 32 for letter in "abc"}
