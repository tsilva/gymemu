"""Published Runs are discovered through the catalog, then opened in Player."""

import hashlib
import json
from datetime import UTC, datetime
from io import BytesIO

from gymemu.research_catalog import ResearchCatalog


class FakeR2:
    def __init__(self):
        self.objects = {}
        self.list_prefixes = []

    def get_object(self, *, Bucket, Key):
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


def test_catalog_reads_projection_and_hides_recovery(tmp_path):
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
        objects[name] = {"key": key, "sha256": digest, "size_bytes": len(payload)}
    manifest = {
        "version": 1,
        "bucket": "gymemu",
        "prefix": prefix,
        "run_id": primary_id,
        "objects": objects,
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
    catalog = ResearchCatalog(client=client, cache=tmp_path)
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
