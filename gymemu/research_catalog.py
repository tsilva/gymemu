"""Read the private R2 Run projection and verified inference bundles."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import HTTPRedirectHandler, Request, build_opener

import yaml

from gymemu.remote_catalog import RemoteCatalog, safe_name
from gymemu.research import checked_in_goals, resolve_goal
from gymemu.storage import r2_client


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, stream, code, message, headers, new_url):
        return None


def _open_public(url):
    request = Request(url, headers={"User-Agent": "Gymemu/1.0"})
    return build_opener(_NoRedirect()).open(request, timeout=30)


class ResearchCatalog(RemoteCatalog):
    def __init__(
        self, *, bucket="gymemu", prefix="runs", cache=None, client=None, public_base_url=None
    ):
        super().__init__(bucket=bucket, prefix=prefix, cache=cache, client=client)
        self.root = "R2 catalog"
        self.public_base_url = (
            public_base_url
            or os.environ.get("GYMEMU_PUBLIC_R2_BASE_URL")
            or "https://gymemu-assets.tsilva.eu"
        ).rstrip("/")
        self.entries = None
        self.checkpoint_paths = {}

    def _index(self, key):
        value = json.loads(self.read_bytes(key))
        if (
            value.get("version") != 1
            or not re.fullmatch(r"[0-9a-f]{32}", str(value.get("id", "")))
            or value.get("id") != Path(key).stem
            or value.get("manifest_key", "").startswith(f"{self.prefix}/") is False
        ):
            raise ValueError(f"Invalid catalog entry: {key}")
        return value

    def _list(self):
        if self.client is None:
            self.client = r2_client()
        rows = []
        for page in self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=f"{self.prefix}/catalog/runs/"
        ):
            for entry in page.get("Contents", []):
                if entry["Key"].endswith(".json"):
                    value = self._index(entry["Key"])
                    value["modified"] = entry["LastModified"].timestamp()
                    rows.append(value)
        self.entries = rows

    def _list_page(self, *, cursor=None, page_size=50, run_id=None):
        if self.client is None:
            self.client = r2_client()
        if type(page_size) is not int or not 1 <= page_size <= 100:
            raise ValueError("Catalog page size must be between 1 and 100")
        if cursor is not None and (not isinstance(cursor, str) or len(cursor) > 2048):
            raise ValueError("Invalid catalog cursor")
        options = {
            "Bucket": self.bucket,
            "Prefix": f"{self.prefix}/catalog/runs/",
            "MaxKeys": page_size,
        }
        if cursor:
            options["ContinuationToken"] = cursor
        response = self.client.list_objects_v2(**options)
        rows = []
        for entry in response.get("Contents", []):
            if entry["Key"].endswith(".json"):
                value = self._index(entry["Key"])
                value["modified"] = entry["LastModified"].timestamp()
                rows.append(value)
        if run_id and all(row["id"] != run_id for row in rows):
            if not re.fullmatch(r"[0-9a-f]{32}", run_id):
                raise ValueError("Invalid Run ID")
            key = f"{self.prefix}/catalog/runs/{run_id}.json"
            try:
                row = self._index(key)
                row["modified"] = self.client.head_object(Bucket=self.bucket, Key=key)[
                    "LastModified"
                ].timestamp()
                rows.append(row)
            except self.client.exceptions.NoSuchKey:
                pass
        return rows, response.get("NextContinuationToken")

    def _checkpoints(self, row):
        manifest = json.loads(self.read_bytes(row["manifest_key"]))
        if (
            manifest.get("bucket") != self.bucket
            or manifest.get("run_id") != row["id"]
            or row["manifest_key"] != f"{manifest.get('prefix')}/manifest.json"
        ):
            raise ValueError("Run manifest does not match the catalog entry")
        if not isinstance(manifest.get("objects"), dict):
            raise ValueError("Run manifest has no artifact map")
        public = manifest.get("public_objects", {})
        if not isinstance(public, dict):
            raise ValueError("Run manifest has no public artifact map")
        if "start-scene.npz" in manifest["objects"]:
            scene = manifest["objects"]["start-scene.npz"]
            published_scene = public.get("start-scene.npz", {})
            if (
                published_scene.get("sha256") != scene.get("sha256")
                or published_scene.get("size_bytes") != scene.get("size_bytes")
                or published_scene.get("url")
                != f"{self.public_base_url}/inference/{scene['sha256']}"
            ):
                raise ValueError("Run playback scene is not published or differs from manifest")
        for name, record in manifest["objects"].items():
            if (
                not safe_name(name)
                or not isinstance(record, dict)
                or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))
                or record.get("key") != f"{manifest['prefix']}/objects/{record['sha256']}"
                or type(record.get("size_bytes")) is not int
                or record["size_bytes"] < 0
            ):
                raise ValueError(f"Invalid Run artifact: {name}")
        checkpoints, recovery = [], []
        details = {}

        def artifact(name):
            record = manifest["objects"].get(name)
            if record is None:
                return None
            payload = self.read_bytes(record["key"])
            if (
                len(payload) != record["size_bytes"]
                or hashlib.sha256(payload).hexdigest() != record["sha256"]
            ):
                raise ValueError(f"Run artifact checksum mismatch: {name}")
            return payload

        run_payload = artifact("run.json")
        if run_payload:
            record = json.loads(run_payload)
            details = {
                "goal_yaml": yaml.safe_dump((record.get("goal") or {}).get("contract")),
                "config_yaml": yaml.safe_dump(record.get("recipe", {}).get("resolved")),
                "launch_overrides": record.get("recipe", {}).get("overrides", []),
                "source": record.get("source"),
                "compute": record.get("compute"),
                "stages": record.get("stages"),
                "evaluation": record.get("evaluation"),
            }
        metrics = artifact("metrics.jsonl")
        if metrics:
            details["stage_history"] = [json.loads(line) for line in metrics.splitlines()]
        diagnostics = artifact("diagnostics.jsonl")
        if diagnostics:
            details["probe_history"] = [json.loads(line) for line in diagnostics.splitlines()]
        probe = artifact("probe.json")
        if probe:
            details["probe_manifest"] = json.loads(probe)
        for name, record in manifest["objects"].items():
            if not name.endswith(".pt"):
                continue
            if name == "resume.pt" or "/resume" in name:
                recovery.append(name)
                continue
            public_record = public.get(name)
            if not public_record:
                continue
            if (
                public_record.get("sha256") != record["sha256"]
                or public_record.get("size_bytes") != record["size_bytes"]
                or public_record.get("url")
                != f"{self.public_base_url}/inference/{record['sha256']}"
            ):
                raise ValueError(f"Invalid public inference artifact: {name}")
            identifier = (
                "r2:"
                + hashlib.sha256(f"{row['id']}/{name}/{record['sha256']}".encode()).hexdigest()
            )
            self.checkpoint_paths[identifier] = (manifest, name)
            checkpoints.append(
                {
                    "id": identifier,
                    "name": name,
                    "stage": name.split("/")[1] if name.startswith("stages/") else None,
                    "final": not name.startswith("stages/"),
                    "size_bytes": record["size_bytes"],
                    "modified": row["modified"],
                }
            )
        return checkpoints, sorted(recovery), details

    def snapshot(self, run_id=None, refresh=False, cursor=None, page_size=None):
        next_cursor = None
        if page_size is not None:
            self.entries, next_cursor = self._list_page(
                cursor=cursor, page_size=page_size, run_id=run_id
            )
        elif refresh or self.entries is None:
            self._list()
        comparable = {}
        for row in self.entries:
            if (
                row.get("comparability")
                and row.get("status") == "complete"
                and row.get("has_final_prediction")
                and isinstance(row.get("best_mse"), (int, float))
            ):
                comparable.setdefault(row["comparability"], []).append(row)
        ranks = {
            row["id"]: rank
            for rows in comparable.values()
            for rank, row in enumerate(
                sorted(rows, key=lambda item: (item["best_mse"], item["id"])), start=1
            )
        }
        environments = {}
        contracts = {}
        differences = {}
        for contract, recipes in checked_in_goals():
            selected = resolve_goal(contract)
            env_id, goal_id = selected["environment"], selected["id"]
            environments.setdefault(env_id, {}).setdefault(goal_id, {}).setdefault(
                selected["revision"], {}
            )
            contracts[(env_id, goal_id)] = {"current": selected["revision"], "recipes": recipes}
        for row in self.entries:
            environment = environments.setdefault(row["environment"], {})
            goal = environment.setdefault(row.get("goal") or "Unassigned", {})
            revision = goal.setdefault(row.get("revision") or "Unversioned", {})
            variant = row.get("variant") or "base"
            differences[(row["environment"], row.get("goal"), row.get("revision"), variant)] = (
                row.get("goal_diff") or []
            )
            runs = revision.setdefault(variant, [])
            checkpoints, recovery, details = (
                self._checkpoints(row) if row["id"] == run_id else ([], [], {})
            )
            runs.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "approach": row.get("approach"),
                    "status": row["status"],
                    "best_mse": row.get("best_mse"),
                    "rank": ranks.get(row["id"]),
                    "comparability": row.get("comparability"),
                    "modified": row["modified"],
                    "created": datetime.fromisoformat(row["created_at"]).timestamp()
                    if row.get("created_at")
                    else row["modified"],
                    "wandb_url": row.get("wandb_url"),
                    "checkpoints": checkpoints,
                    "recovery": recovery,
                    "checkpoint_count": row.get("checkpoint_count", len(checkpoints)),
                    "has_final_prediction": row.get("has_final_prediction", False),
                    **details,
                }
            )
            if row.get("goal_contract"):
                contracts.setdefault((row["environment"], row["goal"]), {})[
                    row.get("revision") or "Unversioned"
                ] = row["goal_contract"]
        result = []
        for env_id, goals in sorted(environments.items()):
            items = []
            for goal_id, revisions in sorted(goals.items()):
                versions = []
                for revision_id, variants in sorted(revisions.items()):
                    versions.append(
                        {
                            "id": revision_id,
                            "variants": [
                                {
                                    "id": variant,
                                    "runs": sorted(runs, key=lambda run: -run["modified"]),
                                    "run_count": len(runs),
                                    "first_activity": min(run["created"] for run in runs),
                                    "last_activity": max(run["modified"] for run in runs),
                                    "diff": differences.get(
                                        (env_id, goal_id, revision_id, variant), []
                                    ),
                                }
                                for variant, runs in sorted(variants.items())
                            ],
                            "runs": sorted(
                                [run for runs in variants.values() for run in runs],
                                key=lambda run: -run["modified"],
                            ),
                        }
                    )
                details = contracts.get((env_id, goal_id), {})
                items.append(
                    {
                        "id": goal_id,
                        "current": details.get("current"),
                        "recipes": details.get("recipes", []),
                        "revisions": versions,
                    }
                )
            result.append({"id": env_id, "goals": items})
        return {"environments": result, "warnings": [], "next_cursor": next_cursor}

    def label(self, identifier):
        manifest, name = self.checkpoint_paths[identifier]
        return f"{manifest['run_id']} / {name}"

    def resolve(self, identifier):
        if identifier not in self.checkpoint_paths:
            raise ValueError("Unknown published Checkpoint. Refresh the catalog.")
        manifest, name = self.checkpoint_paths[identifier]
        directory = self.cache / manifest["run_id"] / manifest["objects"][name]["sha256"]
        for relative in (name, "start-scene.npz"):
            record = manifest.get("public_objects", {}).get(relative)
            if record:
                private = manifest["objects"].get(relative)
                if (
                    private is None
                    or record.get("sha256") != private["sha256"]
                    or record.get("size_bytes") != private["size_bytes"]
                ):
                    raise ValueError(f"Public artifact differs from Run manifest: {relative}")
                self._download_public(record, directory / relative)
        return directory / name

    def _download_public(self, record, path):
        expected = f"{self.public_base_url}/inference/{record['sha256']}"
        if record["url"] != expected:
            raise ValueError("Public artifact URL differs from the configured bucket domain")
        if path.is_file():
            with path.open("rb") as stream:
                if (
                    path.stat().st_size == record["size_bytes"]
                    and hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"]
                ):
                    return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            digest, size = hashlib.sha256(), 0
            with _open_public(record["url"]) as source:
                with NamedTemporaryFile(dir=path.parent, delete=False) as output:
                    temporary = Path(output.name)
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
                        output.write(chunk)
            if digest.hexdigest() != record["sha256"] or size != record["size_bytes"]:
                raise ValueError(f"Public artifact checksum mismatch: {path.name}")
            temporary.replace(path)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
