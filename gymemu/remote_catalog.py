"""Read published R2 manifests and materialize verified playback bundles on demand."""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

from gymemu.storage import r2_client


def digest(value):
    return hashlib.sha256(value).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
        and path.as_posix() == name
    )


class RemoteCatalog:
    def __init__(self, *, bucket="gymemu", prefix="runs", cache=None, client=None):
        self.bucket, self.prefix = bucket, prefix.strip("/")
        self.cache = Path(cache or Path.home() / ".cache/gymemu/checkpoints")
        self.client = client
        self.manifests = {}
        self.json_objects = {}
        self.runs = {}
        self.paths = {}
        self.listed = False
        self.warnings = []

    def read_bytes(self, key):
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        stream = response["Body"]
        try:
            return stream.read()
        finally:
            stream.close()

    def manifest(self, entry):
        key, etag = entry["Key"], entry.get("ETag")
        cached = self.manifests.get(key)
        if cached and cached[0] == etag:
            return cached[1]
        value = json.loads(self.read_bytes(key))
        prefix = key.split("/manifests/")[0] if "/manifests/" in key else key.rsplit("/", 1)[0]
        if (
            not isinstance(value, dict)
            or value.get("version") != 1
            or value.get("bucket") != self.bucket
            or value.get("prefix") != prefix
            or not isinstance(value.get("env_id"), str)
            or not value["env_id"]
            or not isinstance(value.get("objects"), dict)
        ):
            raise ValueError(f"Invalid R2 manifest: {key}")
        for name, record in value["objects"].items():
            if (
                not safe_name(name)
                or not isinstance(record, dict)
                or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))
                or record.get("key") != f"{prefix}/objects/{record['sha256']}"
                or type(record.get("size_bytes")) is not int
                or record["size_bytes"] < 0
            ):
                raise ValueError(f"Invalid R2 artifact: {name}")
        self.manifests[key] = (etag, value)
        return value

    def metadata(self, manifest, name):
        record = manifest["objects"].get(name)
        if record is None:
            return {}
        key = record["key"]
        if key not in self.json_objects:
            payload = self.read_bytes(key)
            if digest(payload) != record["sha256"] or len(payload) != record["size_bytes"]:
                raise ValueError(f"R2 checksum mismatch: {name}")
            value = json.loads(payload)
            if not isinstance(value, dict):
                raise ValueError(f"Invalid R2 metadata: {name}")
            self.json_objects[key] = value
        return self.json_objects[key]

    def list_runs(self):
        if self.client is None:
            self.client = r2_client()
        entries = []
        for page in self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=f"{self.prefix}/"
        ):
            entries.extend(page.get("Contents", []))
        current = [e for e in entries if e["Key"].endswith("/manifest.json")]
        historical = {}
        for entry in entries:
            if "/manifests/" in entry["Key"] and entry["Key"].endswith(".json"):
                historical.setdefault(entry["Key"].split("/manifests/")[0], []).append(entry)
        runs, warnings = {}, []

        def load(entry):
            try:
                manifest = self.manifest(entry)
                summary = self.metadata(manifest, "summary.json")
                config = self.metadata(manifest, "config.json")
                identifier = "r2-run:" + digest(f"{self.bucket}/{manifest['prefix']}".encode())[:24]
                approach = config.get("approach") or {}
                approach = (
                    approach.get("kind", "Unknown") if isinstance(approach, dict) else approach
                )
                return (
                    identifier,
                    {
                        "id": identifier,
                        "name": str(
                            summary.get("name") or manifest.get("run_id") or manifest["prefix"]
                        ),
                        "run_id": str(manifest.get("run_id", "")),
                        "source": "R2",
                        "env_id": manifest["env_id"],
                        "approach": str(summary.get("approach") or approach),
                        "status": str(manifest.get("status", "Unknown")),
                        "modified": entry["LastModified"].timestamp(),
                        "entry": entry,
                        "manifest": manifest,
                        "historical": historical.get(manifest["prefix"], []),
                        "checkpoints": None,
                    },
                    None,
                )
            except Exception as error:
                return None, None, f"R2 {entry['Key']}: {error}"

        with ThreadPoolExecutor(max_workers=8) as pool:
            for identifier, run, warning in pool.map(load, current):
                if warning:
                    warnings.append(warning)
                else:
                    runs[identifier] = run
        self.runs, self.warnings, self.listed = runs, warnings, True

    def checkpoints(self, run):
        if run["checkpoints"] is not None:
            return run["checkpoints"]
        versions = [(run["entry"], run["manifest"], True)]

        def load(entry):
            try:
                return entry, self.manifest(entry), False
            except Exception as error:
                return entry, str(error), False

        with ThreadPoolExecutor(max_workers=8) as pool:
            versions.extend(pool.map(load, run["historical"]))
        items = {}
        for entry, manifest, current in versions:
            if isinstance(manifest, str):
                self.warnings.append(f"R2 {entry['Key']}: {manifest}")
                continue
            for name, record in manifest["objects"].items():
                if not name.endswith(".pt"):
                    continue
                scene = manifest["objects"].get("start-scene.npz", {}).get("sha256", "")
                identifier = "r2:" + digest(
                    f"{self.bucket}/{manifest['prefix']}/{name}/{record['sha256']}/{scene}".encode()
                )
                item = {
                    "id": identifier,
                    "name": name if current else f"{name} · {record['sha256'][:10]}",
                    "source": "R2" if current else "R2 history",
                    "modified": entry["LastModified"].timestamp(),
                    "size_bytes": record["size_bytes"],
                }
                if identifier not in items or (
                    not current
                    and items[identifier]["source"] != "R2"
                    and item["modified"] < items[identifier]["modified"]
                ):
                    items[identifier] = item
                    self.paths[identifier] = (manifest, name)
        run["checkpoints"] = sorted(
            items.values(), key=lambda item: (-item["modified"], item["name"])
        )
        return run["checkpoints"]

    def snapshot(self, run_id=None, refresh=False):
        if refresh or not self.listed:
            self.list_runs()
        result = []
        for run in self.runs.values():
            selected = run["id"] == run_id
            checkpoints = self.checkpoints(run) if selected else (run["checkpoints"] or [])
            count = (
                len(checkpoints)
                if run["checkpoints"] is not None
                else sum(name.endswith(".pt") for name in run["manifest"]["objects"])
            )
            result.append(
                {
                    **{
                        key: run[key]
                        for key in (
                            "id",
                            "name",
                            "source",
                            "env_id",
                            "approach",
                            "status",
                            "modified",
                            "run_id",
                        )
                    },
                    "checkpoints": checkpoints,
                    "checkpoint_count": count,
                }
            )
        return result, self.warnings

    def download(self, record, path):
        if path.is_file():
            with path.open("rb") as stream:
                if (
                    path.stat().st_size == record["size_bytes"]
                    and hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"]
                ):
                    return
        path.parent.mkdir(parents=True, exist_ok=True)
        response = self.client.get_object(Bucket=self.bucket, Key=record["key"])
        stream, temporary = response["Body"], None
        try:
            checksum, size = hashlib.sha256(), 0
            with NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                while chunk := stream.read(1024 * 1024):
                    checksum.update(chunk)
                    size += len(chunk)
                    output.write(chunk)
            if checksum.hexdigest() != record["sha256"] or size != record["size_bytes"]:
                raise ValueError(f"R2 checksum mismatch: {path.name}")
            temporary.replace(path)
        finally:
            stream.close()
            if temporary:
                temporary.unlink(missing_ok=True)

    def label(self, identifier):
        manifest, name = self.paths[identifier]
        run = next(
            run for run in self.runs.values() if run["manifest"]["prefix"] == manifest["prefix"]
        )
        version = manifest["objects"][name]["sha256"][:10]
        return f"{run['name']} [{run['run_id'][:8]}] / {name} · {version}"

    def resolve(self, identifier):
        if identifier not in self.paths:
            raise ValueError("Unknown R2 checkpoint. Refresh and select again.")
        manifest, name = self.paths[identifier]
        bundle = digest(json.dumps(manifest, sort_keys=True).encode())
        directory = self.cache / bundle
        for relative in (name, "start-scene.npz"):
            record = manifest["objects"].get(relative)
            if record:
                self.download(record, directory / relative)
        return directory / name
