"""Local environment/run/checkpoint discovery without constructing inference models."""

import hashlib
import json
from pathlib import Path

import torch


def read_mapping(path):
    if not path.is_file():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path.name}")
    return value


def run_directory(checkpoint):
    parent = Path(checkpoint).parent
    return parent.parent.parent if parent.parent.name == "stages" else parent


class LocalCatalog:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.paths = {}
        self.metadata = {}

    def identifier(self, path):
        return hashlib.sha256(str(path.relative_to(self.root)).encode()).hexdigest()[:24]

    def config(self, directory, checkpoints):
        path = directory / "config.json"
        source = path if path.is_file() else next(iter(checkpoints), None)
        if source is None:
            return {}
        stamp = (str(source), source.stat().st_mtime_ns, source.stat().st_size)
        if stamp not in self.metadata:
            if source == path:
                value = read_mapping(path)
            else:
                # Older/copied runs can lack sidecars. Read tensor metadata onto the
                # meta device; do not allocate a model or execute checkpoint targets.
                value = torch.load(source, weights_only=True, map_location="meta")["config"]
            if not isinstance(value, dict):
                raise ValueError("Invalid checkpoint metadata")
            self.metadata = {k: v for k, v in self.metadata.items() if k[0] != str(source)}
            self.metadata[stamp] = value
        return self.metadata[stamp]

    def snapshot(self, run_id=None, refresh=False):
        groups, warnings, paths = {}, [], {}
        directories = {}
        for path in sorted(self.root.rglob("*.pt")):
            if path.is_symlink() or not path.resolve().is_relative_to(self.root):
                continue
            if path.is_file():
                directories.setdefault(run_directory(path), []).append(path)
        for path in self.root.rglob("config.json"):
            if path.resolve().is_relative_to(self.root) and not path.is_symlink():
                directories.setdefault(path.parent, [])
        for directory, checkpoints in directories.items():
            try:
                config = self.config(directory, checkpoints)
                game = config.get("game") or {}
                env_id = game.get("env_id") or "Unknown environment"
                summary = read_mapping(directory / "summary.json")
                approach = config.get("approach", {})
                approach = (
                    approach.get("kind", "direct") if isinstance(approach, dict) else approach
                )
                items = []
                for checkpoint in checkpoints:
                    stat = checkpoint.stat()
                    identifier = self.identifier(checkpoint)
                    paths[identifier] = checkpoint
                    items.append(
                        {
                            "id": identifier,
                            "name": checkpoint.relative_to(directory).as_posix(),
                            "modified": stat.st_mtime,
                            "size_bytes": stat.st_size,
                        }
                    )
                items.sort(key=lambda item: (-item["modified"], item["name"]))
                groups.setdefault(env_id, []).append(
                    {
                        "id": self.identifier(directory),
                        "name": directory.relative_to(self.root).as_posix(),
                        "approach": str(summary.get("approach") or approach),
                        "status": str(summary.get("status") or "Unknown"),
                        "modified": max((item["modified"] for item in items), default=0),
                        "checkpoints": items,
                    }
                )
            except Exception as error:
                warnings.append(f"{directory.relative_to(self.root)}: {error}")
        self.paths = paths
        return {
            "root": str(self.root),
            "environments": [
                {
                    "id": env_id,
                    "runs": sorted(runs, key=lambda run: (-run["modified"], run["name"])),
                }
                for env_id, runs in sorted(groups.items())
            ],
            "warnings": warnings,
        }

    def label(self, identifier):
        return str(self.resolve(identifier).relative_to(self.root))

    def resolve(self, identifier):
        if not isinstance(identifier, str) or identifier not in self.paths:
            raise ValueError("Unknown checkpoint. Refresh the catalog and select again.")
        path = self.paths[identifier]
        if path.is_symlink() or not path.resolve().is_relative_to(self.root) or not path.is_file():
            raise ValueError("Checkpoint is no longer available. Refresh the catalog.")
        return path


class CheckpointCatalog:
    """Combine local discovery with lazily listed R2 checkpoint history."""

    def __init__(self, root, remote=None):
        self.local = LocalCatalog(root)
        self.root = self.local.root
        self.remote = remote

    def snapshot(self, run_id=None, refresh=False):
        result = self.local.snapshot()
        result["remote"] = self.remote is not None
        groups = {env["id"]: env["runs"] for env in result["environments"]}
        for runs in groups.values():
            for run in runs:
                run["source"] = "Local"
                for checkpoint in run["checkpoints"]:
                    checkpoint["source"] = "Local"
        if self.remote is not None:
            try:
                runs, warnings = self.remote.snapshot(run_id=run_id, refresh=refresh)
                for run in runs:
                    groups.setdefault(run["env_id"], []).append(run)
                result["warnings"].extend(warnings)
            except Exception as error:
                result["warnings"].append(f"R2 unavailable: {error}. Local runs remain available.")
        result["environments"] = [
            {"id": env_id, "runs": sorted(runs, key=lambda run: (-run["modified"], run["name"]))}
            for env_id, runs in sorted(groups.items())
        ]
        return result

    def label(self, identifier):
        if identifier.startswith("r2:") and self.remote is not None:
            return self.remote.label(identifier)
        return self.local.label(identifier)

    def resolve(self, identifier):
        if isinstance(identifier, str) and identifier.startswith("r2:") and self.remote is not None:
            return self.remote.resolve(identifier)
        return self.local.resolve(identifier)
