"""Replayable run recipes and source/environment receipts."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import tarfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

from gymemu.resources import IS_CHECKOUT, PACKAGE_DIR, RESOURCE_ROOT

ROOT = RESOURCE_ROOT


def recipe_path(value):
    path = Path(value).expanduser().resolve()
    if path.suffix != ".yaml" or not path.is_file():
        raise ValueError(f"Recipe must be an existing .yaml file: {path}")
    return path


def recipe_arguments(arguments):
    """Translate a standalone YAML recipe into Hydra's native primary-config flags."""
    result, selected = [], None
    args = iter(arguments)
    for arg in args:
        if arg == "--recipe" or arg.startswith("--recipe="):
            if selected is not None:
                raise ValueError("Pass --recipe only once")
            value = next(args, "") if arg == "--recipe" else arg.split("=", 1)[1]
            selected = recipe_path(value)
        else:
            result.append(arg)
    if selected is None:
        return result
    conflicts = {"--config-path", "-cp", "--config-name", "-cn"}
    if any(arg.split("=", 1)[0] in conflicts for arg in result):
        raise ValueError("--recipe cannot be combined with --config-path or --config-name")
    return ["--config-path", str(selected.parent), "--config-name", selected.stem, *result]


def _freeze(raw, resolved, roots):
    """Keep internal tuning links, but freeze environment/resolver-dependent values."""
    if isinstance(raw, str) and "${" in raw:
        refs = re.findall(r"\$\{([^{}]+)\}", raw)
        remaining = re.sub(r"\$\{\w+(?:\.\w+)*\}", "", raw)
        if (
            refs
            and "${" not in remaining
            and all(re.fullmatch(r"\w+(?:\.\w+)*", r) and r.split(".")[0] in roots for r in refs)
        ):
            return raw
    if isinstance(resolved, dict) and isinstance(raw, dict):
        return {k: _freeze(raw.get(k), v, roots) for k, v in resolved.items()}
    if isinstance(resolved, list) and isinstance(raw, list):
        return [_freeze(raw[i], v, roots) for i, v in enumerate(resolved)]
    return resolved


def write_recipe(path, config, template=None):
    """Save a standalone recipe with a fresh output and the actual dataset revision."""
    raw = OmegaConf.to_container(template, resolve=False) if template is not None else config
    result = _freeze(raw, copy.deepcopy(config), set(config))
    result["output"] = None
    result["game"]["dataset"] = config["game"]["dataset"]
    result["game"]["revision"] = config["game"]["revision"]
    # These are output-routing defaults, not inherited model/training settings.
    result["hydra"] = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs/config.yaml").hydra, resolve=False
    )
    with Path(path).open("x") as stream:
        stream.write("# Standalone recipe. Replay with train.py --recipe <this file>.\n")
        stream.write(OmegaConf.to_yaml(OmegaConf.create(result)))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_reproduction(output, config, cfg, device, dataset_identity):
    write_recipe(output / "recipe.yaml", config, cfg)
    files = set(ROOT.glob("*.py"))
    files.update(PACKAGE_DIR.rglob("*.py"))
    files.update(path for path in (PACKAGE_DIR / "web_assets").rglob("*") if path.is_file())
    files.update((ROOT / "configs").rglob("*.yaml"))
    files.update((ROOT / "start_states").rglob("*.npz"))
    files.update(ROOT / name for name in ("pyproject.toml", "uv.lock", ".python-version"))
    files.update((ROOT / "containers/train").glob("*.py"))
    files.update((ROOT / "containers/train").glob("*.sh"))
    files.update((ROOT / "containers/train").glob("Dockerfile"))
    files.add(ROOT / ".dockerignore")
    files = sorted(path for path in files if path.is_file())
    source_hashes = {}
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for path in files:
            name = path.relative_to(ROOT).as_posix()
            source_hashes[name] = _hash(path)
            archive.add(path, arcname=name, recursive=False)

    def git(*args):
        if not IS_CHECKOUT:
            return None
        try:
            return subprocess.check_output(
                ["git", "-C", str(ROOT), *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    receipt = {
        "format_version": 1,
        "recipe_sha256": _hash(output / "recipe.yaml"),
        "source_archive_sha256": _hash(output / "source.tar.gz"),
        "source_files": source_hashes,
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain", "--untracked-files=normal"),
        "container": {
            "source_commit": os.environ.get("GYMEMU_IMAGE_SOURCE_SHA"),
            "image_ref": os.environ.get("GYMEMU_IMAGE_REF"),
        },
        "dataset": dataset_identity,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        },
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    (output / "reproduction.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt["recipe_sha256"]
