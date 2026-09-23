"""Immutable research contracts and content identities."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from gymemu.resources import RESOURCE_ROOT

GOALS = RESOURCE_ROOT / "experiments" / "goals"


def identity(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _diff(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(before.keys() | after.keys()):
            path = f"{prefix}.{key}" if prefix else key
            result.extend(_diff(before.get(key), after.get(key), path))
        return result
    return [] if before == after else [{"path": prefix, "before": before, "after": after}]


def _override_leaves(value, prefix=""):
    for key, selected in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(selected, dict):
            yield from _override_leaves(selected, path)
        else:
            yield path, selected


def resolve_goal(checked_in, overrides=None):
    """Return the checked-in revision and optional launch-time variant."""
    source = copy.deepcopy(checked_in)
    selected = copy.deepcopy(source)
    for dotted, value in _override_leaves(overrides or {}):
        keys = dotted.split(".")
        target = selected
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise ValueError(f"Unknown Goal field: {dotted}")
            target = target[key]
        if keys[-1] not in target:
            raise ValueError(f"Unknown Goal field: {dotted}")
        target[keys[-1]] = value
    diff = _diff(source, selected)
    comparison = {
        "dataset": selected["dataset"],
        "evaluation": selected["evaluation"],
    }
    return {
        "id": selected["id"],
        "environment": selected["environment"],
        "revision": identity(source),
        "variant": identity(selected) if diff else None,
        "diff": diff,
        "contract": selected,
        "comparability": identity(comparison),
    }


def load_goal(environment, goal):
    path = GOALS / environment / goal / "goal.yaml"
    if not path.is_file():
        raise ValueError(f"Unknown Research Goal: {environment}/{goal}")
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or value.get("id") != goal:
        raise ValueError(f"Invalid Research Goal: {path}")
    return value


def goal_recipes(environment, goal):
    directory = GOALS / environment / goal / "recipes"
    return sorted(path.stem for path in directory.glob("*.yaml"))


def goal_path(environment, goal):
    return Path(GOALS / environment / goal / "goal.yaml")


def checked_in_goals():
    for path in sorted(GOALS.glob("*/*/goal.yaml")):
        value = yaml.safe_load(path.read_text())
        if isinstance(value, dict):
            yield value, goal_recipes(path.parent.parent.name, path.parent.name)
