"""Read-only, single-ball view of annotated Gradlab Breakout trajectories."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

FIELDS = (
    "ball_x_normalized",
    "ball_y_normalized",
    "ball_vx_normalized",
    "ball_vy_normalized",
    "paddle_x_normalized",
    "paddle_vx_normalized",
    "paddle_width_normalized",
)
STATE_SIZE = 115
NATIVE_SCALES = np.array([160, 255, 2, 27 / 8, 160, 160], np.float32)
FORMAT_VERSION = 1
RULES = {
    "state_fields": list(FIELDS),
    "bricks": [6, 18],
    "actions": "selected_action_json: requested policy action; 0=FIRE, 1=RIGHT, 2=LEFT",
    "terminal": "source active and (lives decreases or successor ball_y equals zero)",
    "exclude": "missing state, suspect brick grid, initial layout, waiting to serve",
    "censor": "original episode end, truncation, missing/invalid state; never label as death",
    "boundaries": "no histories or rollouts cross a life loss, quality gap, or episode end",
    "initial_state": "reset scalar labels unavailable; never fabricate them",
    "terminal_state_loss": "masked; terminal classification remains supervised",
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def decode_labels(record):
    """Decode only scalar data, never serialized Python objects."""
    tree = json.loads(json.loads(record)["structure"])
    if tree[0] != "dict":
        raise ValueError("Expected record dictionary")
    labels = dict(tree[1])["labels"]
    if labels[0] != "dict":
        raise ValueError("Expected label dictionary")
    return {key: value[1] for key, value in labels[1] if value[0] == "scalar"}


def state_values(labels, grid):
    """Return finite semantic state; signed velocity is intentionally supported."""
    values = np.array([labels.get(key, np.nan) for key in FIELDS], np.float32)
    bricks = np.asarray(grid, np.float32)
    if bricks.shape != (6, 18):
        raise ValueError("Expected a 6x18 brick grid")
    valid = bool(np.isfinite(values).all() and np.isin(bricks, [0, 1]).all())
    if valid:
        if not np.isin(np.round(values[6], 5), [0.75, 1.0]):
            raise ValueError("Unsupported paddle width")
        if np.any(np.abs(values[[2, 3, 5]]) > 1.00001):
            raise ValueError("Velocity outside documented normalized range")
        if np.any(values[[0, 1, 4]] < 0) or np.any(values[[0, 1, 4]] > 1.00001):
            raise ValueError("Position outside documented normalized range")
    return np.r_[values, bricks.ravel()].astype(np.float32), valid


def segment_episode(states, lives, usable, actions, boundaries, *, episode_id):
    """Yield aligned (states, actions, terminal, original source steps) segments.

    A terminal successor may already contain a new ball. It ends the old segment
    and may independently start the next; its state never enters the old history.
    """
    parts, run = [], []
    for t in range(len(actions)):
        loss = bool(
            usable[t]
            and np.isfinite(lives[t : t + 2]).all()
            and (lives[t + 1] < lives[t] or states[t + 1, 1] == 0)
        )
        allowed = usable[t] and (loss or usable[t + 1])
        # A reset/inconsistent lifecycle inside an episode is a gap, not gameplay.
        if np.isfinite(lives[t : t + 2]).all() and lives[t + 1] > lives[t]:
            allowed = False
        if allowed:
            run.append((t, loss))
        if run and (not allowed or loss or boundaries[t] or t == len(actions) - 1):
            first, last = run[0][0], run[-1][0]
            parts.append(
                {
                    "states": np.nan_to_num(states[first : last + 2]).copy(),
                    "actions": actions[first : last + 1].copy(),
                    "terminal": np.array([done for _, done in run], bool),
                    "episode_id": episode_id,
                    "first_step": first,
                }
            )
            run = []
    return parts


def episode_metadata(root, split):
    rows = []
    for path in sorted((root / "episodes" / split).glob("*.parquet")):
        rows.extend(
            pq.read_table(path, columns=["episode_id", "length", "initial_frame_id"]).to_pylist()
        )
    if not rows or len({r["episode_id"] for r in rows}) != len(rows):
        raise ValueError(f"Missing or duplicate episodes in {split}")
    return {r["episode_id"]: r for r in rows}


def choose_splits(root, config):
    train, heldout = episode_metadata(root, "train"), episode_metadata(root, "heldout")
    if set(train) & set(heldout):
        raise ValueError("Original episode IDs overlap between splits")
    rng = np.random.default_rng(config["split_seed"])
    order = rng.permutation(sorted(train)).tolist()
    nval = config["validation_episodes"] or max(1, len(order) // 5)
    ntrain = config["train_episodes"] or len(order) - nval
    ntest = config["test_episodes"] or len(heldout)
    if min(nval, ntrain, ntest) < 1 or nval + ntrain > len(train) or ntest > len(heldout):
        raise ValueError("Not enough episodes for requested disjoint splits")
    groups = {
        "train": order[nval : nval + ntrain],
        "validation": order[:nval],
        "test": rng.permutation(sorted(heldout))[:ntest].tolist(),
    }
    return groups, {**train, **heldout}


def read_state_episodes(root, split, selected, metadata):
    """Load only scalar/brick columns; images and reset RGB are never read."""
    episodes = {}
    for eid in selected:
        n = metadata[eid]["length"]
        episodes[eid] = {
            "states": np.zeros((n + 1, STATE_SIZE), np.float32),
            "lives": np.full(n + 1, np.nan, np.float32),
            "usable": np.zeros(n + 1, bool),
            "actions": np.zeros(n, np.int64),
            "boundaries": np.zeros(n, bool),
            "seen": np.zeros(n, bool),
            "sources": np.full(n, -1, np.int64),
            "successors": np.full(n, -1, np.int64),
        }
    columns = [
        "episode_id",
        "step",
        "source_frame_id",
        "successor_frame_id",
        "selected_action_json",
        "native_action_json",
        "action_override_rule_id",
        "configured_frame_skip",
        "record_json",
        "brick_grid",
        "brick_grid_suspect",
        "is_initial_brick_layout",
        "terminated",
        "truncated",
    ]
    counts = Counter()
    paths = sorted((root / "transitions" / split).glob("*.parquet"))
    for number, path in enumerate(paths):
        rows = pq.read_table(
            path, columns=columns, filters=[("episode_id", "in", selected)]
        ).to_pylist()
        for row in rows:
            eid, t = row["episode_id"], row["step"]
            e = episodes[eid]
            if not 0 <= t < len(e["actions"]) or e["seen"][t]:
                raise ValueError(f"Duplicate/out-of-range transition {eid}:{t}")
            if row["configured_frame_skip"] != 2:
                raise ValueError("This adapter requires the pinned frameskip-2 contract")
            action = json.loads(row["selected_action_json"])
            if type(action) is not int or action not in (0, 1, 2):
                raise ValueError("Requested actions must be scalar FIRE/RIGHT/LEFT IDs")
            if row["action_override_rule_id"] not in (None, "", "auto_serve"):
                raise ValueError("Unrecognized action override")
            if not row["action_override_rule_id"] and action != json.loads(
                row["native_action_json"]
            ):
                raise ValueError("Requested and executed action encodings disagree")
            labels = decode_labels(row["record_json"])
            state, valid = state_values(labels, row["brick_grid"])
            lives = labels.get("lives", np.nan)
            valid = valid and np.isfinite(lives)
            usable = (
                valid
                and state[1] > 0
                and not row["brick_grid_suspect"]
                and not row["is_initial_brick_layout"]
            )
            e["states"][t + 1], e["lives"][t + 1] = state, lives
            e["usable"][t + 1], e["actions"][t] = usable, action
            e["boundaries"][t] = row["terminated"] or row["truncated"]
            e["seen"][t] = True
            e["sources"][t], e["successors"][t] = row["source_frame_id"], row["successor_frame_id"]
            counts["rows"] += 1
            counts["unusable_successors"] += int(not usable)
        if number % 100 == 0:
            print(
                f"State adapter {split}: shard {number + 1}/{len(paths)}, "
                f"{counts['rows']} selected rows",
                flush=True,
            )
    segments = []
    for eid, e in episodes.items():
        if not e.pop("seen").all():
            raise ValueError(f"Missing transitions in episode {eid}")
        sources, successors = e.pop("sources"), e.pop("successors")
        if len(sources) and (
            sources[0] != metadata[eid]["initial_frame_id"]
            or not np.array_equal(sources[1:], successors[:-1])
        ):
            raise ValueError(f"Broken frame chain in episode {eid}")
        if e["boundaries"][:-1].any():
            raise ValueError(f"Episode {eid} continues after an original boundary")
        segments.extend(segment_episode(**e, episode_id=eid))
    return segments, dict(counts)


def save_segments(folder, segments):
    folder.mkdir(parents=True, exist_ok=False)
    count = sum(len(s["states"]) for s in segments)
    if not count:
        raise ValueError("No valid single-ball segments")
    specs = {
        "states": (np.float32, (count, STATE_SIZE)),
        "actions": (np.int64, (count,)),
        "terminal": (np.bool_, (count,)),
        "starts": (np.int64, (count,)),
        "ends": (np.int64, (count,)),
        "episode_ids": (np.int64, (count,)),
        "steps": (np.int64, (count,)),
    }
    arrays = {
        k: np.lib.format.open_memmap(folder / f"{k}.npy", mode="w+", dtype=d, shape=shape)
        for k, (d, shape) in specs.items()
    }
    offset, indices, receipt = 0, [], []
    for segment in segments:
        n = len(segment["actions"])
        sl = slice(offset, offset + n + 1)
        arrays["states"][sl] = segment["states"]
        arrays["actions"][sl] = np.r_[segment["actions"], 3]
        arrays["terminal"][sl] = np.r_[segment["terminal"], False]
        arrays["starts"][sl], arrays["ends"][sl] = offset, offset + n
        arrays["episode_ids"][sl] = segment["episode_id"]
        arrays["steps"][sl] = np.arange(segment["first_step"], segment["first_step"] + n + 1)
        indices.append(np.arange(offset, offset + n))
        receipt.append(
            {
                "episode_id": segment["episode_id"],
                "first_step": segment["first_step"],
                "transitions": n,
                "terminal": bool(segment["terminal"][-1]),
            }
        )
        offset += n + 1
    for array in arrays.values():
        array.flush()
    np.save(folder / "indices.npy", np.concatenate(indices))
    write_json(folder / "segments.json", receipt)
    return {
        "segments": len(segments),
        "transitions": int(sum(len(i) for i in indices)),
        "terminals": sum(s["terminal"] for s in receipt),
        "censored_segments": sum(not s["terminal"] for s in receipt),
    }


def prepare(root, output, config):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError(f"Cache already exists: {output}")
    manifest_path = root / "manifest.json"
    dataset_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    groups, metadata = choose_splits(root, config)
    output.mkdir(parents=True)
    manifest = {
        "format_version": FORMAT_VERSION,
        "rules": RULES,
        "data_config": config,
        "dataset": str(root.resolve()),
        "dataset_manifest_sha256": dataset_hash,
        "episode_ids": groups,
        "status": "preparing",
    }
    write_json(output / "manifest.json", manifest)
    for original, names in (("train", ("train", "validation")), ("heldout", ("test",))):
        selected = [eid for name in names for eid in groups[name]]
        segments, read_counts = read_state_episodes(root, original, selected, metadata)
        for name in names:
            selected_ids = set(groups[name])
            manifest[name] = save_segments(
                output / name, [s for s in segments if s["episode_id"] in selected_ids]
            )
        manifest[f"{original}_read"] = read_counts
    manifest["status"] = "complete"
    manifest["identity"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    write_json(output / "manifest.json", manifest)
    return manifest


class StateWindows:
    """Vectorized minibatches with independent state/action context and masked tails."""

    def __init__(self, cache, split):
        self.root = Path(cache)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if (
            self.manifest["status"] != "complete"
            or self.manifest["format_version"] != FORMAT_VERSION
            or self.manifest["rules"] != RULES
        ):
            raise ValueError("Incomplete or incompatible state cache")
        self.split = split
        folder = self.root / split
        self.arrays = {p.stem: np.load(p, mmap_mode="r") for p in folder.glob("*.npy")}
        self.indices = self.arrays["indices"]

    def select(self, count, seed):
        if count and count < len(self.indices):
            return np.sort(np.random.default_rng(seed).choice(self.indices, count, replace=False))
        return np.array(self.indices)

    def batch(self, indices, history, past_actions, horizon=1, device="cpu"):
        if history < 1 or past_actions < 0 or horizon < 1:
            raise ValueError("Invalid context or horizon")
        idx = np.asarray(indices)
        a = self.arrays
        lo, hi = a["starts"][idx], a["ends"][idx]
        h = idx[:, None] - np.arange(history - 1, -1, -1)
        hvalid = h >= lo[:, None]
        states = np.array(a["states"][np.maximum(h, lo[:, None])])
        states[~hvalid] = 0
        ah = idx[:, None] - np.arange(past_actions, -1, -1)
        actions = np.array(a["actions"][np.maximum(ah, lo[:, None])])
        actions[ah < lo[:, None]] = 3  # Padding is not a game action.
        future = idx[:, None] + np.arange(horizon)
        valid = future < hi[:, None]
        safe = np.minimum(future, hi[:, None])
        result = {
            "history": states,
            "history_valid": hvalid,
            "action_history": actions,
            "actions": a["actions"][safe],
            "valid": valid,
            "target": a["states"][np.minimum(future + 1, hi[:, None])],
            "terminal": a["terminal"][safe],
        }
        return {
            key: torch.as_tensor(np.array(value), device=device) for key, value in result.items()
        }
