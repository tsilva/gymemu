"""Reconstruct source-time internal inputs for controlled, isolated diagnostics."""

import bisect
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from gymemu.state_data import StateWindows, decode_labels, write_json
from gymemu.state_probe_diagnostics import event_masks

HIDDEN_FIELDS = ("paddle_charge", "paddle_measure", "paddle_repeat", "paddle_held", "paddle_hits")
HIDDEN_SCALES = np.array([3856, 235, 60, 1, 12], np.float32)
NATIVE_COMMIT = "f06b9c81f4d717fd32782c5d877bb79a9042d266"


class PaddleController:
    """Diagnostic transcription of the pinned native controller, not playback code."""

    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.lower = [x[0] for x in thresholds]
        self.x, self.vx = 115, 0
        self.charge, self.measure, self.repeat, self.held = 2048, 162, 0, False

    def state(self):
        return [self.charge, self.measure, self.repeat, int(self.held)]

    def step(self, action):
        new = max(55, min(191, (self.x + 47 + 235 - self.measure) // 2)) - 47
        self.vx, self.x = new - self.x, new
        if self.held:
            self.repeat += 1
            if self.repeat > 5:
                self.repeat = 60
        if action == 2 and self.charge > self.repeat:
            self.charge -= self.repeat
        elif action == 3 and self.charge + self.repeat < 3856:
            self.charge += self.repeat
        self.held = action in (2, 3)
        self.measure = self.thresholds[max(0, bisect.bisect_right(self.lower, self.charge) - 1)][1]


def controller_native_frames(previous_labels):
    """The pinned game stops frameskip early if its first frame ends the last life.

    Uses only the preceding record's source-state labels, never successor labels.
    Native loss threshold 217 minus the RAM-coordinate offset 9 equals 208.
    """
    if (
        previous_labels
        and previous_labels.get("lives") == 1
        and previous_labels.get("ball_y", 0) >= 208
    ):
        return 1
    return 2


def source_hit_counts(data):
    """Count prior observed bounces. Never include the transition being predicted."""
    a, ii = data.arrays, data.indices
    values = np.zeros(len(a["states"]), np.float32)
    old, y = a["states"][ii], a["states"][ii + 1]
    bounce = event_masks(old, y, a["terminal"][ii])["paddle_collision_proxy"]
    last, count = -1, 0
    for j, i in enumerate(ii):
        start = int(a["starts"][i])
        if start != last:
            velocity = a["states"][i, 2:4] * np.array([2, 3.375])
            if not np.allclose(np.abs(velocity), [1, 1]) or velocity[1] <= 0:
                raise ValueError("Cannot initialize hit count at a non-serve segment start")
            count, last = 0, start
        values[i] = count
        if bounce[j]:
            count = min(count + 1, 12)
    return values


def prepare_hidden_inputs(cache, output, native_source, *, test_only=False):
    """Read original train/validation records; write a separate, provenance-bound input cache."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    splits = ("test",) if test_only else ("train", "validation")
    data = {s: StateWindows(cache, s) for s in splits}
    manifest = next(iter(data.values())).manifest
    original_split = "heldout" if test_only else "train"
    root = Path(manifest["dataset"])
    digest = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    if digest != manifest["dataset_manifest_sha256"]:
        raise ValueError("Original dataset provenance differs from the state cache")
    source_hash = hashlib.sha256(Path(native_source).read_bytes()).hexdigest()
    if source_hash != "a2a47d8ec876657697cb2113ddcb5545bfd8ed3d2c394649423759247213e4c7":
        raise ValueError("Expected the pinned native source snapshot")
    source = Path(native_source).read_text()
    block = source.split("const PADDLE_MEASURE_THRESHOLDS:")[1].split("];")[0]
    thresholds = [tuple(map(int, p)) for p in re.findall(r"\((\d+),\s*(\d+)\)", block)]
    if len(thresholds) != 89:
        raise ValueError("Unexpected controller threshold table")
    ids = [eid for split in splits for eid in manifest["episode_ids"][split]]
    metadata = {}
    for p in (root / "episodes" / original_split).glob("*.parquet"):
        for row in pq.read_table(
            p, columns=["episode_id", "seed", "length"], filters=[("episode_id", "in", ids)]
        ).to_pylist():
            metadata[row["episode_id"]] = row
    records = {i: [None] * metadata[i]["length"] for i in ids}
    for n, p in enumerate(sorted((root / "transitions" / original_split).glob("*.parquet"))):
        rows = pq.read_table(
            p,
            columns=["episode_id", "step", "native_action_json", "record_json"],
            filters=[("episode_id", "in", ids)],
        ).to_pylist()
        for row in rows:
            slot = records[row["episode_id"]]
            if slot[row["step"]] is not None:
                raise ValueError("Duplicate source transition")
            action = json.loads(row["native_action_json"])
            if type(action) is not int or action not in (0, 1, 2):
                raise ValueError("Unexpected custom-discrete native action")
            slot[row["step"]] = (action + 1, decode_labels(row["record_json"]))
        if n % 200 == 0:
            print("Hidden input source shard", n + 1, flush=True)
    latent = {}
    checked = 0
    early_stops = 0
    for eid, rows in records.items():
        controller = PaddleController(thresholds)
        noops = int(np.random.default_rng(metadata[eid]["seed"]).integers(1, 31, dtype=np.uint64))
        for _ in range(noops):
            controller.step(0)
        values = [controller.state()]
        previous_labels = None
        for t, record in enumerate(rows):
            if record is None:
                raise ValueError("Missing source transition")
            action, lab = record
            native_frames = controller_native_frames(previous_labels)
            early_stops += int(native_frames == 1)
            for _ in range(native_frames):
                controller.step(action)
            if controller.x != lab["paddle_x"] / 65536 or controller.vx != round(
                lab["paddle_vx_normalized"] * 160
            ):
                raise ValueError(f"Controller replay does not match source {eid}:{t}")
            checked += 1
            values.append(controller.state())
            previous_labels = lab
        latent[eid] = np.array(values, np.float32)
    report = {
        "format": "gymemu-hidden-inputs-v1",
        "cache_identity": manifest["identity"],
        "dataset_manifest_sha256": digest,
        "native_commit": NATIVE_COMMIT,
        "native_source_sha256": hashlib.sha256(Path(native_source).read_bytes()).hexdigest(),
        "fields": list(HIDDEN_FIELDS),
        "scales": HIDDEN_SCALES.tolist(),
        "controller_transitions_verified": checked,
        "controller_mismatches": 0,
        "last_life_one_frame_stops": early_stops,
        "controller_timing": "source state; past actions and loss labels set frame count",
        "hit_count_timing": "prior bounces only; cap 12; zero at verified serve start",
        "life_boundary": "controller persists; hit count and observation history reset",
        "test_read": test_only,
        "files": {},
    }
    for split, d in data.items():
        a = d.arrays
        features = np.zeros((len(a["states"]), 5), np.float32)
        for eid in manifest["episode_ids"][split]:
            pick = np.flatnonzero(a["episode_ids"] == eid)
            features[pick, :4] = latent[eid][a["steps"][pick]]
        features[:, 4] = source_hit_counts(d)
        features /= HIDDEN_SCALES
        if not np.isfinite(features).all() or np.any(features < 0) or np.any(features > 1):
            raise ValueError("Invalid normalized hidden inputs")
        path = output / f"{split}.npy"
        np.save(path, features)
        report["files"][split] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output / "manifest.json", report)
    return report


class HiddenInputs:
    def __init__(self, data, path):
        path = Path(path)
        report = json.loads((path / "manifest.json").read_text())
        if (
            report["format"] != "gymemu-hidden-inputs-v1"
            or report["cache_identity"] != data.manifest["identity"]
        ):
            raise ValueError("Hidden input provenance differs from state cache")
        filename = path / f"{data.split}.npy"
        if hashlib.sha256(filename.read_bytes()).hexdigest() != report["files"][data.split]:
            raise ValueError("Hidden input checksum mismatch")
        self.values = np.load(filename, mmap_mode="r", allow_pickle=False)
        if self.values.shape != (len(data.arrays["states"]), 5):
            raise ValueError("Hidden input shape mismatch")
        self.provenance = report

    def attach(self, indices, batch):
        batch["hidden_state"] = torch.from_numpy(np.array(self.values[np.asarray(indices)]))


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument(
        "--test-only", action="store_true", help="Prepare a separate final-test input cache"
    )
    args = parser.parse_args()
    report = prepare_hidden_inputs(
        args.cache, args.output, args.native_source, test_only=args.test_only
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
