"""Recover valid paddle observations before brick-quality training boundaries."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from gymemu.state_data import FIELDS, decode_labels, episode_metadata


class RecordedPaddleContext:
    """Source/past context only; life losses still clear the history.

    Startup brick annotations do not invalidate recorded paddle measurements.
    The original source is read but never changed. Target eligibility stays intact.
    """

    def __init__(self, data, length):
        if length < 1:
            raise ValueError("Paddle context must be positive")
        self.length = length
        root = Path(data.manifest["dataset"])
        digest = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
        if digest != data.manifest["dataset_manifest_sha256"]:
            raise ValueError("Paddle context source provenance differs from state cache")
        original = "heldout" if data.split == "test" else "train"
        ids = data.manifest["episode_ids"][data.split]
        metadata = episode_metadata(root, original)
        offset = 0
        offsets = {}
        for eid in ids:
            offsets[eid] = offset
            offset += metadata[eid]["length"] + 1
        self.states = np.zeros((offset, 3), np.float32)
        self.valid = np.zeros(offset, bool)
        self.actions = np.full(offset, 3, np.int64)
        self.starts = np.zeros(offset, np.int64)
        lives = np.full(offset, np.nan, np.float32)
        ball_y = np.full(offset, np.nan, np.float32)
        seen = np.zeros(offset, bool)
        for number, path in enumerate(sorted((root / "transitions" / original).glob("*.parquet"))):
            rows = pq.read_table(
                path,
                columns=["episode_id", "step", "selected_action_json", "record_json"],
                filters=[("episode_id", "in", ids)],
            ).to_pylist()
            for row in rows:
                eid, t = row["episode_id"], row["step"]
                j = offsets[eid] + t
                if not 0 <= t < metadata[eid]["length"] or seen[j]:
                    raise ValueError("Invalid paddle context transition alignment")
                lab = decode_labels(row["record_json"])
                value = np.asarray([lab.get(k, np.nan) for k in FIELDS[4:7]], np.float32)
                self.valid[j + 1] = np.isfinite(value).all()
                self.states[j + 1] = np.nan_to_num(value)
                lives[j + 1] = lab.get("lives", np.nan)
                ball_y[j + 1] = lab.get("ball_y_normalized", np.nan)
                action = json.loads(row["selected_action_json"])
                if type(action) is not int or action not in (0, 1, 2):
                    raise ValueError("Invalid requested action in paddle context")
                self.actions[j] = action
                seen[j] = True
            if number % 200 == 0:
                print("Paddle context", data.split, "shard", number + 1, flush=True)
        for eid in ids:
            start = offsets[eid]
            end = start + metadata[eid]["length"]
            lower = start
            if not seen[start:end].all():
                raise ValueError("Missing paddle context transitions")
            for j in range(start, end + 1):
                if j > start + 1 and (
                    not np.isfinite(lives[j - 1 : j + 1]).all()
                    or lives[j] != lives[j - 1]
                    or (ball_y[j - 1] > 0 and ball_y[j] == 0)
                ):
                    lower = j
                if j > start and not self.valid[j]:
                    lower = j
                self.starts[j] = lower
        a = data.arrays
        self.source_rows = np.fromiter(
            (
                offsets[int(eid)] + int(step)
                for eid, step in zip(a["episode_ids"], a["steps"], strict=True)
            ),
            dtype=np.int64,
            count=len(a["steps"]),
        )
        current = self.states[self.source_rows[data.indices]]
        if not np.array_equal(current, a["states"][data.indices, 4:7]):
            raise ValueError("Recovered current paddle values disagree with state cache")

    def attach(self, indices, batch):
        source = self.source_rows[np.asarray(indices)]
        rows = source[:, None] - np.arange(self.length - 1, -1, -1)
        lower = self.starts[source, None]
        allowed = rows >= lower
        safe = np.maximum(rows, lower)
        states = self.states[safe].copy()
        valid = allowed & self.valid[safe]
        states[~valid] = 0
        actions = self.actions[safe].copy()
        actions[~allowed] = 3
        context = np.concatenate(
            (states, valid[..., None].astype(np.float32), np.eye(4, dtype=np.float32)[actions]),
            axis=-1,
        )
        batch["paddle_context"] = torch.from_numpy(context)
        return batch
