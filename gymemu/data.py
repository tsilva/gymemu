"""Episode-safe RGB trajectories shared by all emulator approaches."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, snapshot_download
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_DATASET = "tsilva/gradlab-breakout-trajectories"
DEFAULT_REVISION = "b8091d248295eb5135011dd9b943c75f4a7d50be"
DEFAULT_HISTORY = 8


def resolve_dataset(dataset: str, revision: str | None) -> tuple[Path, dict]:
    path = Path(dataset).expanduser()
    if path.is_dir():
        return path.resolve(), {"dataset": str(path.resolve()), "revision": None}
    revision = revision or (DEFAULT_REVISION if dataset == DEFAULT_DATASET else "main")
    sha = HfApi().dataset_info(dataset, revision=revision).sha
    path = snapshot_download(
        dataset,
        repo_type="dataset",
        revision=sha,
        allow_patterns=["manifest.json", "frames/**", "transitions/**", "episodes/**"],
    )
    return Path(path), {"dataset": dataset, "revision": sha}


def table(root: Path, kind: str, split: str, columns: list[str]) -> pa.Table:
    paths = sorted((root / kind / split).glob("*.parquet"))
    if not paths:
        raise ValueError(f"Missing {kind}/{split}/*.parquet in {root}")
    return pa.concat_tables([pq.read_table(p, columns=columns) for p in paths])


class Frames:
    """Encoded images in Arrow, a compact ID lookup, and a bounded decoded-image cache."""

    def __init__(self, root: Path, compact: bool = False, cache: Path | None = None):
        self.compact = compact
        data = table(root, "frames", "assets", ["frame_id"] if cache else ["frame_id", "image"])
        ids = data["frame_id"].to_numpy()
        self.order = np.argsort(ids)
        self.ids = ids[self.order]
        if not len(ids) or np.any(np.diff(self.ids) <= 0):
            raise ValueError("Frame IDs must be nonempty and unique")
        self.cached = cache is not None
        if cache:
            from gymemu.cache import open_cache

            self.images, self.shape = open_cache(root, cache, ids)
        else:
            self.images = data["image"]
        first = self.get(int(self.ids[0]))
        self.shape = tuple(first.shape)  # C, H, W; no cropping, resizing or binarization.

    @lru_cache(maxsize=256)
    def get(self, frame_id: int) -> torch.Tensor:
        position = int(np.searchsorted(self.ids, frame_id))
        if position == len(self.ids) or self.ids[position] != frame_id:
            raise ValueError(f"Unknown frame ID {frame_id}")
        value = self.images[int(self.order[position])]
        if self.cached:
            raw = pa.decompress(value.as_buffer(), int(np.prod(self.shape)), codec="lz4")
            result = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).reshape(self.shape))
        else:
            value = value.as_py()
            if not value["bytes"]:
                raise ValueError("Expected an embedded image, not an external image path")
            with Image.open(io.BytesIO(value["bytes"])) as image:
                if image.mode != "RGB":
                    raise ValueError("Expected RGB images")
                pixels = np.asarray(image).copy()
            result = torch.from_numpy(pixels).permute(2, 0, 1)
        if not self.compact:
            result = result.float().div_(255)
        if hasattr(self, "shape") and tuple(result.shape) != self.shape:
            raise ValueError("All images must have the same dimensions")
        return result

    def check_ids(self, ids: np.ndarray) -> None:
        positions = np.searchsorted(self.ids, ids)
        if np.any(positions == len(self.ids)) or np.any(self.ids[positions] != ids):
            raise ValueError("Trajectory references an unknown frame ID")


@dataclass
class Episode:
    episode_id: int
    frames: np.ndarray  # Initial frame followed by each transition's successor.
    actions: np.ndarray  # Executed action leading from frames[t] to frames[t + 1].
    states: np.ndarray | None = None  # Frame-aligned normalized values plus availability.


def recorded_state(record, fields):
    """Read scalar labels from Gradlab's JSON tree without executing serialized objects."""
    try:
        tree = json.loads(json.loads(record)["structure"])
        if tree[0] != "dict":
            raise ValueError("Expected record dictionary")
        labels = dict(tree[1])["labels"]
        if labels[0] != "dict":
            raise ValueError("Expected label dictionary")
        labels = dict(labels[1])
        values = []
        for field in fields:
            tag, value = labels[field]
            if tag != "scalar" or type(value) not in (int, float):
                raise ValueError(f"Expected numeric scalar {field}")
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Expected normalized {field} in [0, 1]")
            values.append(value)
        return [*values, 1.0]
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ValueError(f"Missing or invalid normalized state labels {fields}: {error}") from error


def read_episodes(
    root: Path, split: str, limit: int | None = None, *, state_fields=()
) -> list[Episode]:
    metadata = table(root, "episodes", split, ["episode_id", "initial_frame_id", "length"])
    meta = sorted(metadata.to_pylist(), key=lambda e: e["episode_id"])
    if len({e["episode_id"] for e in meta}) != len(meta):
        raise ValueError("Duplicate episode IDs")
    data = table(
        root,
        "transitions",
        split,
        ["episode_id", "step", "source_frame_id", "successor_frame_id", "native_action_json"]
        + (["record_json"] if state_fields else []),
    )
    episode_ids = data["episode_id"].to_numpy()
    steps = data["step"].to_numpy()
    order = np.lexsort((steps, episode_ids))
    episode_ids, steps = episode_ids[order], steps[order]
    sources = data["source_frame_id"].to_numpy()[order]
    successors = data["successor_frame_id"].to_numpy()[order]
    raw_actions = data["native_action_json"].to_pylist()
    action_values = []
    for value in raw_actions:
        value = json.loads(value)
        if type(value) is not int:
            raise ValueError("This baseline supports scalar integer executed actions only")
        action_values.append(value)
    actions = np.asarray(action_values, dtype=np.int64)[order]
    states = None
    if state_fields:
        states = np.asarray(
            [recorded_state(value.as_py(), state_fields) for value in data["record_json"]],
            dtype=np.float32,
        ).reshape(-1, len(state_fields) + 1)[order]
    if set(episode_ids.tolist()) - {e["episode_id"] for e in meta}:
        raise ValueError("Transitions without episode metadata")
    episodes = []
    for item in meta[:limit]:
        eid = item["episode_id"]
        lo = int(np.searchsorted(episode_ids, eid, side="left"))
        hi = int(np.searchsorted(episode_ids, eid, side="right"))
        length = int(item["length"])
        if hi - lo != length or not np.array_equal(steps[lo:hi], np.arange(length)):
            raise ValueError(f"Episode {eid} has missing, duplicated or inconsistent steps")
        if item["initial_frame_id"] is None:
            if length:
                raise ValueError(f"Episode {eid} has no initial frame")
            continue  # An allocated but never started episode has no training target.
        frame_ids = np.concatenate([[item["initial_frame_id"]], successors[lo:hi]])
        if not np.array_equal(sources[lo:hi], frame_ids[:-1]):
            raise ValueError(f"Episode {eid} has a broken frame chain")
        # record_json labels describe the successor, never the source. Episode metadata
        # has no reset labels, so the initial frame carries zeros with availability=0.
        aligned_states = None
        if states is not None:
            aligned_states = np.concatenate(
                [np.zeros((1, len(state_fields) + 1), dtype=np.float32), states[lo:hi]]
            )
        episodes.append(Episode(eid, frame_ids, actions[lo:hi], aligned_states))
    if not episodes:
        raise ValueError(f"No usable episodes in {split}")
    return episodes


def frame_stack(
    history: list[torch.Tensor], length: int, shape: tuple, dtype=torch.float32
) -> torch.Tensor:
    """Oldest to newest, missing frames padded with zeros on the left."""
    recent = history[-length:]
    if len(recent) == length:
        return torch.stack(recent).to(dtype=dtype)
    result = torch.zeros((length, *shape), dtype=dtype)
    if recent:
        result[-len(recent) :] = torch.stack(recent)
    return result


def pad_actions(indices, length, start_action):
    """Oldest-to-newest action tokens, including current action in the last slot."""
    recent = list(indices)[-length:]
    return [start_action] * (length - len(recent)) + recent


class Windows(Dataset):
    def __init__(
        self,
        frames: Frames,
        episodes: list[Episode],
        history: int,
        actions: list[int],
        *,
        action_history: int = 1,
        rollout_steps: int = 0,
        state_fields=(),
    ):
        if type(action_history) is not int or not 1 <= action_history <= history:
            raise ValueError("action_history must be between 1 and the RGB history length")
        self.action_history = action_history
        self.state_fields = tuple(state_fields)
        if type(rollout_steps) is not int or rollout_steps < 0:
            raise ValueError("rollout_steps must be a nonnegative integer")
        self.rollout_steps = rollout_steps
        self.input_history = history + rollout_steps
        self.action_shape = (
            (rollout_steps + 1, action_history)
            if rollout_steps
            else (() if action_history == 1 else (action_history,))
        )
        self.frames, self.episodes, self.history = frames, episodes, history
        self.action_index = {value: i for i, value in enumerate(actions)}
        self.start_action = len(actions)  # Explicit absence of a game action at reset.
        self.ends = np.cumsum([len(e.frames) for e in episodes])
        for episode in episodes:
            if self.state_fields:
                validate_states(episode.states, len(episode.frames), self.state_fields)
            frames.check_ids(episode.frames)
            if set(episode.actions.tolist()) - self.action_index.keys():
                raise ValueError("An episode uses an action absent from the training vocabulary")

    def __len__(self):
        return int(self.ends[-1])

    def action_at(self, episode, position):
        if self.rollout_steps:
            return [
                self._action_tokens(episode, p) if p >= 0 else [-1] * self.action_history
                for p in range(position - self.rollout_steps, position + 1)
            ]
        if self.action_history == 1:
            return (
                self.start_action
                if position == 0
                else self.action_index[int(episode.actions[position - 1])]
            )
        return self._action_tokens(episode, position)

    def _action_tokens(self, episode, position):
        past = episode.actions[max(0, position - self.action_history) : position]
        return pad_actions(
            [self.action_index[int(a)] for a in past], self.action_history, self.start_action
        )

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        number = int(np.searchsorted(self.ends, index, side="right"))
        position = index - (int(self.ends[number - 1]) if number else 0)
        episode = self.episodes[number]
        past = episode.frames[max(0, position - self.input_history) : position]
        history = frame_stack(
            [self.frames.get(int(i)) for i in past],
            self.input_history,
            self.frames.shape,
            dtype=torch.uint8 if self.frames.compact else torch.float32,
        )
        action = self.action_at(episode, position)
        if self.action_shape:
            action = torch.tensor(action, dtype=torch.long)
        return (
            history,
            action,
            self.frames.get(int(episode.frames[position])),
            *self.state_at(episode, position),
        )

    def state_at(self, episode, position):
        if not self.state_fields:
            return ()
        past = episode.states[max(0, position - self.input_history) : position]
        history = frame_stack(
            list(torch.from_numpy(past)), self.input_history, (len(self.state_fields) + 1,)
        )
        return history, torch.from_numpy(episode.states[position])


def validate_states(states, length, fields):
    """One availability flag follows the normalized coordinates in each frame row."""
    if states is None or tuple(states.shape) != (length, len(fields) + 1):
        raise ValueError("State history must align with every frame and declared state field")
    values = np.asarray(states)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("State values must be finite and normalized to [0, 1]")
    if np.any((values[:, -1] != 0) & (values[:, -1] != 1)):
        raise ValueError("State availability must be 0 or 1")
    if np.any(values[values[:, -1] == 0, :-1] != 0):
        raise ValueError("Unavailable state coordinates must be zero")
