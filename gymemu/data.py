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


def read_episodes(root: Path, split: str, limit: int | None = None) -> list[Episode]:
    metadata = table(root, "episodes", split, ["episode_id", "initial_frame_id", "length"])
    meta = sorted(metadata.to_pylist(), key=lambda e: e["episode_id"])
    if len({e["episode_id"] for e in meta}) != len(meta):
        raise ValueError("Duplicate episode IDs")
    data = table(
        root,
        "transitions",
        split,
        ["episode_id", "step", "source_frame_id", "successor_frame_id", "native_action_json"],
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
        episodes.append(Episode(eid, frame_ids, actions[lo:hi]))
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
    ):
        if type(action_history) is not int or not 1 <= action_history <= history:
            raise ValueError("action_history must be between 1 and the RGB history length")
        self.action_history = action_history
        self.frames, self.episodes, self.history = frames, episodes, history
        self.action_index = {value: i for i, value in enumerate(actions)}
        self.start_action = len(actions)  # Explicit absence of a game action at reset.
        self.ends = np.cumsum([len(e.frames) for e in episodes])
        for episode in episodes:
            frames.check_ids(episode.frames)
            if set(episode.actions.tolist()) - self.action_index.keys():
                raise ValueError("An episode uses an action absent from the training vocabulary")

    def __len__(self):
        return int(self.ends[-1])

    def action_at(self, episode, position):
        if self.action_history == 1:
            return (
                self.start_action
                if position == 0
                else self.action_index[int(episode.actions[position - 1])]
            )
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
        past = episode.frames[max(0, position - self.history) : position]
        history = frame_stack(
            [self.frames.get(int(i)) for i in past],
            self.history,
            self.frames.shape,
            dtype=torch.uint8 if self.frames.compact else torch.float32,
        )
        action = self.action_at(episode, position)
        if self.action_history > 1:
            action = torch.tensor(action, dtype=torch.long)
        return history, action, self.frames.get(int(episode.frames[position]))
