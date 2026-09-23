"""Teacher forcing for the published frameskip-1 state contract and recorded RGB."""

import io
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from gymemu.config import CONFIG_DIR
from gymemu.replay import ReplayPlayer
from gymemu.state_data import FIELDS, decode_labels
from gymemu.state_hidden_context import PaddleController


@dataclass
class StateEpisode:
    episode_id: int
    sources: torch.Tensor
    targets: torch.Tensor
    terminal: np.ndarray
    source_frames: np.ndarray
    target_frames: np.ndarray
    recorded_steps: np.ndarray

    @property
    def actions(self):
        return self.sources[:, 118].numpy().astype(np.int64)


def reconstruct_episode(metadata, rows, thresholds):
    """Reconstruct source-time memory from prior observations, as in training.

    Controller replay reads recorded actions only. Fraction, contact and hit count
    at time t use observations up to t, never the successor being predicted.
    """
    rows = sorted(rows, key=lambda row: row["step"])
    n = metadata["length"]
    if len(rows) != n or [r["step"] for r in rows] != list(range(n)):
        raise ValueError("Episode has missing or duplicate transition steps")
    frames = np.array([metadata["initial_frame_id"], *[r["successor_frame_id"] for r in rows]])
    if [r["source_frame_id"] for r in rows] != frames[:-1].tolist():
        raise ValueError("Broken source/successor frame chain")
    states = np.zeros((n + 1, 115), np.float32)
    lives = np.full(n + 1, -1)
    usable = np.zeros(n + 1, bool)
    actions = []
    controller = PaddleController(thresholds)
    for _ in range(int(np.random.default_rng(metadata["seed"]).integers(1, 31, dtype=np.uint64))):
        controller.step(0)
    charge = np.zeros(n + 1, np.float32)
    charge[0] = controller.charge
    scale = np.array([160, 255, 2, 3.375, 160, 160, 16], np.float32)
    for t, row in enumerate(rows):
        if row["episode_id"] != metadata["episode_id"] or row["configured_frame_skip"] != 1:
            raise ValueError("Expected one frameskip-1 episode")
        if t < n - 1 and (row["terminated"] or row["truncated"]):
            raise ValueError("Episode boundary before final transition")
        action = json.loads(row["native_action_json"])
        if type(action) is not int or action not in (0, 1, 2):
            raise ValueError("Expected scalar provider action 0/1/2")
        if action != json.loads(row["effective_action_json"]):
            raise ValueError("Native and effective actions disagree")
        actions.append(action)
        labels = decode_labels(row["record_json"])
        numeric = np.array([labels[field] for field in FIELDS], np.float32) * scale
        bricks = np.asarray(row["brick_grid"], np.float32)
        if not np.isfinite(numeric).all() or bricks.shape != (6, 18):
            raise ValueError("Invalid recorded state")
        if not np.isin(bricks, [0, 1]).all():
            raise ValueError("Invalid brick occupancy")
        states[t + 1, :7] = np.rint(numeric * 8) / 8
        states[t + 1, 7:] = bricks.ravel()
        lives[t + 1] = labels["lives"]
        usable[t + 1] = (
            numeric[1] > 0
            and row["brick_grid_suspect"] is False
            and row["is_initial_brick_layout"] is False
        )
        controller.step(action + 1)
        if (controller.x, controller.vx) != tuple(states[t + 1, 4:6]):
            raise ValueError(f"Recorded controller reconstruction differs at step {t}")
        charge[t + 1] = controller.charge
    moved = (states[:-1, 1] > 0) & (states[:-1, 1] < 208) & (states[1:, 1] > 0)
    fraction = np.r_[0, np.cumsum(np.where(moved, np.rint(states[1:, 3] * 8), 0)) % 8]
    contact = np.zeros(n + 1, bool)
    counts = np.zeros(n + 1, np.int64)
    removed = ((states[:-1, 7:] > 0) & (states[1:, 7:] == 0)).any(1)
    hits = (states[:-1, 1] >= 160) & (states[:-1, 3] > 0) & (states[1:, 3] < 0)
    for t in range(n):
        if states[t, 1] == 0 or states[t + 1, 1] == 0:
            continue
        counts[t + 1] = min(12, counts[t] + int(hits[t]))
        native_y = states[t, 1] + 9
        contact[t + 1] = (contact[t] and not (native_y > 93 or native_y + 3 < 56)) or removed[t]
        if native_y < 49:
            contact[t + 1] = False
    terminal = usable[:-1] & ((lives[1:] < lives[:-1]) | (states[1:, 1] == 0))
    indices = np.flatnonzero(usable[:-1] & (terminal | usable[1:]))
    if not len(indices):
        raise ValueError("Episode has no eligible state transitions")
    current, following = states[indices], states[indices + 1]
    sources = np.c_[
        current[:, :4],
        current[:, 4],
        current[:, 6],
        charge[indices],
        counts[indices],
        fraction[indices],
        contact[indices],
        current[:, 7:],
        np.asarray(actions)[indices],
    ].astype(np.float32)
    targets = np.c_[
        following[:, 0],
        following[:, 1] + fraction[indices + 1] / 8,
        following[:, 2:4],
        following[:, 7:],
        contact[indices + 1],
        counts[indices + 1],
        following[:, 6],
        charge[indices + 1],
        following[:, 4],
    ].astype(np.float32)
    targets[terminal[indices]] = 0
    return StateEpisode(
        metadata["episode_id"],
        torch.from_numpy(sources),
        torch.from_numpy(targets),
        terminal[indices],
        frames[indices],
        frames[indices + 1],
        indices,
    )


def _read_state_episode(config, *, dataset=None, split="validation", episode_id=None):
    """Read and reconstruct one episode without fetching RGB assets."""
    provenance = config["replay"]
    repository = dataset or provenance["repository"]
    local = Path(repository).expanduser()

    def file(name, revision=provenance["revision"]):
        if local.is_dir():
            path = local / name
            if not path.is_file():
                raise ValueError(f"Missing replay artifact: {path}")
            return path
        return Path(hf_hub_download(repository, name, repo_type="dataset", revision=revision))

    split_root = f"splits/{provenance['split_id']}"
    manifest = json.loads(file(f"{split_root}/manifest.json").read_text())
    if manifest["split_id"] != provenance["split_id"]:
        raise ValueError("Replay split identity differs from the pinned configuration")
    ids = manifest["episodes"][split]
    episode_id = ids[0] if episode_id is None else episode_id
    if episode_id not in ids:
        raise ValueError(f"Episode {episode_id} is absent from {split}")
    print(f"Loading {split} episode {episode_id} for state playback...", flush=True)
    meta = []
    rows = []
    columns = [
        "episode_id",
        "step",
        "source_frame_id",
        "successor_frame_id",
        "record_json",
        "brick_grid",
        "brick_grid_suspect",
        "is_initial_brick_layout",
        "terminated",
        "truncated",
        "configured_frame_skip",
        "native_action_json",
        "effective_action_json",
    ]
    for name in sorted(manifest["tables"]):
        if name.startswith(f"{split_root}/episodes/{split}/"):
            meta.extend(
                pq.read_table(file(name), filters=[("episode_id", "=", episode_id)]).to_pylist()
            )
        elif name.startswith(f"{split_root}/transitions/{split}/"):
            rows.extend(
                pq.read_table(
                    file(name), columns=columns, filters=[("episode_id", "=", episode_id)]
                ).to_pylist()
            )
    if len(meta) != 1:
        raise ValueError("Expected one episode metadata row")
    thresholds = json.loads((CONFIG_DIR / "state_replay_controller.json").read_text())["thresholds"]
    episode = reconstruct_episode(meta[0], rows, thresholds)
    return episode, manifest, file


def load_state_episode(config, *, dataset=None, split="validation", episode_id=None):
    """Load a complete recorded state sequence for initializing generated play."""
    episode, _, _ = _read_state_episode(config, dataset=dataset, split=split, episode_id=episode_id)
    return episode


def load_state_replay(
    model, decoder, config, device, *, dataset=None, split="validation", episode_id=None
):
    """Read one selected episode and its referenced images; never fit any model."""
    episode, manifest, file = _read_state_episode(
        config, dataset=dataset, split=split, episode_id=episode_id
    )
    publication_path = manifest["source_publication"]
    publication = json.loads(file(publication_path, manifest["source_revision"]).read_text())
    prefix = str(Path(publication_path).parent)
    needed = set(map(int, np.r_[episode.source_frames, episode.target_frames]))
    images = {}
    for name in sorted(publication["tables"]):
        if not name.startswith("frames/assets/"):
            continue
        path = file(f"{prefix}/{name}", manifest["source_revision"])
        table = pq.read_table(
            path, columns=["frame_id", "image"], filters=[("frame_id", "in", sorted(needed))]
        )
        for row in table.to_pylist():
            frame_id = row["frame_id"]
            if frame_id in images:
                raise ValueError("Duplicate frame ID")
            # Keep compressed bytes; decode only displayed frames.
            images[frame_id] = row["image"]["bytes"]
        needed.difference_update(images)
        if not needed:
            break
    if needed:
        raise ValueError("Replay references missing frame IDs")
    frames = RecordedImages(images)
    return StateReplay(model, decoder, config, device, episode, frames)


class RecordedImages:
    def __init__(self, images):
        self.images = images

    def get(self, frame_id):
        with Image.open(io.BytesIO(self.images[int(frame_id)])) as image:
            if image.mode != "RGB" or image.size != (160, 210):
                raise ValueError("Expected recorded 160x210 RGB frames")
            pixels = np.asarray(image).copy()
        return torch.from_numpy(pixels).permute(2, 0, 1).float() / 255


class StateReplay(ReplayPlayer):
    """Every prediction consumes recorded state/action, including after a false stop."""

    history_editable = False

    def __init__(self, model, decoder, config, device, episode, frames):
        self.model, self.decoder = model.eval(), decoder.eval()
        self.config = {**config, "history": 1, "shape": [3, 210, 160]}
        self.device = device
        self.actions = config["action_values"]
        self.playback_fps = config["playback_fps"]
        self.windows = SimpleNamespace(episodes=[episode], frames=frames)
        self.episode_index = 0
        self.reset()

    @property
    def context_note(self):
        text = "Recorded state and action on every step. Predictions never feed back."
        if not self.has_prediction:
            return text + " Inactive or unreliable state rows are skipped."
        result = (
            "not scored on terminal targets"
            if self.state_exact is None
            else ("exact" if self.state_exact else "differs")
        )
        text += (
            f" Recorded step {self.episode.recorded_steps[self.steps - 1]}. "
            f"Next state: {result}. Life loss: predicted {self.predicted_terminal}, "
            f"recorded {self.target_terminal}."
        )
        if self.predicted_terminal or self.target_terminal:
            text += " Terminal RGB is unsupported; RGB MSE is not scored."
        return text

    def reset(self, *, cycle=False):
        self.continuous = False
        self.held_keys = []
        self.steps = 0
        self.frame = None
        self.mse = None
        self.recorded_action = None
        self.has_prediction = False
        self.state_exact = None
        self.predicted_terminal = None
        self.target_terminal = None
        self.target = self.windows.frames.get(self.episode.source_frames[0])
        self.input_stack = self.target[None].clone()

    @torch.inference_mode()
    def advance(self, action=None):
        if self.finished:
            self.continuous = False
            return
        index = self.steps
        source = self.episode.sources[index : index + 1].to(self.device)
        prediction = self.model.predict(source)
        if prediction.shape != (1, 118) or not torch.isfinite(prediction).all():
            raise ValueError("Dynamics returned an invalid state")
        self.predicted_terminal = bool(prediction[0, 117])
        self.target_terminal = bool(self.episode.terminal[index])
        self.state_exact = (
            None
            if self.target_terminal
            else bool(
                not self.predicted_terminal
                and torch.equal(prediction[0, :117].cpu(), self.episode.targets[index])
            )
        )
        self.target = self.windows.frames.get(self.episode.target_frames[index])
        self.input_stack = self.windows.frames.get(self.episode.source_frames[index])[None]
        self.frame = (
            torch.zeros_like(self.target)
            if self.predicted_terminal
            else self.decoder.render(self.decoder.from_dynamics(prediction))[0].float().cpu()
        )
        self.mse = (
            None
            if self.target_terminal or self.predicted_terminal
            else (self.frame - self.target.float()).square().mean().item()
        )
        self.recorded_action = int(source[0, 118])
        self.has_prediction = True
        self.steps += 1
        if self.finished:
            self.continuous = False
