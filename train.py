"""Plain CNN next-frame training. This module also defines the player's exact model contract."""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from itertools import islice
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, snapshot_download
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

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

    def __init__(self, root: Path, compact: bool = False):
        self.compact = compact
        data = table(root, "frames", "assets", ["frame_id", "image"])
        ids = data["frame_id"].to_numpy()
        self.order = np.argsort(ids)
        self.ids = ids[self.order]
        if not len(ids) or np.any(np.diff(self.ids) <= 0):
            raise ValueError("Frame IDs must be nonempty and unique")
        self.images = data["image"]
        first = self.get(int(self.ids[0]))
        self.shape = tuple(first.shape)  # C, H, W; no cropping, resizing or binarization.

    @lru_cache(maxsize=256)
    def get(self, frame_id: int) -> torch.Tensor:
        position = int(np.searchsorted(self.ids, frame_id))
        if position == len(self.ids) or self.ids[position] != frame_id:
            raise ValueError(f"Unknown frame ID {frame_id}")
        value = self.images[int(self.order[position])].as_py()
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
    result = torch.zeros((length, *shape), dtype=dtype)
    recent = history[-length:]
    if recent:
        result[-len(recent) :] = torch.stack(recent)
    return result


class Windows(Dataset):
    def __init__(self, frames: Frames, episodes: list[Episode], history: int, actions: list[int]):
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
        action = (
            self.start_action
            if position == 0
            else self.action_index[int(episode.actions[position - 1])]
        )
        return history, action, self.frames.get(int(episode.frames[position]))


class Autoencoder(nn.Module):
    """Three strided convolutions and three transposed convolutions; direct RGB output."""

    def __init__(self, history: int, actions: int, shape: tuple, width: int = 32):
        super().__init__()
        self.history, self.actions, self.shape, self.width = history, actions, tuple(shape), width
        channels = history * shape[0] + actions + 1
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, width, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(width, width * 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(width * 2, width * 4, 4, 2, 1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(width * 4, width * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(width * 2, width, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(width, shape[0], 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, history, action):
        batch, _, _, height, width = history.shape
        action = F.one_hot(action, self.actions + 1).to(history.dtype)
        planes = action[:, :, None, None].expand(-1, -1, height, width)
        x = torch.cat([history.flatten(1, 2), planes], dim=1)
        # Pad only to make stride-8 geometry exact; restore the original full canvas.
        x = F.pad(x, (0, (-width) % 8, 0, (-height) % 8))
        return self.decoder(self.encoder(x))[:, :, :height, :width]


def device_for(name: str) -> torch.device:
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    return torch.device(name)


def load_model(checkpoint: Path, device: torch.device) -> tuple[Autoencoder, dict]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    if config.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format; old gymemu models are incompatible")
    model = Autoencoder(
        config["history"], len(config["action_values"]), config["shape"], config["width"]
    )
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval(), config


def save_model(path: Path, model: Autoencoder, config: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "config": config,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        temporary,
    )
    temporary.replace(path)


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    max_batches=None,
    *,
    precision="fp32",
    label="",
    checkpoint=None,
    checkpoint_seconds=60,
):
    model.train(optimizer is not None)
    total = torch.zeros(
        (), device=device, dtype=torch.float64 if device.type != "mps" else torch.float32
    )
    samples, batches = 0, 0
    started = last_checkpoint = time.monotonic()
    count = min(len(loader), max_batches) if max_batches else len(loader)
    iterator = islice(loader, count)
    for history, action, target in iterator:
        history = history.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).float()
        # Workers transfer exact uint8 pixels; normalization happens on the device.
        if loader.dataset.frames.compact:
            history.div_(255)
            target.div_(255)
        action = action.to(device, non_blocking=True)
        with torch.set_grad_enabled(optimizer is not None):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                prediction = model(history, action)
                loss = F.mse_loss(prediction, target)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += loss.detach().to(total.dtype) * len(target)
        samples += len(target)
        batches += 1
        elapsed = time.monotonic() - started
        if batches % 100 == 0 or batches == count:
            mse = total.item() / samples
            if not math.isfinite(mse):
                raise ValueError("Non-finite pixel MSE")
            print(
                f"{label} {'train' if optimizer else 'eval'} batches={batches}/{count} "
                f"mse={mse:.6f} samples/s={samples / elapsed:.1f} "
                f"eta_seconds={elapsed / batches * (count - batches):.0f}",
                flush=True,
            )
        if checkpoint and time.monotonic() - last_checkpoint >= checkpoint_seconds:
            # Synchronizing here also catches an invalid loss before publishing weights.
            if not math.isfinite(total.item()):
                raise ValueError("Non-finite pixel MSE")
            checkpoint(batches, samples)
            last_checkpoint = time.monotonic()
    if not samples:
        raise ValueError("An epoch must contain at least one sample")
    return {
        "mse": total.item() / samples,
        "samples": samples,
        "batches": batches,
        "seconds": time.monotonic() - started,
    }


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET, help="Hub ID or local snapshot folder"
    )
    parser.add_argument("--revision", help="Hub revision; the default dataset uses a pinned commit")
    parser.add_argument("--output", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--history", type=positive, default=DEFAULT_HISTORY)
    parser.add_argument("--width", type=positive, default=32)
    parser.add_argument("--epochs", type=positive, default=10)
    parser.add_argument("--batch-size", type=positive, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--threads", type=positive, help="Optional PyTorch CPU thread limit")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--workers", type=int, default=0, help="Parallel image-loader processes")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--checkpoint-seconds",
        type=positive,
        default=60,
        help="Interval for atomic latest.pt inference snapshots",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="heldout")
    parser.add_argument(
        "--limit-episodes", type=positive, help="First N episodes per split for smokes"
    )
    parser.add_argument("--train-batches", type=positive, help="Optional per-epoch smoke limit")
    parser.add_argument("--eval-batches", type=positive, help="Optional evaluation smoke limit")
    args = parser.parse_args(argv)
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if args.train_split == args.eval_split:
        parser.error("Training and evaluation splits must differ")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; choose a new run directory")
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = device_for(args.device)
    if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("--precision bf16 requires a CUDA GPU with bfloat16 support")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print("Loading the Parquet snapshot directly from the Hub or disk (no Viewer API).", flush=True)
    root, provenance = resolve_dataset(args.dataset, args.revision)
    frames = Frames(root, compact=True)
    train_episodes = read_episodes(root, args.train_split, args.limit_episodes)
    eval_episodes = read_episodes(root, args.eval_split, args.limit_episodes)
    if {e.episode_id for e in train_episodes} & {e.episode_id for e in eval_episodes}:
        raise ValueError("Training and evaluation episode IDs overlap")
    actions = sorted({int(a) for e in train_episodes for a in e.actions})
    if not actions:
        raise ValueError("Training requires at least one executed action")
    training = Windows(frames, train_episodes, args.history, actions)
    evaluation = Windows(frames, eval_episodes, args.history, actions)
    config = {
        "format_version": 1,
        "history": args.history,
        "shape": list(frames.shape),
        "action_values": actions,
        "width": args.width,
        "dataset": provenance,
        "padding": "left_zero",
        "bootstrap": "empty_history_and_start_action",
        "objective": "next_frame_rgb_mse",
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "seed": args.seed,
        "limit_episodes": args.limit_episodes,
        "train_batches": args.train_batches,
        "eval_batches": args.eval_batches,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "workers": args.workers,
        "precision": args.precision,
        "checkpoint_seconds": args.checkpoint_seconds,
        "device": str(device),
    }
    if (root / "manifest.json").exists():
        config["dataset_manifest"] = json.loads((root / "manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model = Autoencoder(args.history, len(actions), frames.shape, args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    if args.workers:
        # Spawn avoids forking initialized CUDA/thread pools. Workers persist across epochs.
        loader_options.update(
            multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2
        )
    train_loader = DataLoader(training, shuffle=True, **loader_options)
    eval_loader = DataLoader(evaluation, shuffle=False, **loader_options)
    print(
        json.dumps(
            {
                "device": str(device),
                "history": args.history,
                "parameters": sum(p.numel() for p in model.parameters()),
                "training_examples": len(training),
                "evaluation_examples": len(evaluation),
                "actions": actions,
            }
        ),
        flush=True,
    )
    best = float("inf")
    for epoch in range(1, args.epochs + 1):

        def snapshot(batches, samples):
            save_model(
                args.output / "latest.pt",
                model,
                {
                    **config,
                    "epoch": epoch,
                    "epoch_complete": False,
                    "train_batches_completed": batches,
                    "train_samples_completed": samples,
                },
            )

        train = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            args.train_batches,
            precision=args.precision,
            label=f"epoch={epoch}/{args.epochs}",
            checkpoint=snapshot,
            checkpoint_seconds=args.checkpoint_seconds,
        )
        snapshot(train["batches"], train["samples"])
        validation = run_epoch(
            model,
            eval_loader,
            device,
            max_batches=args.eval_batches,
            precision=args.precision,
            label=f"epoch={epoch}/{args.epochs}",
        )
        record = {"epoch": epoch, "train": train, "validation": validation}
        print(json.dumps(record), flush=True)
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        save_model(
            args.output / "last.pt", model, {**config, "epoch": epoch, "epoch_complete": True}
        )
        if validation["mse"] < best:
            best = validation["mse"]
            save_model(
                args.output / "best.pt", model, {**config, "epoch": epoch, "epoch_complete": True}
            )
    print(f"Checkpoint: {args.output.resolve() / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
