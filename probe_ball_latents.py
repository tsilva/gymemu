"""Fit frozen-encoder linear probes of current and next recorded ball position.

Streams the complete dataset without caching the large spatial feature matrix.
Only recorded training episodes fit normalization, means, or probe parameters.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from gymemu.batches import CachedBatchLoader
from gymemu.checkpoints import load_model
from gymemu.data import Frames, Windows, read_episodes, recorded_state

FIELDS = ("ball_x_normalized", "ball_y_normalized")
SCALE = np.array([160.0, 255.0])  # Native label coordinates, not image-row coordinates.


def attach_labels(root, split, episodes, cache):
    """Stream JSON labels, attaching successor labels to frame step+1."""
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{split}-labels.npz"
    by_id = {e.episode_id: e for e in episodes}
    if path.exists():
        with np.load(path) as saved:
            for episode in episodes:
                episode.states = saved[str(episode.episode_id)]
                assert len(episode.states) == len(episode.frames)
        return
    seen = {}
    for episode in episodes:
        episode.states = np.zeros((len(episode.frames), 3), dtype=np.float32)
        seen[episode.episode_id] = np.zeros(len(episode.actions), dtype=bool)
    for path in sorted((root / "transitions" / split).glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=8192, columns=["episode_id", "step", "successor_frame_id", "record_json"]
        ):
            columns = batch.to_pydict()
            for eid, step, frame, record in zip(*columns.values(), strict=True):
                if eid not in by_id:
                    continue
                episode = by_id[eid]
                assert episode.frames[step + 1] == frame
                assert not seen[eid][step]
                episode.states[step + 1] = recorded_state(record, FIELDS)
                seen[eid][step] = True
    assert all(v.all() for v in seen.values())
    np.savez(cache / f"{split}-labels.npz", **{str(e.episode_id): e.states for e in episodes})


def targets(state_history, state_target):
    """Current = last input frame; next = model target frame. No bootstrap labels."""
    pairs = torch.stack((state_history[:, -1], state_target), 1)
    present = pairs[..., 2].bool() & (pairs[..., 1] > 0)
    return pairs[..., :2], present


class Encoder(nn.Module):
    """Exactly the float32 inference encoder, including action planes and padding."""

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.requires_grad_(False)
        self.eval()

    def forward(self, history, action):
        p = self.predictor
        batch, _, _, height, width = history.shape
        encoded = F.one_hot(action.reshape(batch, p.action_history), p.actions + 1)
        planes = encoded.flatten(1).to(history.dtype)[:, :, None, None]
        planes = planes.expand(-1, -1, height, width)
        inputs = torch.cat((history.flatten(1, 2), planes), 1)
        inputs = F.pad(inputs, (0, (-width) % 8, 0, (-height) % 8))
        return p.encoder(inputs).flatten(1)


class Scores:
    def __init__(self):
        self.n = 0
        self.error = np.zeros(2)
        self.square = np.zeros(2)
        self.y = np.zeros(2)
        self.yy = np.zeros(2)
        self.distances = []

    def add(self, prediction, target):
        prediction, target = np.asarray(prediction), np.asarray(target)
        if not len(target):
            return
        error = (prediction - target) * SCALE
        self.n += len(target)
        self.error += np.abs(error).sum(0)
        self.square += np.square(error).sum(0)
        self.y += (target * SCALE).sum(0)
        self.yy += np.square(target * SCALE).sum(0)
        self.distances.append(np.linalg.norm(error, axis=1).astype(np.float32))

    def result(self):
        if not self.n:
            return {"count": 0}
        distances = np.concatenate(self.distances)
        return {
            "count": self.n,
            "mae_xy_native": (self.error / self.n).tolist(),
            "rmse_xy_native": np.sqrt(self.square / self.n).tolist(),
            "r2_xy": (1 - self.square / np.maximum(self.yy - self.y**2 / self.n, 1e-12)).tolist(),
            "mean_distance_native": float(distances.mean()),
            "median_distance_native": float(np.median(distances)),
            "p90_distance_native": float(np.quantile(distances, 0.9)),
            "within_2_native": float((distances <= 2).mean()),
            "within_5_native": float((distances <= 5).mean()),
        }


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.00003)
    parser.add_argument("--calibration-samples", type=int, default=8192)
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--labels-only", action="store_true")
    parser.add_argument("--resume", type=Path, help="Continue from a completed probe pass")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(47)
    np.random.seed(47)
    device = torch.device(args.device)
    model, config = load_model(args.checkpoint, device)
    model.requires_grad_(False)
    predictor = model.predictor
    encoder = Encoder(predictor)
    assert config["shape"] == [3, 210, 160], "Native coordinate scales are Breakout-specific"
    splits = {}
    for split in ["train", "heldout"]:
        print(f"Reading {split} episodes and recorded labels", flush=True)
        episodes = read_episodes(args.dataset, split, args.limit_episodes)
        # Caches are scoped to this experiment and its immutable dataset/checkpoint manifest.
        attach_labels(args.dataset, split, episodes, args.output / "labels")
        splits[split] = episodes
    assert not ({e.episode_id for e in splits["train"]} & {e.episode_id for e in splits["heldout"]})
    manifest = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_config": config,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "feature": "float32 final encoder ReLU, full spatial flatten, action conditioned",
        "alignment": "current = last history frame; next = target frame",
        "labels": "recorded native normalized coordinates; y=0 and missing labels excluded",
        "native_coordinate_scales": SCALE.tolist(),
        "interpretation": (
            "Linear decodability; not proof of causal use or absence of nonlinear information"
        ),
        "episode_ids": {s: [e.episode_id for e in eps] for s, eps in splits.items()},
    }
    write_json(args.output / "manifest.json", manifest)
    if args.labels_only:
        return
    frames = Frames(args.dataset, compact=True, cache=args.frame_cache)
    datasets = {
        s: Windows(
            frames,
            eps,
            config["history"],
            config["action_values"],
            action_history=config["action_history"],
            state_fields=FIELDS,
        )
        for s, eps in splits.items()
    }

    def loader(split, shuffle=False):
        dataset = datasets[split]
        if args.frame_cache:
            return CachedBatchLoader(
                dataset, args.batch_size, args.workers, shuffle, pin_memory=device.type == "cuda"
            )
        return DataLoader(dataset, args.batch_size, shuffle=shuffle, num_workers=0)

    def batches(split, shuffle=False):
        for i, batch in enumerate(loader(split, shuffle)):
            if args.limit_batches and i >= args.limit_batches:
                break
            history, action, _, sh, st = batch
            with torch.no_grad():
                features = encoder(history.to(device).float().div_(255), action.to(device))
            yield i, features, sh.to(device), st.to(device)

    resume = (
        torch.load(args.resume, map_location=device, weights_only=True) if args.resume else None
    )
    if resume:
        mean, std = resume["feature_mean"], resume["feature_std"]
        dimension = resume["dimension"]
        print(
            f"Continuing after pass {resume['epoch']}; optimizer restored if available", flush=True
        )
    else:
        # Fixed train-only affine normalization preserves the linear probe class.
        count = 0
        total = square = None
        for _, z, _, _ in batches("train", True):
            if total is None:
                total, square = torch.zeros_like(z[0]), torch.zeros_like(z[0])
            total += z.sum(0)
            square += z.square().sum(0)
            count += len(z)
            if count >= args.calibration_samples:
                break
        mean = total / count
        std = (square / count - mean.square()).clamp_min(1e-4).sqrt()
        dimension = len(mean)
        print(f"Frozen latent dimension: {dimension}; normalization samples: {count}", flush=True)
    # Each row pair is an independent affine model, trained with its own availability mask.
    head = nn.Linear(dimension, 4, device=device)
    nn.init.zeros_(head.weight)
    means = []
    for offset in [-1, 0]:
        values = []
        for e in splits["train"]:
            s = e.states[:-1] if offset == -1 else e.states
            values.append(s[(s[:, 2] > 0) & (s[:, 1] > 0), :2])
        means.append(np.concatenate(values).mean(0, dtype=np.float64))
    means = np.stack(means)
    with torch.no_grad():
        head.bias.copy_(torch.as_tensor(means.reshape(-1), device=device, dtype=torch.float32))
    if resume:
        head.load_state_dict(resume["state_dict"])
    optimizer = torch.optim.Adam(head.parameters(), lr=args.learning_rate)
    if resume and "optimizer" in resume:
        optimizer.load_state_dict(resume["optimizer"])
    manifest["resumed_after_pass"] = resume["epoch"] if resume else None
    manifest["optimizer_reinitialized_on_resume"] = bool(resume and "optimizer" not in resume)
    write_json(args.output / "manifest.json", manifest)
    start = time.monotonic()
    losses = []
    for epoch in range(resume["epoch"] if resume else 0, args.epochs):
        seen = 0
        running = 0.0
        n = 0
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * (0.2**epoch)
        for i, z, sh, st in batches("train", True):
            z = (z - mean) / std / math.sqrt(dimension)
            y, mask = targets(sh, st)
            prediction = head(z).reshape(-1, 2, 2)
            loss = (
                ((prediction - y).square().mean(-1) * mask).sum(0) / mask.sum(0).clamp_min(1)
            ).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            assert all(p.grad is None for p in encoder.parameters())
            running += loss.detach().item()
            n += 1
            seen += len(z)
            if (i + 1) % 200 == 0:
                status = dict(
                    phase="fit",
                    epoch=epoch + 1,
                    batches=i + 1,
                    samples=seen,
                    total_samples=len(datasets["train"]),
                    loss=running / n,
                    elapsed_seconds=time.monotonic() - start,
                )
                print(json.dumps(status), flush=True)
                write_json(args.output / "progress.json", status)
                losses.append(status)
                running = 0.0
                n = 0
        torch.save(
            {
                "state_dict": head.state_dict(),
                "feature_mean": mean.cpu(),
                "feature_std": std.cpu(),
                "dimension": dimension,
                "epoch": epoch + 1,
                "optimizer": optimizer.state_dict(),
            },
            args.output / "linear-probes.tmp",
        )
        (args.output / "linear-probes.tmp").replace(args.output / "linear-probes.pt")
    write_json(args.output / "learning_curve.json", losses)
    # No holdout sample has contributed to model fitting or normalization.
    scores = {
        name: Scores()
        for name in [
            "current",
            "next",
            "current_mean",
            "next_mean",
            "next_paired",
            "persistence",
            "current_readout_as_next",
            "displacement",
            "zero_displacement",
            "next_velocity_subset",
            "constant_velocity",
        ]
    }
    rows = 0
    for i, z, sh, st in batches("heldout"):
        with torch.no_grad():
            prediction = head((z - mean) / std / math.sqrt(dimension)).reshape(-1, 2, 2)
        y, mask = targets(sh, st)
        prediction, y, mask = prediction.cpu().numpy(), y.cpu().numpy(), mask.cpu().numpy()
        h = sh.cpu().numpy()
        for j, name in enumerate(["current", "next"]):
            scores[name].add(prediction[mask[:, j], j], y[mask[:, j], j])
            scores[name + "_mean"].add(
                np.broadcast_to(means[j], y[mask[:, j], j].shape), y[mask[:, j], j]
            )
        pair = mask.all(1)
        scores["next_paired"].add(prediction[pair, 1], y[pair, 1])
        scores["current_readout_as_next"].add(prediction[pair, 0], y[pair, 1])
        scores["displacement"].add(
            prediction[pair, 1] - prediction[pair, 0], y[pair, 1] - y[pair, 0]
        )
        scores["zero_displacement"].add(np.zeros_like(y[pair, 0]), y[pair, 1] - y[pair, 0])
        scores["persistence"].add(y[pair, 0], y[pair, 1])
        velocity = pair & (h[:, -2, 2] > 0) & (h[:, -2, 1] > 0)
        scores["next_velocity_subset"].add(prediction[velocity, 1], y[velocity, 1])
        scores["constant_velocity"].add(
            2 * h[velocity, -1, :2] - h[velocity, -2, :2], y[velocity, 1]
        )
        rows += len(z)
        if (i + 1) % 200 == 0:
            status = dict(
                phase="heldout",
                samples=rows,
                total_samples=len(datasets["heldout"]),
                elapsed_seconds=time.monotonic() - start,
            )
            print(json.dumps(status), flush=True)
            write_json(args.output / "progress.json", status)
    result = {k: v.result() for k, v in scores.items()}
    result["heldout_windows"] = rows
    result["train_windows_per_pass"] = len(datasets["train"])
    result["elapsed_seconds"] = time.monotonic() - start
    write_json(args.output / "results.json", result)
    write_json(args.output / "progress.json", {"phase": "complete"})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
