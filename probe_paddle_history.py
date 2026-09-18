"""Train small RGB-derived paddle probes with independent frame/action contexts.

An isolated experiment: no emulator approach or player behavior is changed.
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from gymemu.data import Frames, read_episodes

REVISION = "676ff6388f4218d3c3a3ce9f2f33e075fa7314a3"
DEFAULT_DATA = (
    Path.home()
    / f".cache/huggingface/hub/datasets--tsilva--gradlab-breakout-trajectories/snapshots/{REVISION}"
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def extract_paddle(rgb):
    """Read the visible paddle from RGB only; fixed game geometry, no label inputs."""
    pixels = rgb[:, 189:193, 8:152]
    mask = ((pixels[0] > 150) & (pixels[1] < 110) & (pixels[2] < 110)).all(0)
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts):
        return [0.0, 0.0, 0.0]
    i = np.argmax(ends - starts)
    width = ends[i] - starts[i]
    if width not in (12, 16) and not (ends[i] == 144 and 8 <= width <= 16):
        return [0.0, 0.0, 0.0]
    return [float(starts[i] + 8), float(width), 1.0]


def prepare(args):
    rng = np.random.default_rng(917)
    train = read_episodes(args.dataset, "train")
    heldout = read_episodes(args.dataset, "heldout")
    chosen = rng.permutation(len(train))[:160]
    groups = {
        "train": [train[i] for i in chosen[:128]],
        "validation": [train[i] for i in chosen[128:]],
        "test": [heldout[i] for i in rng.permutation(len(heldout))[:64]],
    }
    ids = {name: [e.episode_id for e in eps] for name, eps in groups.items()}
    assert not (set(ids["train"]) & set(ids["validation"]))
    assert not (set(ids["train"] + ids["validation"]) & set(ids["test"]))
    manifest = {
        "dataset": str(args.dataset.resolve()),
        "revision": args.dataset.name,
        "dataset_manifest_sha256": hashlib.sha256(
            (args.dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        "episode_ids": ids,
        "seed": 917,
        "targets": ["paddle_x / 65536", "paddle_vx_normalized * 160"],
        "velocity_units": "pixels per native tick, not displacement between captures",
        "features": "RGB-only visible paddle left edge, width, visibility at y=189:193",
        "accuracy_rule": "both MAE <= 0.25; >=95% jointly within 1 pixel and 1 pixel/native tick",
    }
    write_json(args.output / "manifest.json", manifest)
    by_id = {e.episode_id: e for eps in groups.values() for e in eps}
    labels = {eid: np.full((len(e.frames), 2), np.nan, np.float32) for eid, e in by_id.items()}
    for split in ["train", "heldout"]:
        selected = ids["train"] + ids["validation"] if split == "train" else ids["test"]
        paths = sorted((args.dataset / "transitions" / split).glob("*.parquet"))
        n = 0
        for j, path in enumerate(paths):
            data = pq.read_table(
                path,
                columns=["episode_id", "step", "successor_frame_id", "record_json"],
                filters=[("episode_id", "in", selected)],
            )
            for eid, step, fid, record in zip(*data.to_pydict().values(), strict=True):
                assert by_id[eid].frames[step + 1] == fid
                tree = dict(json.loads(json.loads(record)["structure"])[1])
                values = {k: v[1] for k, v in tree["labels"][1]}
                labels[eid][step + 1] = [
                    values["paddle_x"] / 65536,
                    values["paddle_vx_normalized"] * 160,
                ]
                n += 1
            if j % 100 == 0:
                print("labels", split, j, "/", len(paths), n, flush=True)
    print("Opening verified RGB cache", flush=True)
    frames = Frames(args.dataset, compact=True, cache=args.frame_cache)
    features = {}
    for j, (eid, e) in enumerate(by_id.items()):
        for fid in e.frames:
            if int(fid) not in features:
                features[int(fid)] = extract_paddle(frames.get(int(fid)).numpy())
        if j % 20 == 0:
            print("RGB episodes", j, "unique frames", len(features), flush=True)
    for name, eps in groups.items():
        payload = {}
        errors = []
        missing = 0
        for e in eps:
            x = np.array([features[int(fid)] for fid in e.frames], np.float32)
            y = labels[e.episode_id]
            assert np.isfinite(y[1:]).all()
            valid = x[1:, 2] > 0
            missing += int((~valid).sum())
            errors.extend((x[1:, 0][valid] - y[1:, 0][valid]).tolist())
            payload[f"{e.episode_id}_rgb"] = x
            payload[f"{e.episode_id}_labels"] = y
            payload[f"{e.episode_id}_actions"] = e.actions
            payload[f"{e.episode_id}_frames"] = e.frames
        np.savez_compressed(args.output / f"{name}.npz", **payload)
        manifest[name] = {
            "transitions": sum(len(e.actions) for e in eps),
            "rgb_missing": missing,
            "rgb_x_max_abs_error": float(np.abs(errors).max()),
            "rgb_x_exact_fraction": float((np.abs(errors) < 1e-6).mean()),
        }
        print(name, manifest[name], flush=True)
    write_json(args.output / "manifest.json", manifest)


def examples(path, frames, actions, samples, seed, current=False, categorical=False):
    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    bases = []
    identities = []
    with np.load(path) as data:
        eids = sorted(int(k.split("_")[0]) for k in data.files if k.endswith("_rgb"))
        for eid in eids:
            rgb = data[f"{eid}_rgb"]
            labels = data[f"{eid}_labels"]
            act = data[f"{eid}_actions"]
            # Same eligible positions across all context sizes; retain padded early windows.
            positions = np.arange(2, len(rgb))
            if samples and len(positions) > samples:
                positions = np.sort(rng.choice(positions, samples, replace=False))
            end = positions + int(current)
            fi = end[:, None] - np.arange(frames, 0, -1)[None, :]
            ai = positions[:, None] - np.arange(actions, 0, -1)[None, :]
            visible = rgb[np.maximum(fi, 0)].copy()
            visible[fi < 0] = 0
            base = rgb[positions - 1 + int(current), 0] if frames else np.full(len(positions), 80.0)
            f = []
            if frames:
                f = [
                    ((visible[:, :, 0] - base[:, None]) / 10) * (visible[:, :, 2] > 0),
                    visible[:, :, 1] / 16,
                    visible[:, :, 2],
                    base[:, None] / 160,
                ]
            if categorical and frames:
                f.append(np.eye(161, dtype=np.float32)[np.rint(base).astype(int)])
            if actions:
                a = act[np.maximum(ai, 0)] + 1
                a[ai < 0] = 0
                f.append(np.eye(4, dtype=np.float32)[a].reshape(len(a), -1))
            x = np.concatenate(f, axis=1) if f else np.zeros((len(positions), 1), np.float32)
            xs.append(x)
            ys.append(labels[positions])
            bases.append(base)
            identities.append(np.column_stack([np.full(len(positions), eid), positions]))
    return (
        torch.from_numpy(np.concatenate(xs).astype(np.float32)),
        np.concatenate(ys),
        np.concatenate(bases).astype(np.float32),
        np.concatenate(identities),
    )


class Probe(nn.Module):
    def __init__(self, inputs, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputs, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 2),
        )

    def forward(self, x):
        return self.net(x)


def score(pred, y, ids):
    err = np.abs(pred - y)
    return {
        "n": len(y),
        "mae": err.mean(0).tolist(),
        "rmse": np.sqrt((err**2).mean(0)).tolist(),
        "p95": np.quantile(err, 0.95, axis=0).tolist(),
        "within_1_joint": float((err <= 1).all(1).mean()),
        "rounded_exact_joint": float((np.rint(pred) == np.rint(y)).all(1).mean()),
        "moving": {"n": int((y[:, 1] != 0).sum()), "mae": err[y[:, 1] != 0].mean(0).tolist()},
        "opening": {
            "n": int((ids[:, 1] <= 64).sum()),
            "mae": err[ids[:, 1] <= 64].mean(0).tolist(),
        },
    }


def predict(model, x, base):
    with torch.inference_mode():
        out = torch.cat([model(b) for b in x.split(8192)]).numpy() * 10
    out[:, 0] += base
    return out


def train(args):
    torch.set_num_threads(2)
    for spec in args.contexts.split(","):
        h, a = map(int, spec.split(":"))
        started = time.monotonic()
        torch.manual_seed(args.seed)
        x, y, base, ids = examples(
            args.output / "train.npz", h, a, args.samples, 917, args.current, args.categorical
        )
        vx, vy, vbase, vids = examples(
            args.output / "validation.npz", h, a, 1024, 918, args.current, args.categorical
        )
        target = y.copy()
        target[:, 0] -= base
        target /= 10
        target = torch.from_numpy(target)
        model = Probe(x.shape[1], args.width)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.00001)
        best = float("inf")
        state = None
        records = []
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(len(x))
            for ix in order.split(1024):
                prediction = model(x[ix])
                loss = (prediction - target[ix]).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
            pred = predict(model, vx, vbase)
            mse = float(((pred - vy) ** 2).mean())
            if mse < best:
                best = mse
                state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
            records.append(mse)
            if epoch in [0, args.epochs - 1] or (epoch + 1) % 10 == 0:
                print(
                    spec,
                    "epoch",
                    epoch + 1,
                    "val_mse",
                    round(mse, 5),
                    "sec",
                    round(time.monotonic() - started, 1),
                    flush=True,
                )
            for g in optimizer.param_groups:
                g["lr"] = 0.002 * (0.15 + 0.85 * (1 - (epoch + 1) / args.epochs))
        model.load_state_dict(state)
        pred = predict(model, vx, vbase)
        name = (
            f"{'current' if args.current else 'next'}-f{h}-a{a}-s{args.seed}"
            f"-n{args.samples}-e{args.epochs}-w{args.width}"
            f"{'-cat' if args.categorical else ''}"
        )
        metrics = {
            "frames": h,
            "actions": a,
            "seed": args.seed,
            "training_samples": len(x),
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "seconds": time.monotonic() - started,
            "validation": score(pred, vy, vids),
            "validation_mse_curve": records,
        }
        torch.save(
            {
                "state_dict": state,
                "inputs": x.shape[1],
                "width": args.width,
                "frames": h,
                "actions": a,
                "current": args.current,
                "categorical": args.categorical,
                "manifest": json.loads((args.output / "manifest.json").read_text()),
            },
            args.output / f"{name}.pt",
        )
        write_json(args.output / f"{name}.json", metrics)
        print(name, json.dumps(metrics["validation"]), flush=True)


def infer_rgb(checkpoint, rgb_history, executed_actions):
    """Return [x pixels, velocity pixels/native tick] from raw uint8 CHW images.

    Images and actions are oldest-to-newest. For a next-state checkpoint, the
    last action leads to the unseen target. For a current-state checkpoint, it
    leads to the newest supplied image. Short episode prefixes are left padded.
    """
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    h, a = saved["frames"], saved["actions"]
    visible = np.zeros((h, 3), np.float32)
    recent = list(rgb_history)[-h:] if h else []
    for i, image in enumerate(recent, start=h - len(recent)):
        image = np.asarray(image)
        if image.shape != (3, 210, 160) or image.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB images with shape (3,210,160)")
        visible[i] = extract_paddle(image)
    base = visible[-1, 0] if h else 80.0
    parts = []
    if h:
        parts = [
            ((visible[:, 0] - base) / 10) * (visible[:, 2] > 0),
            visible[:, 1] / 16,
            visible[:, 2],
            np.array([base / 160]),
        ]
    if saved.get("categorical", False) and h:
        parts.append(np.eye(161, dtype=np.float32)[int(round(base))])
    if a:
        values = list(executed_actions)[-a:]
        if any(v not in (0, 1, 2) for v in values):
            raise ValueError("Recorded actions must be 0, 1, or 2")
        tokens = np.zeros(a, np.int64)
        if values:
            tokens[-len(values) :] = np.array(values) + 1
        parts.append(np.eye(4, dtype=np.float32)[tokens].reshape(-1))
    features = np.concatenate(parts) if parts else np.zeros(1)
    model = Probe(saved["inputs"], saved["width"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return predict(
        model, torch.from_numpy(features.astype(np.float32))[None], np.array([base], np.float32)
    )[0]


def evaluate(args):
    torch.set_num_threads(2)
    for filename in args.checkpoints:
        path = Path(filename)
        saved = torch.load(path, weights_only=True, map_location="cpu")
        model = Probe(saved["inputs"], saved["width"])
        model.load_state_dict(saved["state_dict"])
        model.eval()
        x, y, base, ids = examples(
            args.output / "test.npz",
            saved["frames"],
            saved["actions"],
            0,
            919,
            saved["current"],
            saved.get("categorical", False),
        )
        pred = predict(model, x, base)
        result = score(pred, y, ids)
        result["episode_mae"] = {
            str(eid): np.abs(pred[ids[:, 0] == eid] - y[ids[:, 0] == eid]).mean(0).tolist()
            for eid in np.unique(ids[:, 0])
        }
        write_json(path.with_suffix(".test.json"), result)
        np.savez_compressed(path.with_suffix(".test.npz"), prediction=pred, target=y, identity=ids)
        print(
            path.name,
            json.dumps({k: v for k, v in result.items() if k != "episode_mae"}),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "evaluate"])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--frame-cache", type=Path, default=Path.home() / ".cache/gymemu/676ff638-lz4"
    )
    parser.add_argument("--output", type=Path, default=Path("logs/paddle-history-20260917"))
    parser.add_argument("--contexts", default="1:1,2:1,4:1,8:1,16:1,2:4,2:16,2:64,8:16,16:64,64:64")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--current", action="store_true")
    parser.add_argument("--categorical", action="store_true")
    parser.add_argument("--checkpoints", nargs="*", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
