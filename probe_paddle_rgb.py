"""Train full-RGB CNN state probes against recorded normalized paddle/ball labels."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from gymemu.data import Frames
from gymemu.models import build_model
from probe_paddle_history import score


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(args):
    original = json.loads((args.prior / "manifest.json").read_text())
    groups = original["episode_ids"]
    if args.include_ball:
        args.include_width = True
    targets = ["paddle_x_normalized", "paddle_vx_normalized"]
    scales = [160, 160]
    if args.include_width:
        targets.append("paddle_width_normalized")
        scales.append(16)
    if args.include_ball:
        targets.extend(
            ["ball_x_normalized", "ball_y_normalized", "ball_vx_normalized", "ball_vy_normalized"]
        )
        scales.extend([160, 255, 2, 3.375])
    arrays = {}
    for split in groups:
        with np.load(args.prior / f"{split}.npz") as old:
            arrays[split] = {
                key: old[key]
                for key in old.files
                if key.endswith("_frames") or key.endswith("_actions")
            }
            for eid in groups[split]:
                arrays[split][f"{eid}_labels"] = np.full(
                    (len(old[f"{eid}_frames"]), len(targets)), np.nan, np.float32
                )
    for split in ["train", "heldout"]:
        selected = groups["train"] + groups["validation"] if split == "train" else groups["test"]
        destinations = {eid: part for part in groups for eid in groups[part]}
        for j, path in enumerate(sorted((args.dataset / "transitions" / split).glob("*.parquet"))):
            table = pq.read_table(
                path,
                columns=["episode_id", "step", "successor_frame_id", "record_json"],
                filters=[("episode_id", "in", selected)],
            )
            for eid, step, fid, record in zip(*table.to_pydict().values(), strict=True):
                dst = arrays[destinations[eid]]
                assert dst[f"{eid}_frames"][step + 1] == fid
                tree = dict(json.loads(json.loads(record)["structure"])[1])
                labels = {k: v[1] for k, v in tree["labels"][1]}
                dst[f"{eid}_labels"][step + 1] = [labels[key] for key in targets]
                if args.include_width:
                    assert (
                        abs(labels["paddle_width_normalized"] * 16 - labels["paddle_width"]) < 1e-5
                    )
                if args.include_ball:
                    for key, scale, divisor in [
                        ("ball_x", 160, 65536),
                        ("ball_y", 255, 1),
                        ("ball_vx", 2, 65536),
                        ("ball_vy", 3.375, 65536),
                    ]:
                        assert (
                            abs(labels[key + "_normalized"] * scale - labels[key] / divisor) < 5e-5
                        )
            if j % 100 == 0:
                print("labels", split, j, flush=True)
    for split, data in arrays.items():
        assert all(
            np.isfinite(value[1:]).all() for key, value in data.items() if key.endswith("_labels")
        )
        np.savez_compressed(args.output / f"{split}.npz", **data)
    manifest = {
        "dataset_revision": original["revision"],
        "dataset_manifest_sha256": original["dataset_manifest_sha256"],
        "episode_ids": groups,
        "targets": targets,
        "native_scales": scales,
        "inputs": ("full uint8 3x210x160 RGB; /255 scaling, FrameCodec padding, one-hot actions"),
        "encoder": "FrameCodec: Conv4s2 3->32, ReLU, Conv4s2 32->64, ReLU, Conv4s2 64->32",
        "initialization": "from scratch; no pretrained weights or extracted paddle inputs",
        "criterion": "native MAE <=.25 for every label and joint within-1 >=.95",
    }
    assert (
        hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest()
        == manifest["dataset_manifest_sha256"]
    )
    write_json(args.output / "manifest.json", manifest)


def state_metrics(prediction, target, identity, scales):
    """Report native units while retaining the exact recorded normalized targets."""
    scales = np.asarray(scales, dtype=np.float32)
    metrics = score(prediction * scales, target * scales, identity)
    metrics["normalized_mae"] = np.abs(prediction - target).mean(0).tolist()
    error = np.abs(prediction - target) * scales
    metrics["within_1_per_target"] = (error <= 1).mean(0).tolist()
    if target.shape[1] >= 3:
        metrics["by_width"] = {}
        for width in np.unique(target[:, 2]):
            selected = target[:, 2] == width
            metrics["by_width"][str(float(width))] = {
                "n": int(selected.sum()),
                "mae": (np.abs(prediction[selected] - target[selected]) * scales).mean(0).tolist(),
            }
    if target.shape[1] == 7:
        metrics["paddle_within_1_joint"] = float((error[:, :3] <= 1).all(1).mean())
        metrics["ball_position_within_1_joint"] = float((error[:, 3:5] <= 1).all(1).mean())
        metrics["ball_velocity_within_025_joint"] = float((error[:, 5:7] <= 0.25).all(1).mean())
    return metrics


def load_expanded_outputs(model, saved, targets):
    """Preserve existing direct outputs while adding new trainable regression rows."""
    previous = saved["manifest"]["targets"]
    assert saved["spec"]["kind"] == "paddle_state_cnn"
    assert len(targets) > len(previous) and targets[: len(previous)] == previous
    state = model.state_dict()
    for key, value in saved["state_dict"].items():
        if key in ("head.4.weight", "head.4.bias"):
            state[key][: len(previous)].copy_(value)
        else:
            assert state[key].shape == value.shape
            state[key] = value
    model.load_state_dict(state)


def regression_loss(prediction, target, loss_scale, native_scale, kind="mse"):
    residual = (prediction - target) * loss_scale
    if kind == "mse":
        return residual.square().mean()
    # Same quadratic loss below one native unit; linear growth beyond it.
    threshold = loss_scale / native_scale
    absolute = residual.abs()
    return torch.where(
        absolute <= threshold,
        residual.square(),
        2 * threshold * absolute - threshold.square(),
    ).mean()


def ball_regime_metrics(prediction, target, identity, source, scales):
    """Post-hoc error slices from recorded labels, never model inputs or filters."""
    previous = np.empty_like(target)
    with np.load(source) as data:
        for eid in np.unique(identity[:, 0]):
            rows = np.flatnonzero(identity[:, 0] == eid)
            previous[rows] = data[f"{eid}_labels"][identity[rows, 1] - 1]
    changed = (np.abs(target[:, 5:7] - previous[:, 5:7]) > 1e-6).any(1)
    stationary = (np.abs(target[:, 3:5] - previous[:, 3:5]) < 1e-6).all(1)
    outside = (target[:, 4] * 255 >= 210) | (target[:, 4] < 0)
    result = {}
    for name, mask in {
        "velocity_unchanged": ~changed,
        "velocity_changed": changed,
        "position_unchanged": stationary,
        "ball_y_outside_canvas": outside,
        "ball_velocity_zero": (np.abs(target[:, 5:7]) < 1e-6).all(1),
        "ball_y_zero": target[:, 4] == 0,
    }.items():
        if mask.any():
            result[name] = {
                "n": int(mask.sum()),
                "normalized_mae": np.abs(prediction[mask] - target[mask]).mean(0).tolist(),
                "mae": (np.abs(prediction[mask] - target[mask]) * scales).mean(0).tolist(),
            }
    return result


def windows(path, history, action_history, samples, seed, current=False):
    rng = np.random.default_rng(seed)
    frame_ids = []
    actions = []
    labels = []
    identity = []
    with np.load(path) as data:
        eids = sorted(int(k.split("_")[0]) for k in data.files if k.endswith("_frames"))
        for eid in eids:
            ids = data[f"{eid}_frames"]
            act = data[f"{eid}_actions"]
            ys = data[f"{eid}_labels"]
            p = np.arange(2, len(ids))
            if samples and len(p) > samples:
                p = np.sort(rng.choice(p, samples, replace=False))
            fi = p[:, None] + int(current) - np.arange(history, 0, -1)[None, :]
            ai = p[:, None] - np.arange(action_history, 0, -1)[None, :]
            fs = ids[np.maximum(fi, 0)].copy()
            fs[fi < 0] = -1
            ac = act[np.maximum(ai, 0)].copy() + 1
            ac[ai < 0] = 0
            frame_ids.append(fs)
            actions.append(ac)
            labels.append(ys[p])
            identity.append(np.column_stack([np.full(len(p), eid), p]))
    return tuple(np.concatenate(x) for x in [frame_ids, actions, labels, identity])


def load_pixels(args, datasets):
    """Decode exact RGB using the verified cache; IDs map through explicit lookup."""
    started = time.monotonic()
    unique = np.unique(np.concatenate([d[0].reshape(-1) for d in datasets]))
    unique = unique[unique >= 0]
    print("Verifying cache; unique RGB frames", len(unique), flush=True)
    frames = Frames(args.dataset, compact=True, cache=args.frame_cache)
    frames.check_ids(unique)
    pixels = torch.empty((len(unique) + 1, 3, 210, 160), dtype=torch.uint8, device=args.device)
    pixels[0] = 0
    for start in range(0, len(unique), 1024):
        block = torch.stack([frames.get(int(fid)) for fid in unique[start : start + 1024]])
        pixels[start + 1 : start + 1 + len(block)].copy_(block)
        if start % 32768 == 0:
            print("RGB loaded", start, "seconds", round(time.monotonic() - started, 1), flush=True)
    result = []
    for ids, actions, labels, identity in datasets:
        index = np.searchsorted(unique, np.maximum(ids, 0)) + 1
        index[ids < 0] = 0
        result.append(
            (
                torch.tensor(index, device=args.device),
                torch.tensor(actions, device=args.device),
                torch.tensor(labels, device=args.device),
                identity,
            )
        )
    print("RGB ready", round(time.monotonic() - started, 1), flush=True)
    return pixels, result


def predict(model, pixels, data, batch_size):
    idx, actions, labels, identity = data
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(idx), batch_size):
            outputs.append(
                model(
                    pixels[idx[start : start + batch_size]], actions[start : start + batch_size]
                ).cpu()
            )
    return torch.cat(outputs).numpy(), labels.cpu().numpy(), identity


def position_targets(path, identity, history, current, device):
    """Recorded positions supervise the head; never used as predictor inputs."""
    targets = np.full((len(identity), history), -1, np.int64)
    with np.load(path) as data:
        for eid in np.unique(identity[:, 0]):
            rows = np.flatnonzero(identity[:, 0] == eid)
            index = identity[rows, 1, None] + int(current) - np.arange(history, 0, -1)[None, :]
            values = data[f"{eid}_labels"][np.maximum(index, 0), 0]
            good = (index >= 1) & np.isfinite(values)
            target = np.where(good, np.rint(np.nan_to_num(values) * 160), -1).astype(np.int64)
            targets[rows] = target
    return torch.tensor(targets, device=device)


def train(args):
    manifest = json.loads((args.output / "manifest.json").read_text())
    scales = np.asarray(manifest.get("native_scales", [160, 160]), dtype=np.float32)
    assert not args.position_head or len(scales) == 2
    contexts = [tuple(map(int, c.split(":"))) for c in args.contexts.split(",")]
    maxh, maxa = max(h for h, a in contexts), max(a for h, a in contexts)
    td = windows(args.output / "train.npz", maxh, maxa, args.samples, 917, args.current)
    vd = windows(
        args.output / "validation.npz", maxh, maxa, args.validation_samples, 918, args.current
    )
    pixels, (training, validation) = load_pixels(args, [td, vd])
    native_scale = torch.tensor(scales, device=args.device)
    if args.loss_weighting == "standardized":
        # Residual weighting only: labels and predictions retain their dataset units.
        loss_scale = training[2].std(dim=0, unbiased=False).clamp_min(1e-3).reciprocal()
    elif args.loss_weighting == "normalized":
        loss_scale = torch.ones(len(scales), device=args.device)
    else:
        loss_scale = torch.tensor(scales / 10, device=args.device)
    for context_index, (h, a) in enumerate(contexts):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        spec = {
            "kind": "paddle_position_cnn" if args.position_head else "paddle_state_cnn",
            "history": h,
            "action_history": a,
            "encoder_amp": not args.fp32,
            "width": 32,
            "latent_channels": 32,
            "hidden": 128,
        }
        if not args.position_head:
            spec["outputs"] = len(scales)
        model = build_model(spec).to(args.device)
        initialization = None
        if args.resume_checkpoints:
            previous = args.resume_checkpoints[context_index]
            saved = torch.load(previous, map_location="cpu", weights_only=True)
            assert saved["spec"]["history"] == h and saved["spec"]["action_history"] == a
            assert saved["current"] == args.current
            if args.expand_outputs:
                assert not args.position_head
                load_expanded_outputs(model, saved, manifest["targets"])
            elif args.position_head and saved["spec"]["kind"] != "paddle_position_cnn":
                vision = {
                    k: v
                    for k, v in saved["state_dict"].items()
                    if k.startswith(("encoder.", "frame_head."))
                }
                result = model.load_state_dict(vision, strict=False)
                assert not result.unexpected_keys
                assert all(
                    k.startswith(("head.", "position.", "coordinates")) for k in result.missing_keys
                )
            else:
                model.load_state_dict(saved["state_dict"])
            initialization = {
                "path": previous,
                "sha256": hashlib.sha256(Path(previous).read_bytes()).hexdigest(),
                "expanded_outputs": args.expand_outputs,
            }
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        train_data = (
            training[0][:, -h:],
            training[1][:, -a:] if a else training[1][:, :0],
            training[2],
            training[3],
        )
        val_data = (
            validation[0][:, -h:],
            validation[1][:, -a:] if a else validation[1][:, :0],
            validation[2],
            validation[3],
        )
        auxiliary = (
            position_targets(args.output / "train.npz", training[3], h, args.current, args.device)
            if args.position_head
            else None
        )
        started = time.monotonic()
        best = float("inf")
        curve = []
        best_state = None
        name = (
            f"{'current' if args.current else 'next'}-f{h}-a{a}-s{args.seed}"
            f"-n{args.samples}-e{args.epochs}{args.tag}"
        )
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(len(train_data[0]), device=args.device)
            for ix in order.split(args.batch_size):
                if args.position_head:
                    pred, logits = model.forward_with_positions(
                        pixels[train_data[0][ix]], train_data[1][ix]
                    )
                    auxiliary_loss = F.cross_entropy(
                        logits.flatten(0, 1), auxiliary[ix].flatten(), ignore_index=-1
                    )
                else:
                    pred = model(pixels[train_data[0][ix]], train_data[1][ix])
                    auxiliary_loss = 0.0
                loss = (
                    regression_loss(
                        pred, train_data[2][ix], loss_scale, native_scale, args.regression_loss
                    )
                    + auxiliary_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
            pred, y, ids = predict(model, pixels, val_data, args.batch_size)
            metrics = state_metrics(pred, y, ids, scales)
            mse = float(np.square((pred - y) * scales).mean())
            selection = max(metrics["mae"]) if args.selection == "worst-native-mae" else mse
            curve.append(
                {
                    "epoch": epoch + 1,
                    "mse": mse,
                    "selection": selection,
                    "metrics": metrics,
                    "seconds": time.monotonic() - started,
                }
            )
            if selection < best:
                best = selection
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            for g in optimizer.param_groups:
                g["lr"] = args.lr * (0.1 + 0.9 * (1 - (epoch + 1) / args.epochs))
            print(
                name,
                "epoch",
                epoch + 1,
                "mae",
                metrics["mae"],
                "mse",
                mse,
                "sec",
                round(time.monotonic() - started, 1),
                flush=True,
            )
            write_json(args.output / f"{name}.progress.json", curve)
        model.load_state_dict(best_state)
        pred, y, ids = predict(model, pixels, val_data, args.batch_size)
        receipt = {
            "spec": spec,
            "initialization": initialization,
            "auxiliary_position_cross_entropy_weight": 1.0 if args.position_head else 0.0,
            "loss_weighting": args.loss_weighting,
            "loss_scales": loss_scale.cpu().tolist(),
            "regression_loss": args.regression_loss,
            "selection_metric": args.selection,
            "best_validation_selection": best,
            "seed": args.seed,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "samples_per_episode": args.samples,
            "training_samples": len(train_data[0]),
            "validation": state_metrics(pred, y, ids, scales),
            "seconds": time.monotonic() - started,
            "curve": curve,
        }
        torch.save(
            {
                "state_dict": best_state,
                "spec": spec,
                "current": args.current,
                "manifest": json.loads((args.output / "manifest.json").read_text()),
            },
            args.output / f"{name}.pt",
        )
        write_json(args.output / f"{name}.json", receipt)
        print("COMPLETED", name, json.dumps(receipt["validation"]), flush=True)
        if args.head_epochs:
            assert args.position_head
            refine_head(args, model, pixels, train_data, val_data, name, spec)
        del model, optimizer
        if args.device == "cuda":
            torch.cuda.empty_cache()


def cache_positions(model, pixels, data, batch_size):
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data[0]), batch_size):
            logits = model.position(
                model.encode_history(pixels[data[0][start : start + batch_size]])
            )
            chunks.append(logits.softmax(-1))
        result = torch.cat(chunks)
    return result.clone()


def refine_head(
    args, model, pixels, training, validation, name, spec, cached=None, vision_checkpoint=None
):
    """Cache learned CNN probabilities, freeze perception, and fit dynamics cheaply."""
    model.eval()
    model.requires_grad_(False)
    train_p, val_p = (
        cached
        if cached is not None
        else [
            cache_positions(model, pixels, data, args.batch_size) for data in [training, validation]
        ]
    )
    valid_positions = position_targets(
        args.output / "validation.npz", validation[3], spec["history"], args.current, args.device
    )
    valid = valid_positions >= 0
    position_accuracy = float(((val_p.argmax(-1) == valid_positions)[valid]).float().mean())
    model.head.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.002, weight_decay=1e-5)
    y = validation[2].cpu().numpy()

    def validate():
        model.head.eval()
        with torch.no_grad():
            pred = model.predict_from_positions(val_p, validation[1]).cpu().numpy()
        return pred, float(np.square((pred - y) * 160).mean())

    pred, best = validate()
    best_state = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
    best_epoch = 0
    curve = []
    started = time.monotonic()
    for epoch in range(args.head_epochs):
        # Leave encoder, image projection, and classifier frozen and in eval mode.
        model.head.train()
        order = torch.randperm(len(training[0]), device=args.device)
        for ix in order.split(1024):
            pred = model.predict_from_positions(train_p[ix], training[1][ix])
            loss = ((pred - training[2][ix]) * 16).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        pred, mse = validate()
        curve.append({"epoch": epoch + 1, "mse": mse})
        if mse < best:
            best = mse
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
        for group in optimizer.param_groups:
            group["lr"] = 0.002 * (0.1 + 0.9 * (1 - (epoch + 1) / args.head_epochs))
    model.head.load_state_dict(best_state)
    pred, _ = validate()
    receipt = {
        "spec": spec,
        "stage": "frozen learned perception, dynamics refinement",
        "vision_checkpoint": vision_checkpoint or name + ".pt",
        "encoder_frozen": True,
        "position_class_accuracy": position_accuracy,
        "best_epoch": best_epoch,
        "epochs": args.head_epochs,
        "seconds": time.monotonic() - started,
        "validation": score(pred * 160, y * 160, validation[3]),
        "curve": curve,
    }
    with torch.inference_mode():
        raw = model(pixels[training[0][:64]], training[1][:64])
        cached_output = model.predict_from_positions(train_p[:64], training[1][:64])
        receipt["cache_to_rgb_max_abs_native"] = float((raw - cached_output).abs().max() * 160)
        assert receipt["cache_to_rgb_max_abs_native"] < 0.001
    torch.save(
        {
            "state_dict": model.state_dict(),
            "spec": spec,
            "current": args.current,
            "manifest": json.loads((args.output / "manifest.json").read_text()),
        },
        args.output / f"{name}-head.pt",
    )
    write_json(args.output / f"{name}-head.json", receipt)
    print("HEAD COMPLETED", name, json.dumps(receipt), flush=True)


def heads(args):
    """Compare histories with identical frozen, learned RGB perception."""
    saved = torch.load(args.resume_checkpoints[0], weights_only=True, map_location="cpu")
    assert saved["spec"]["kind"] == "paddle_position_cnn"
    assert saved["current"] == args.current
    contexts = [tuple(map(int, c.split(":"))) for c in args.contexts.split(",")]
    hmax, amax = max(h for h, a in contexts), max(a for h, a in contexts)
    raw_train = windows(args.output / "train.npz", hmax, amax, args.samples, 917, args.current)
    raw_val = windows(
        args.output / "validation.npz", hmax, amax, args.validation_samples, 918, args.current
    )
    pixels, (training, validation) = load_pixels(args, [raw_train, raw_val])
    vision = {k: v for k, v in saved["state_dict"].items() if not k.startswith("head.")}

    def new_model(h, a):
        torch.manual_seed(args.seed)
        spec = dict(saved["spec"], history=h, action_history=a, encoder_amp=False)
        model = build_model(spec).to(args.device)
        result = model.load_state_dict(vision, strict=False)
        assert not result.unexpected_keys
        assert all(k.startswith("head.") for k in result.missing_keys)
        model.eval()
        model.requires_grad_(False)
        return model, spec

    perception, _ = new_model(hmax, amax)
    train_p, val_p = [
        cache_positions(perception, pixels, data, args.batch_size)
        for data in [training, validation]
    ]
    del perception
    for h, a in contexts:
        model, spec = new_model(h, a)
        td = (
            training[0][:, -h:],
            training[1][:, -a:] if a else training[1][:, :0],
            training[2],
            training[3],
        )
        vd = (
            validation[0][:, -h:],
            validation[1][:, -a:] if a else validation[1][:, :0],
            validation[2],
            validation[3],
        )
        alignment = "current" if args.current else "next"
        name = f"{alignment}-f{h}-a{a}-s{args.seed}-n{args.samples}-position-search"
        refine_head(
            args,
            model,
            pixels,
            td,
            vd,
            name,
            spec,
            cached=(train_p[:, -h:], val_p[:, -h:]),
            vision_checkpoint=args.resume_checkpoints[0],
        )
        del model


def evaluate(args):
    checkpoints = [
        (Path(name), torch.load(name, map_location="cpu", weights_only=True))
        for name in args.checkpoints
    ]
    # Decode each held-out RGB frame once for all checkpoints with the same target alignment.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for current in sorted({saved["current"] for _, saved in checkpoints}):
        group = [(path, saved) for path, saved in checkpoints if saved["current"] == current]
        h = max(saved["spec"]["history"] for _, saved in group)
        a = max(saved["spec"]["action_history"] for _, saved in group)
        raw = windows(
            args.output / f"{args.evaluation_split}.npz",
            h,
            a,
            args.validation_samples if args.evaluation_split == "validation" else args.test_samples,
            918 if args.evaluation_split == "validation" else 919,
            current,
        )
        pixels, (common,) = load_pixels(args, [raw])
        for path, saved in group:
            spec = saved["spec"]
            model = build_model(spec).to(args.device)
            model.load_state_dict(saved["state_dict"])
            count = spec["action_history"]
            actions = common[1][:, -count:] if count else common[1][:, :0]
            data = (common[0][:, -spec["history"] :], actions, common[2], common[3])
            pred, y, ids = predict(model, pixels, data, args.batch_size)
            scales = np.asarray(
                saved["manifest"].get("native_scales", [160, 160]), dtype=np.float32
            )
            suffix = ".validation-fp32" if args.evaluation_split == "validation" else ".test"
            metrics = state_metrics(pred, y, ids, scales)
            if y.shape[1] == 7:
                metrics["ball_regimes"] = ball_regime_metrics(
                    pred, y, ids, args.output / f"{args.evaluation_split}.npz", scales
                )
            write_json(path.with_suffix(suffix + ".json"), metrics)
            np.savez_compressed(
                path.with_suffix(suffix + ".npz"),
                prediction=pred * scales,
                target=y * scales,
                prediction_normalized=pred,
                target_normalized=y,
                identity=ids,
            )
            print(path.name, metrics, flush=True)
            del model, data
        del pixels, common
        if args.device == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "heads", "evaluate"])
    parser.add_argument("--prior", type=Path, default=Path("logs/paddle-history-20260917"))
    parser.add_argument("--output", type=Path, default=Path("logs/paddle-rgb-20260917"))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path.home()
        / ".cache/huggingface/hub/datasets--tsilva--gradlab-breakout-trajectories"
        / "snapshots/676ff6388f4218d3c3a3ce9f2f33e075fa7314a3",
    )
    parser.add_argument(
        "--frame-cache", type=Path, default=Path.home() / ".cache/gymemu/676ff638-lz4"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--contexts", default="1:4,2:2,2:4")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--evaluation-split", choices=["validation", "test"], default="test")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--regression-loss", choices=["mse", "huber"], default="mse")
    parser.add_argument(
        "--loss-weighting", choices=["native", "normalized", "standardized"], default="native"
    )
    parser.add_argument(
        "--selection", choices=["native-mse", "worst-native-mae"], default="native-mse"
    )
    parser.add_argument("--current", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--position-head", action="store_true")
    parser.add_argument(
        "--include-width",
        action="store_true",
        help="Prepare the recorded normalized paddle width too",
    )
    parser.add_argument(
        "--include-ball",
        action="store_true",
        help="Prepare all seven normalized paddle and ball targets; implies --include-width",
    )
    parser.add_argument(
        "--expand-outputs",
        action="store_true",
        help="Initialize added direct regression outputs while retaining prior outputs",
    )
    parser.add_argument("--head-epochs", type=int, default=0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--resume-checkpoints", nargs="*", default=[])
    parser.add_argument("--checkpoints", nargs="*", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = not args.fp32
        torch.backends.cudnn.allow_tf32 = not args.fp32
    {"prepare": prepare, "train": train, "heads": heads, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
