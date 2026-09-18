"""Learn paddle position or charge changes from current source-state inputs."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from gymemu.cache import file_hash
from gymemu.models import build_model
from gymemu.state_data import StateWindows, write_json
from gymemu.state_hidden_context import HIDDEN_SCALES, HiddenInputs

FORMAT = "gymemu-paddle-transition-v1"


def load_data(cache, hidden, split, *, allow_test=False, target="paddle_x", annotations=None):
    if split not in ("train", "validation") and not (allow_test and split == "test"):
        raise ValueError("Development fits use train/validation only")
    d = StateWindows(cache, split)
    h = HiddenInputs(d, hidden)
    a = d.arrays
    ii = d.indices[~a["terminal"][d.indices]]
    current = np.rint(a["states"][ii, 4] * 160)
    controller = np.rint(h.values[ii, :4] * HIDDEN_SCALES[:4])
    source = np.column_stack((current, controller, a["actions"][ii])).astype(np.float32)
    delta = np.rint(a["states"][ii + 1, 4] * 160) - current
    annotation_provenance = None
    if target == "paddle_charge":
        if annotations is None:
            raise ValueError("Charge targets require an annotated dataset")
        root = Path(annotations)
        receipt_root = root / "annotations/breakout-paddle-controller-v1"
        report = json.loads((receipt_root / "validation.json").read_text())
        if report["status"] != "validated" or d.manifest["dataset_manifest_sha256"] not in (
            report["source_manifest_sha256"],
            file_hash(root / "manifest.json"),
        ):
            raise ValueError("Charge annotations differ from state cache provenance")
        hashes = {
            r["relative"]: r["sha256"]
            for r in json.loads((receipt_root / "files.json").read_text())
        }
        positions = {(int(a["episode_ids"][i]), int(a["steps"][i])): j for j, i in enumerate(ii)}
        found = np.zeros(len(ii), bool)
        delta = np.zeros(len(ii), np.float32)
        original_split = "heldout" if split == "test" else "train"
        episode_ids = np.unique(a["episode_ids"][ii]).tolist()
        for path in sorted((root / "transitions" / original_split).glob("*.parquet")):
            if file_hash(path) != hashes[str(path.relative_to(root))]:
                raise ValueError("Charge annotation checksum mismatch")
            rows = pq.read_table(
                path,
                columns=["episode_id", "step", "source_paddle_charge", "paddle_charge"],
                filters=[("episode_id", "in", episode_ids)],
            ).to_pylist()
            for row in rows:
                j = positions.get((row["episode_id"], row["step"]))
                if j is None:
                    continue
                if found[j] or row["source_paddle_charge"] != source[j, 1]:
                    raise ValueError("Duplicate or misaligned charge annotation")
                delta[j] = row["paddle_charge"] - row["source_paddle_charge"]
                found[j] = True
        if not found.all():
            raise ValueError("Missing charge targets")
        annotation_provenance = {
            "dataset": str(root.resolve()),
            "manifest_sha256": file_hash(root / "manifest.json"),
            "target_sha256": hashlib.sha256(delta.tobytes()).hexdigest(),
        }
    elif target != "paddle_x":
        raise ValueError("Unknown paddle transition target")
    return {
        "x": torch.from_numpy(source),
        "y": torch.from_numpy(delta.astype(np.float32)),
        "episodes": np.array(a["episode_ids"][ii]),
        "steps": np.array(a["steps"][ii]),
        "early": np.array(ii - a["starts"][ii] < 8),
        "identity": hashlib.sha256(ii.tobytes()).hexdigest(),
        "cache_identity": d.manifest["identity"],
        "hidden_provenance": h.provenance,
        "target": target,
        "annotation_provenance": annotation_provenance,
    }


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    raw = torch.cat([model.delta(model(x)) for x in data["x"].split(4096)])
    err = raw - data["y"]
    rounded = raw.round() - data["y"]
    result = {
        "samples": len(raw),
        "evaluation_identity": data["identity"],
        "mae_pixels": float(err.abs().mean()),
        "rmse_pixels": float(err.square().mean().sqrt()),
        "exact_pixel_accuracy": float((rounded == 0).float().mean()),
        "rounded_mae_pixels": float(rounded.abs().mean()),
        "p95_pixels": float(torch.quantile(err.abs(), 0.95)),
        "max_pixel_error": float(rounded.abs().max()),
        "errors": int((rounded != 0).sum()),
        "groups": {},
        "target": data.get("target", "paddle_x"),
    }
    native_names = {
        "mae_pixels": "mae_native",
        "rmse_pixels": "rmse_native",
        "exact_pixel_accuracy": "exact_integer_accuracy",
        "rounded_mae_pixels": "rounded_mae_native",
        "p95_pixels": "p95_native",
        "max_pixel_error": "max_error_native",
    }
    for old, new in native_names.items():
        result[new] = result[old]
        if result["target"] != "paddle_x":
            del result[old]
    for name, mask in [
        ("moving", data["y"].numpy() != 0),
        ("stationary", data["y"].numpy() == 0),
        ("early", data["early"]),
        ("repeat_unsaturated", data["x"][:, 3].numpy() < 60),
    ]:
        selected = rounded[mask]
        accuracy_name = (
            "exact_pixel_accuracy" if result["target"] == "paddle_x" else "exact_integer_accuracy"
        )
        result["groups"][name] = {
            "samples": len(selected),
            accuracy_name: float((selected == 0).float().mean()) if len(selected) else None,
        }
    result["per_episode"] = {
        str(eid): {
            "samples": int((data["episodes"] == eid).sum()),
            "errors": int((rounded[data["episodes"] == eid] != 0).sum()),
        }
        for eid in np.unique(data["episodes"])
    }
    bad = torch.nonzero(rounded != 0).flatten().numpy()
    result["examples"] = [
        {
            "episode": int(data["episodes"][i]),
            "step": int(data["steps"][i]),
            "source": data["x"][i].tolist(),
            "target_delta": float(data["y"][i]),
            "predicted_delta": float(raw[i]),
        }
        for i in bad[:20]
    ]
    return result


def load_checkpoint(path):
    c = torch.load(path, map_location="cpu", weights_only=True)
    if c["format"] != FORMAT or c["model_spec"]["kind"] != "paddle_transition_mlp":
        raise ValueError("Not an isolated paddle transition checkpoint")
    model = build_model(c["model_spec"])
    model.load_state_dict(c["model"])
    model.eval()
    return model, c


def train(
    train_data,
    val_data,
    output,
    *,
    encoding="scalar",
    objective="regression",
    width=128,
    depth=2,
    epochs=100,
    seed=1591,
    lr=0.001,
    batch_size=1024,
    controller_fields=("charge", "measure", "repeat", "held"),
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    target = train_data.get("target", "paddle_x")
    if val_data.get("target", "paddle_x") != target:
        raise ValueError("Training and validation targets differ")
    lower, upper = int(train_data["y"].min()), int(train_data["y"].max())
    if val_data["y"].min() < lower or val_data["y"].max() > upper:
        raise ValueError("Validation contains unseen movement classes; do not silently clip")
    spec = dict(
        kind="paddle_transition_mlp",
        encoding=encoding,
        objective=objective,
        width=width,
        depth=depth,
        delta_min=lower,
        delta_max=upper,
        controller_fields=list(controller_fields),
    )
    values = None
    if target == "paddle_charge" and objective == "classification":
        values = torch.unique(train_data["y"], sorted=True)
        if not torch.isin(val_data["y"], values).all():
            raise ValueError("Validation contains unseen charge-change classes")
        spec["delta_values"] = values.long().tolist()
    config = {
        "model": spec,
        "epochs": epochs,
        "seed": seed,
        "lr": lr,
        "batch_size": batch_size,
        "training_samples": len(train_data["x"]),
        "train_identity": train_data["identity"],
        "validation_identity": val_data["identity"],
        "cache_identity": train_data["cache_identity"],
        "hidden_provenance": train_data["hidden_provenance"],
        "selection": "lowest validation integer errors, then raw native-unit RMSE",
        "target": target,
        "annotation_provenance": train_data.get("annotation_provenance"),
        "features": ["paddle_x", *controller_fields, "requested_action"],
        "test_evaluated": False,
    }
    write_json(output / "config.json", config)
    model = build_model(spec)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr / 20)
    best = (float("inf"), float("inf"))
    best_weights = None
    best_metrics = None
    begin = time.monotonic()
    for epoch in range(1, epochs + 1):
        order = np.random.default_rng(seed + epoch).permutation(len(train_data["x"]))
        model.train()
        total = 0.0
        for offset in range(0, len(order), batch_size):
            ii = order[offset : offset + batch_size]
            x, y = train_data["x"][ii], train_data["y"][ii]
            opt.zero_grad(set_to_none=True)
            p = model(x)
            loss = (
                F.cross_entropy(
                    p, torch.searchsorted(values, y) if values is not None else y.long() - lower
                )
                if objective == "classification"
                else F.mse_loss(p[:, 0], y)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite isolated paddle loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            opt.step()
            total += float(loss.detach()) * len(ii)
        scheduler.step()
        metrics = evaluate(model, val_data)
        score = (metrics["errors"], metrics["rmse_native"])
        row = {
            "epoch": epoch,
            "training_loss": total / len(order),
            "validation": metrics,
            "seconds": time.monotonic() - begin,
        }
        with (output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        if score < best:
            best = score
            best_weights = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            selected_epoch = epoch
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                json.dumps(
                    {
                        "run": output.name,
                        "epoch": epoch,
                        "loss": row["training_loss"],
                        "mae": metrics["mae_native"],
                        "exact": metrics["exact_integer_accuracy"],
                        "best_errors": best[0],
                    }
                ),
                flush=True,
            )
    model.load_state_dict(best_weights)
    checkpoint = {
        "format": FORMAT,
        "model_spec": spec,
        "model": best_weights,
        "config": config,
        "best_epoch": selected_epoch,
        "validation": best_metrics,
    }
    torch.save(checkpoint, output / "best.pt")
    reloaded, _ = load_checkpoint(output / "best.pt")
    torch.testing.assert_close(model(train_data["x"][:32]), reloaded(train_data["x"][:32]))
    result = {
        "best_epoch": selected_epoch,
        "validation": best_metrics,
        "training": evaluate(model, train_data),
        "seconds": time.monotonic() - begin,
        "parameters": sum(p.numel() for p in model.parameters()),
        "test_evaluated": False,
    }
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--target", choices=("paddle_x", "paddle_charge"), default="paddle_x")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoding", choices=("scalar", "hybrid"), default="scalar")
    parser.add_argument(
        "--objective", choices=("regression", "classification"), default="regression"
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1591)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--controller-fields",
        nargs="*",
        choices=("charge", "measure", "repeat", "held"),
        default=["charge", "measure", "repeat", "held"],
    )
    args = parser.parse_args()
    if min(args.epochs, args.threads, args.width, args.depth) < 1:
        parser.error("Dimensions and budgets must be positive")
    torch.set_num_threads(args.threads)
    train_data = load_data(
        args.cache, args.hidden, "train", target=args.target, annotations=args.annotations
    )
    val_data = load_data(
        args.cache, args.hidden, "validation", target=args.target, annotations=args.annotations
    )
    train(
        train_data,
        val_data,
        args.output,
        encoding=args.encoding,
        objective=args.objective,
        width=args.width,
        depth=args.depth,
        epochs=args.epochs,
        seed=args.seed,
        controller_fields=args.controller_fields,
    )


if __name__ == "__main__":
    main()
