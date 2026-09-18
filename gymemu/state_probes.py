"""Independent one-target dynamics fits; no shared weights or generated feedback."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from gymemu.commands.dynamics import compose_state_config
from gymemu.models import build_model
from gymemu.state_data import FIELDS, NATIVE_SCALES, StateWindows, write_json
from gymemu.state_hidden_context import HiddenInputs
from gymemu.state_paddle_context import RecordedPaddleContext
from gymemu.state_probe_diagnostics import diagnose_target, sample_training_indices
from gymemu.state_training import batches, choose_threshold, terminal_metrics

TARGETS = (*FIELDS, "brick_grid", "terminal")
DEFAULT_TARGETS = tuple(target for target in TARGETS if target != "paddle_vx_normalized")


class SingleTargetApproach(torch.nn.Module):
    """Reuse the registered architecture but backpropagate exactly one target loss."""

    def __init__(self, config, target, positive_weight):
        super().__init__()
        if target not in TARGETS:
            raise ValueError("Unknown state target")
        if target == "paddle_vx_normalized" and not config["model"].get(
            "predict_paddle_velocity", True
        ):
            raise ValueError("Legacy velocity probes require predict_paddle_velocity=True")
        self.predictor = build_model(config["model"])
        self.target = target
        self.loss_config = config["loss"]
        self.positive_weight = positive_weight

    def forward(self, batch):
        extra = {"paddle_context": batch["paddle_context"]} if "paddle_context" in batch else {}
        if "hidden_state" in batch:
            extra["hidden_state"] = batch["hidden_state"]
        output, _ = self.predictor(
            batch["history"], batch["history_valid"], batch["action_history"], **extra
        )
        if self.target in FIELDS[:6]:
            return output["motion"][:, FIELDS.index(self.target)]
        return output[
            {FIELDS[6]: "width", "brick_grid": "bricks", "terminal": "terminal"}[self.target]
        ]

    def loss(self, prediction, batch):
        live = ~batch["terminal"][:, 0]
        y = batch["target"][:, 0]
        if self.target in FIELDS[:6]:
            k = FIELDS.index(self.target)
            scale = NATIVE_SCALES[k] / self.loss_config["motion_tolerances"][k]
            return ((prediction[live] - y[live, k]) * scale).square().sum() / live.sum().clamp_min(
                1
            )
        if self.target == "terminal":
            return F.binary_cross_entropy_with_logits(
                prediction,
                (~live).float(),
                pos_weight=torch.as_tensor(self.positive_weight, device=prediction.device),
            )
        if self.target == FIELDS[6]:
            return F.binary_cross_entropy_with_logits(
                prediction[live], (y[live, 6] < 0.875).float()
            )
        changed = y[..., 7:] != batch["history"][:, -1, 7:]
        weights = 1 + changed.float() * self.loss_config["brick_change_weight"]
        return (
            F.binary_cross_entropy_with_logits(prediction[live], y[live, 7:], reduction="none")
            * weights[live]
        ).mean()


@torch.no_grad()
def evaluate_target(model, data, indices, config, threshold=None):
    model.eval()
    predictions, targets, previous, terminals = [], [], [], []
    losses, count = 0.0, 0
    mc = config["model"]
    for ii in batches(indices, 2048):
        b = data.batch(ii, mc["history"], mc["past_actions"])
        if hasattr(data, "paddle_context"):
            data.paddle_context.attach(ii, b)
        if hasattr(data, "hidden_inputs"):
            data.hidden_inputs.attach(ii, b)
        p = model(b)
        if not torch.isfinite(p).all():
            raise FloatingPointError("Nonfinite isolated prediction")
        loss = model.loss(p, b)
        predictions.append(p.cpu().numpy())
        targets.append(b["target"][:, 0].numpy())
        previous.append(b["history"][:, -1].numpy())
        terminals.append(b["terminal"][:, 0].numpy())
        losses += float(loss) * len(ii)
        count += len(ii)
    p, y, old, terminal = map(np.concatenate, (predictions, targets, previous, terminals))
    live = ~terminal
    result = {
        "samples": count,
        "loss": losses / count,
        "evaluation_identity": hashlib.sha256(np.asarray(indices).tobytes()).hexdigest(),
    }
    target = model.target
    if target in FIELDS[:6]:
        k = FIELDS.index(target)
        err = (p[live] - y[live, k]) * NATIVE_SCALES[k]
        changed = np.abs(y[live, k] - old[live, k]) > 1e-5
        collision = np.any(np.abs(y[live, 2:4] - old[live, 2:4]) > 1e-5, axis=1)
        result.update(
            mae_native=float(np.abs(err).mean()),
            rmse_native=float(np.sqrt(np.square(err).mean())),
            p95_native=float(np.quantile(np.abs(err), 0.95)),
            within_quarter=float((np.abs(err) <= 0.25).mean()),
            within_one=float((np.abs(err) <= 1).mean()),
            changed_samples=int(changed.sum()),
            changed_mae_native=float(np.abs(err[changed]).mean()) if changed.any() else None,
            collision_samples=int(collision.sum()),
            collision_mae_native=float(np.abs(err[collision]).mean()) if collision.any() else None,
        )
        result["selection_score"] = result["rmse_native"]
    elif target == "terminal":
        prob = torch.tensor(p).sigmoid().numpy()
        if threshold is None:
            if data.split != "validation":
                raise ValueError("Choose a threshold on validation only")
            threshold = choose_threshold(prob, terminal)
        result.update(threshold=threshold, **terminal_metrics(prob, terminal, threshold))
        result["selection_score"] = -result["f1"]
    elif target == "brick_grid":
        pb, yb, ob = p[live] >= 0, y[live, 7:] >= 0.5, old[live, 7:] >= 0.5
        removal, truth = ob & ~pb, ob & ~yb
        tp = int((removal & truth).sum())
        fp = int((removal & ~truth).sum())
        fn = int((~removal & truth).sum())
        result.update(
            cell_accuracy=float((pb == yb).mean()),
            removal_precision=tp / max(tp + fp, 1),
            removal_recall=tp / max(tp + fn, 1),
            removal_f1=2 * tp / max(2 * tp + fp + fn, 1),
            removal_tp=tp,
            removal_fp=fp,
            removal_fn=fn,
            false_appearances=int((~ob & pb & ~yb).sum()),
        )
        result["selection_score"] = -result["removal_f1"]
    else:
        pred, truth, before = p[live] >= 0, y[live, 6] < 0.875, old[live, 6] < 0.875
        changed = truth != before
        result.update(
            accuracy=float((pred == truth).mean()),
            change_samples=int(changed.sum()),
            change_accuracy=float((pred[changed] == truth[changed]).mean())
            if changed.any()
            else None,
        )
        result["selection_score"] = result["loss"]
    return result


def train_target(
    config,
    target,
    output,
    epochs,
    sample_count,
    seed,
    paddle_contexts=None,
    *,
    initial_checkpoint=None,
    hidden_inputs=None,
    sampling="uniform",
    learning_rate=0.001,
):
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    train, validation = (
        StateWindows(config["cache"], "train"),
        StateWindows(config["cache"], "validation"),
    )
    if paddle_contexts:
        train.paddle_context, validation.paddle_context = paddle_contexts
    if hidden_inputs:
        train.hidden_inputs, validation.hidden_inputs = hidden_inputs
    # State targets at death are undefined; the terminal predictor keeps both classes.
    train_indices = np.array(train.indices)
    val_indices = np.array(validation.indices)
    if target != "terminal":
        train_indices = train_indices[~train.arrays["terminal"][train_indices]]
        val_indices = val_indices[~validation.arrays["terminal"][val_indices]]
    pos = train.manifest["train"]["terminals"]
    weight = min(100, (len(train.indices) - pos) / max(pos, 1))
    model = SingleTargetApproach(config, target, weight)
    initialization = None
    if initial_checkpoint is not None:
        checkpoint = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        if (
            checkpoint["format"] != "gymemu-state-probe-v1"
            or checkpoint["target"] != target
            or checkpoint["config"]["model"] != config["model"]
            or checkpoint["cache_identity"] != train.manifest["identity"]
        ):
            raise ValueError("Continuation checkpoint does not match target/model/data")
        model.load_state_dict(checkpoint["model"])
        initialization = {
            "path": str(initial_checkpoint),
            "sha256": hashlib.sha256(Path(initial_checkpoint).read_bytes()).hexdigest(),
            "optimizer": "fresh AdamW; optimizer state is unavailable in probe checkpoints",
        }
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    best = float("inf")
    best_metrics = None
    best_weights = None
    best_epoch = None
    write_json(
        output / "config.json",
        {
            "config": config,
            "target": target,
            "epochs": epochs,
            "sample_count": sample_count,
            "seed": seed,
            "cache_identity": train.manifest["identity"],
            "positive_weight": weight,
            "initialization": initialization,
            "hidden_input_provenance": train.hidden_inputs.provenance if hidden_inputs else None,
            "sampling": sampling,
            "learning_rate": learning_rate,
            "selection": "native RMSE; terminal/removal F1; width BCE, on validation only",
        },
    )
    for epoch in range(1, epochs + 1):
        chosen = sample_training_indices(train, train_indices, sample_count, seed + epoch, sampling)
        model.train()
        total = 0.0
        for ii in batches(chosen, 512):
            b = train.batch(ii, config["model"]["history"], config["model"]["past_actions"])
            if hasattr(train, "paddle_context"):
                train.paddle_context.attach(ii, b)
            if hasattr(train, "hidden_inputs"):
                train.hidden_inputs.attach(ii, b)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(model(b), b)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite isolated loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(ii)
        metrics = evaluate_target(model, validation, val_indices, config)
        row = {"epoch": epoch, "training_loss": total / len(chosen), "validation": metrics}
        with (output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        if metrics["selection_score"] < best:
            best = metrics["selection_score"]
            best_metrics = metrics
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch % 20 == 0:
            torch.save(
                {
                    "format": "gymemu-state-probe-v1",
                    "config": config,
                    "target": target,
                    "positive_weight": weight,
                    "model": best_weights,
                    "validation": best_metrics,
                    "cache_identity": train.manifest["identity"],
                    "best_epoch": best_epoch,
                    "through_epoch": epoch,
                },
                output / f"best-through-{epoch:03d}.pt",
            )
        if epoch in (1, epochs) or epoch % 5 == 0:
            print(
                json.dumps(
                    {
                        "run": output.name,
                        "epoch": epoch,
                        "training_loss": row["training_loss"],
                        "validation_score": metrics["selection_score"],
                    }
                ),
                flush=True,
            )
    model.load_state_dict(best_weights)
    train_subset = np.random.default_rng(812).choice(
        train_indices, min(20000, len(train_indices)), replace=False
    )
    train_metrics = evaluate_target(
        model, train, train_subset, config, threshold=best_metrics.get("threshold")
    )
    result = {
        "target": target,
        "model": config["model"],
        "best_epoch": best_epoch,
        "validation": best_metrics,
        "training_diagnostic": train_metrics,
        "test_evaluated": False,
    }
    torch.save(
        {
            "format": "gymemu-state-probe-v1",
            "config": config,
            "target": target,
            "positive_weight": weight,
            "model": best_weights,
            "validation": best_metrics,
            "cache_identity": train.manifest["identity"],
        },
        output / "best.pt",
    )
    result["event_diagnostics"] = diagnose_target(
        model, validation, config, threshold=best_metrics.get("threshold")
    )
    result["training_event_diagnostics"] = diagnose_target(
        model, train, config, indices=train_subset, threshold=best_metrics.get("threshold")
    )
    write_json(output / "summary.json", result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--contexts", default="1:7,8:7")
    p.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    p.add_argument("--legacy-paddle-velocity", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--samples", type=int, default=100000)
    p.add_argument("--seed", type=int, default=91)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--paddle-context", type=int, default=0)
    p.add_argument("--sampling", choices=("uniform", "balanced_events"), default="uniform")
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--hidden-inputs", type=Path)
    p.add_argument("--hidden-mode", choices=("none", "controller", "hits", "both"), default="both")
    args = p.parse_args(argv)
    if min(args.epochs, args.samples, args.threads) < 1:
        p.error("Epochs, samples, and threads must be positive")
    if args.hidden_inputs and not args.paddle_context:
        p.error("Hidden-input probes require --paddle-context")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    paddle_contexts = None
    if args.paddle_context:
        paddle_contexts = tuple(
            RecordedPaddleContext(StateWindows(args.cache, split), args.paddle_context)
            for split in ("train", "validation")
        )
    hidden_inputs = None
    if args.hidden_inputs:
        hidden_inputs = tuple(
            HiddenInputs(StateWindows(args.cache, split), args.hidden_inputs)
            for split in ("train", "validation")
        )
    result = []
    for context in args.contexts.split(","):
        h, a = map(int, context.split(":"))
        cfg = compose_state_config()
        cfg["cache"] = str(args.cache)
        cfg["model"].update(history=h, past_actions=a)
        if args.legacy_paddle_velocity:
            cfg["model"]["predict_paddle_velocity"] = True
        if args.paddle_context:
            cfg["model"].update(
                kind="state_paddle_context_probe", paddle_context=args.paddle_context
            )
        if args.hidden_inputs:
            cfg["model"].update(kind="state_hidden_probe", hidden_mode=args.hidden_mode)
            cfg["hidden_input_cache"] = str(args.hidden_inputs)
        for target in args.targets.split(","):
            result.append(
                train_target(
                    cfg,
                    target,
                    args.output / f"h{h}-a{a}-{target}",
                    args.epochs,
                    args.samples,
                    args.seed,
                    paddle_contexts,
                    hidden_inputs=hidden_inputs,
                    sampling=args.sampling,
                    learning_rate=args.learning_rate,
                )
            )
            write_json(args.output / "summary.json", result)


if __name__ == "__main__":
    main()
