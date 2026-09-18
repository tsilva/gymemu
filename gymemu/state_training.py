"""Reproducible state-only training, selection, and held-out rollout diagnostics."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gymemu.state_approach import StateApproach, StatePlayer
from gymemu.state_data import FIELDS, NATIVE_SCALES, RULES, StateWindows, write_json

CHECKPOINT_FORMAT = "gymemu-state-dynamics-v1"


def batches(indices, size):
    for offset in range(0, len(indices), size):
        yield indices[offset : offset + size]


def terminal_metrics(probability, truth, threshold):
    predicted = probability >= threshold
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    tn = int((~predicted & ~truth).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "miss_rate": fn / max(fn + tp, 1),
    }


def choose_threshold(probability, truth):
    # On validation only. Prefer fewer false early stops on equal F1.
    candidates = np.unique(np.r_[np.linspace(0.01, 0.99, 99), 0.995, 0.999])
    return float(
        max(
            candidates,
            key=lambda t: (
                terminal_metrics(probability, truth, t)["f1"],
                -terminal_metrics(probability, truth, t)["fp"],
                t,
            ),
        )
    )


class StateScores:
    def __init__(self, predict_paddle_velocity=True):
        self.motion_count = 6 if predict_paddle_velocity else 5
        self.n = 0
        self.absolute = np.zeros(6, np.float64)
        self.square = np.zeros(6, np.float64)
        self.within = 0
        self.width_correct = self.brick_correct = 0
        self.removal_tp = self.removal_fp = self.removal_fn = self.reappear = 0

    def add(self, prediction, target, previous, valid):
        p, y, old = prediction[valid], target[valid], previous[valid]
        self.n += len(y)
        error = (p[:, :6] - y[:, :6]) * NATIVE_SCALES
        self.absolute += np.abs(error).sum(0)
        self.square += np.square(error).sum(0)
        self.within += int((np.abs(error[:, :2]).max(-1) <= 1).sum())
        self.width_correct += int((np.abs(p[:, 6] - y[:, 6]) < 0.01).sum())
        pb, yb, ob = p[:, 7:] >= 0.5, y[:, 7:] >= 0.5, old[:, 7:] >= 0.5
        self.brick_correct += int((pb == yb).sum())
        predicted_removal, true_removal = ob & ~pb, ob & ~yb
        self.removal_tp += int((predicted_removal & true_removal).sum())
        self.removal_fp += int((predicted_removal & ~true_removal).sum())
        self.removal_fn += int((~predicted_removal & true_removal).sum())
        self.reappear += int((~ob & pb & ~yb).sum())

    def result(self):
        n = max(self.n, 1)
        return {
            "samples": self.n,
            "mae_native": dict(zip(FIELDS[: self.motion_count], (self.absolute / n).tolist())),
            "rmse_native": dict(
                zip(FIELDS[: self.motion_count], np.sqrt(self.square / n).tolist())
            ),
            "ball_position_within_one": self.within / n,
            "width_accuracy": self.width_correct / n,
            "brick_cell_accuracy": self.brick_correct / (n * 108),
            "brick_removal_precision": self.removal_tp / max(self.removal_tp + self.removal_fp, 1),
            "brick_removal_recall": self.removal_tp / max(self.removal_tp + self.removal_fn, 1),
            "brick_removal_tp": self.removal_tp,
            "brick_removal_fp": self.removal_fp,
            "brick_removal_fn": self.removal_fn,
            "spurious_brick_reappearances": self.reappear,
        }


def persistence(history):
    return history.copy()


def constant_motion(history):
    p = history.copy()
    p[:, 0] += 2 * history[:, 2] * 2 / 160
    p[:, 1] += 2 * history[:, 3] * (27 / 8) / 255
    p[:, 4] += 2 * history[:, 5]
    return p


@torch.no_grad()
def evaluate(approach, data, config, *, threshold=None, samples=None, starts=None):
    approach.eval()
    device = next(approach.parameters()).device
    history, past = config["model"]["history"], config["model"]["past_actions"]
    ec, tc = config["evaluation"], config["trainer"]
    indices = data.select(ec["samples"] if samples is None else samples, ec["seed"])
    include_velocity = approach.predictor.predict_paddle_velocity
    scores = {
        name: StateScores(include_velocity)
        for name in ("all", "collision", "brick_change", "ordinary")
    }
    baselines = {name: StateScores(include_velocity) for name in ("persistence", "constant_motion")}
    probabilities, truths = [], []
    begin = time.perf_counter()
    for ii in batches(indices, tc["batch_size"]):
        batch = data.batch(ii, history, past, device=device)
        output, states = approach.rollout(batch)
        if not torch.isfinite(states).all():
            raise FloatingPointError("Nonfinite one-step prediction")
        p = states[:, 0].cpu().numpy()
        y, old = batch["target"][:, 0].cpu().numpy(), batch["history"][:, -1].cpu().numpy()
        terminal = batch["terminal"][:, 0].cpu().numpy()
        valid = ~terminal
        collision = np.any(np.abs(y[:, 2:4] - old[:, 2:4]) > 1e-5, axis=1)
        changed = np.any(y[:, 7:] != old[:, 7:], axis=1)
        for name, mask in (
            ("all", valid),
            ("collision", valid & collision),
            ("brick_change", valid & changed),
            ("ordinary", valid & ~collision & ~changed),
        ):
            scores[name].add(p, y, old, mask)
        baselines["persistence"].add(persistence(old), y, old, valid)
        baselines["constant_motion"].add(constant_motion(old), y, old, valid)
        probabilities.append(output["terminal"][:, 0].sigmoid().cpu().numpy())
        truths.append(terminal)
    probability, truth = np.concatenate(probabilities), np.concatenate(truths)
    if threshold is None:
        if data.split != "validation":
            raise ValueError("Threshold selection requires validation, never test data")
        threshold = choose_threshold(probability, truth)
    result = {
        "split": data.split,
        "samples": len(indices),
        "threshold": threshold,
        "evaluation_identity": hashlib.sha256(indices.tobytes()).hexdigest(),
        "one_step": {k: v.result() for k, v in scores.items()},
        "baselines": {k: v.result() for k, v in baselines.items()},
        "terminal": terminal_metrics(probability, truth, threshold),
        "one_step_examples_per_second": len(indices) / max(time.perf_counter() - begin, 1e-9),
    }
    horizons = ec["horizons"]
    selected = data.select(ec["starts"] if starts is None else starts, ec["seed"] + 1)
    rollout_scores = {h: StateScores(include_velocity) for h in horizons}
    reliable_steps, available_steps, early, missed = [], [], 0, 0
    for ii in batches(selected, min(tc["batch_size"], 128)):
        batch = data.batch(ii, history, past, max(horizons), device)
        output, states = approach.rollout(batch)
        p, y = states.cpu().numpy(), batch["target"].cpu().numpy()
        if not np.isfinite(p).all():
            raise FloatingPointError("Nonfinite recursive prediction")
        valid, terminal = batch["valid"].cpu().numpy(), batch["terminal"].cpu().numpy()
        live = valid & ~terminal
        old = np.concatenate((batch["history"][:, -1:].cpu().numpy(), y[:, :-1]), 1)
        for h in horizons:
            rollout_scores[h].add(p[:, h - 1], y[:, h - 1], old[:, h - 1], live[:, h - 1])
        predicted_done = output["terminal"].sigmoid().cpu().numpy() >= threshold
        stopped = np.maximum.accumulate(predicted_done, axis=1)
        early += int((stopped & live).any(1).sum())
        missed += int((terminal & valid & ~stopped).any(1).sum())
        error = np.abs(p[..., :6] - y[..., :6]) * NATIVE_SCALES
        bad = live & (
            (error[..., :2].max(-1) > ec["divergence_ball_pixels"])
            | (error[..., 4] > ec["divergence_paddle_pixels"])
            | np.any((p[..., 7:] >= 0.5) != (y[..., 7:] >= 0.5), axis=-1)
        )
        bad |= valid & (predicted_done != terminal)
        failure_seen = np.maximum.accumulate(bad, axis=1)
        reliable_steps.extend((valid & ~failure_seen).sum(1).tolist())
        available_steps.extend(valid.sum(1).tolist())
    result["rollout"] = {str(h): score.result() for h, score in rollout_scores.items()}
    result["rollout_identity"] = hashlib.sha256(selected.tobytes()).hexdigest()
    result["rollout_summary"] = {
        "starts": len(selected),
        "mean_reliable_steps": float(np.mean(reliable_steps)),
        "mean_reference_steps": float(np.mean(available_steps)),
        "early_terminal_fraction": early / len(selected),
        "missed_terminal_rollouts": missed,
        "state_errors": "Uninterrupted diagnostic predictions; never hide errors after early stops",
    }
    # Fixed score shared by every candidate. Separate raw diagnostics remain authoritative.
    horizon_errors = []
    for score in result["rollout"].values():
        if score["samples"]:
            mae = list(score["mae_native"].values())
            horizon_errors.append(
                min(100, (mae[0] + mae[1]) / 2)
                + min(50, mae[4])
                + 20 * (1 - score["brick_cell_accuracy"])
            )
    result["selection_score"] = (
        float(np.mean(horizon_errors))
        + 20 * result["rollout_summary"]["early_terminal_fraction"]
        + 10 * (1 - result["terminal"]["f1"])
    )
    return result


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("rules") != RULES:
        raise ValueError("Not a compatible state-only dynamics checkpoint")
    config = payload["config"]
    if config["model"]["kind"] not in ("state_mlp", "state_gru"):
        raise ValueError("Checkpoint requests a non-state model")
    approach = StateApproach(config["model"], config["loss"]).to(device)
    approach.load_state_dict(payload["model"])
    approach.eval()
    return approach, payload


def train(config, *, initialize=None):
    config = copy.deepcopy(config)
    folder = Path(config["output"])
    if folder.exists():
        raise ValueError(f"Output directory already exists: {folder}")
    folder.mkdir(parents=True)
    tc = config["trainer"]
    torch.set_num_threads(tc["threads"])
    torch.manual_seed(tc["seed"])
    np.random.seed(tc["seed"])
    device = torch.device(tc["device"])
    train_data, validation = (
        StateWindows(config["cache"], "train"),
        StateWindows(config["cache"], "validation"),
    )
    metadata = train_data.manifest
    positive = metadata["train"]["terminals"]
    negative = metadata["train"]["transitions"] - positive
    if not positive:
        raise ValueError("Training split has no loss boundaries")
    approach = StateApproach(config["model"], config["loss"], min(100, negative / positive)).to(
        device
    )
    if initialize:
        previous, payload = load_checkpoint(initialize, device)
        if (
            payload["cache_identity"] != metadata["identity"]
            or payload["config"]["model"] != config["model"]
        ):
            raise ValueError("Fine-tuning requires the same state cache and model contract")
        approach.load_state_dict(previous.state_dict())
    OmegaConf.save(OmegaConf.create(config), folder / "resolved.yaml")
    write_json(folder / "data-manifest.json", metadata)
    optimizer = torch.optim.AdamW(
        approach.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"]
    )
    stages = [(1, tc["epochs"])] + [(h, tc["rollout_epochs"]) for h in tc["rollout_horizons"]]
    best_score, best_path, epoch, updates = float("inf"), None, 0, 0
    started = time.perf_counter()
    training_examples = 0
    try:
        for horizon, epochs in stages:
            if epochs <= 0:
                continue
            if best_path:
                selected, payload = load_checkpoint(best_path, device)
                approach.load_state_dict(selected.state_dict())
                optimizer.load_state_dict(payload["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = tc["learning_rate"] * (0.5 if horizon > 1 else 1)
            for local_epoch in range(epochs):
                epoch += 1
                approach.train()
                indices = train_data.select(tc["train_samples"], tc["seed"] + epoch)
                np.random.default_rng(tc["seed"] + epoch).shuffle(indices)
                epoch_started = time.perf_counter()
                sums, seen = {}, 0
                for ii in batches(indices, tc["batch_size"]):
                    batch = train_data.batch(
                        ii,
                        config["model"]["history"],
                        config["model"]["past_actions"],
                        horizon,
                        device,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss, terms = approach.loss(batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite dynamics loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        approach.parameters(), tc["gradient_clip"], error_if_nonfinite=True
                    )
                    optimizer.step()
                    seen += len(ii)
                    updates += 1
                    for key, value in {"total": loss, **terms}.items():
                        sums[key] = sums.get(key, 0) + value.detach().item() * len(ii)
                training_examples += seen
                if epoch % tc["eval_every"] and local_epoch != epochs - 1:
                    continue
                duration = time.perf_counter() - epoch_started
                metrics = evaluate(
                    approach,
                    validation,
                    config,
                    samples=tc["validation_samples"],
                    starts=tc["rollout_starts"],
                )
                row = {
                    "epoch": epoch,
                    "horizon": horizon,
                    "optimizer_steps": updates,
                    "loss": {k: v / seen for k, v in sums.items()},
                    "training_examples_per_second": seen / duration,
                    "validation": metrics,
                }
                with (folder / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                payload = {
                    "format": CHECKPOINT_FORMAT,
                    "rules": RULES,
                    "config": config,
                    "cache_identity": metadata["identity"],
                    "model": approach.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "optimizer_steps": updates,
                    "validation": metrics,
                    "parent_checkpoint": str(initialize) if initialize else None,
                }
                checkpoint = folder / f"epoch-{epoch:03d}.pt"
                torch.save(payload, checkpoint)
                if metrics["selection_score"] < best_score:
                    best_score, best_path = metrics["selection_score"], checkpoint
                    torch.save(payload, folder / "best.pt")
                print(
                    json.dumps(
                        {
                            "run": folder.name,
                            "epoch": epoch,
                            "horizon": horizon,
                            "loss": row["loss"]["total"],
                            "score": metrics["selection_score"],
                            "ball_mae": list(metrics["one_step"]["all"]["mae_native"].values())[:2],
                            "terminal_f1": metrics["terminal"]["f1"],
                        }
                    ),
                    flush=True,
                )
        if best_path is None:
            raise ValueError("No training epochs configured")
        _, best = load_checkpoint(folder / "best.pt")
        result = {
            "status": "complete",
            "best_checkpoint": str((folder / "best.pt").resolve()),
            "best_epoch": best["epoch"],
            "selection_score": best_score,
            "validation": best["validation"],
            "optimizer_steps": updates,
            "parameters": sum(p.numel() for p in approach.parameters()),
            "training_examples": training_examples,
            "duration_seconds": time.perf_counter() - started,
            "cache_identity": metadata["identity"],
            "test_evaluated": False,
        }
        write_json(folder / "summary.json", result)
        return result
    except Exception as error:
        write_json(
            folder / "summary.json",
            {"status": "failed", "error": str(error), "optimizer_steps": updates},
        )
        raise


def evaluate_checkpoint(checkpoint, cache, config, output):
    approach, payload = load_checkpoint(checkpoint, config["trainer"]["device"])
    data = StateWindows(cache, config["evaluation"]["split"])
    if payload["cache_identity"] != data.manifest["identity"]:
        raise ValueError("Checkpoint and evaluation cache identities disagree")
    ec = copy.deepcopy(config)
    ec["model"] = payload["config"]["model"]
    threshold = config["evaluation"]["threshold"]
    if threshold is None:
        threshold = payload["validation"]["threshold"]
    result = evaluate(approach, data, ec, threshold=threshold)
    result["checkpoint"] = str(Path(checkpoint).resolve())
    result["cache_identity"] = payload["cache_identity"]
    write_json(output, result)
    return result


def playback(checkpoint, cache, config, output):
    approach, payload = load_checkpoint(checkpoint, config["trainer"]["device"])
    pc, model = config["playback"], payload["config"]["model"]
    data = StateWindows(cache, pc["split"])
    if payload["cache_identity"] != data.manifest["identity"]:
        raise ValueError("Checkpoint and scene cache identities disagree")
    if not 0 <= pc["start_index"] < len(data.indices):
        raise ValueError("Invalid playback start index")
    index = int(data.indices[pc["start_index"]])
    device = next(approach.parameters()).device
    batch = data.batch([index], model["history"], model["past_actions"], pc["steps"], device)
    player = StatePlayer(
        approach,
        batch["history"],
        batch["history_valid"],
        batch["action_history"][:, :-1],
        payload["validation"]["threshold"],
    )
    rows = []
    for step in range(pc["steps"]):
        if not batch["valid"][0, step]:
            break
        action = int(batch["actions"][0, step])
        row = player.step(action)
        row.update(
            {
                "step": step + 1,
                "requested_action": action,
                "reference_terminal": bool(batch["terminal"][0, step]),
                "reference": batch["target"][0, step].cpu().tolist()
                if not batch["terminal"][0, step]
                else None,
            }
        )
        rows.append(row)
        if player.terminal:
            break
    result = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "split": data.split,
        "episode_id": int(data.arrays["episode_ids"][index]),
        "source_step": int(data.arrays["steps"][index]),
        "initial_state": batch["history"][0, -1].cpu().tolist(),
        "terminal": player.terminal,
        "steps": rows,
        "stop_reason": "predicted_terminal" if player.terminal else "reference_or_step_limit",
    }
    write_json(output, result)
    return result


def search(config):
    """Screen contexts, then compare capacity/memory; test only the selected finalist."""
    root = Path(config["output"])
    if root.exists():
        raise ValueError(f"Search directory already exists: {root}")
    root.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create(config), root / "resolved.yaml")
    sc = config["search"]
    results = []

    def candidate(name, model, seed, *, full=False):
        cfg = copy.deepcopy(config)
        cfg["output"] = str(root / name)
        cfg["model"].update(model)
        cfg["trainer"]["seed"] = seed
        if not full:
            cfg["trainer"].update(epochs=sc["screening_epochs"], rollout_epochs=0)
        summary = train(cfg)
        row = {
            "name": name,
            "model": cfg["model"],
            "seed": seed,
            "score": summary["selection_score"],
            "checkpoint": summary["best_checkpoint"],
            "parameters": summary["parameters"],
            "seconds": summary["duration_seconds"],
        }
        results.append(row)
        write_json(root / "candidates.json", results)
        return row

    contexts = []
    for h in sc["histories"]:
        for k in sc["past_actions"]:
            rows = [
                candidate(f"screen-h{h}-a{k}-s{seed}", {"history": h, "past_actions": k}, seed)
                for seed in sc["seeds"]
            ]
            contexts.append((float(np.mean([r["score"] for r in rows])), h, k))
    selected = sorted(contexts)[: sc["finalists"]]
    finalists = []
    for _, h, k in selected:
        for kind in ["state_mlp", "state_gru"] if sc["recurrent"] else ["state_mlp"]:
            for width in sc["widths"]:
                rows = [
                    candidate(
                        f"final-{kind}-h{h}-a{k}-w{width}-s{seed}",
                        {"kind": kind, "history": h, "past_actions": k, "width": width},
                        seed,
                        full=True,
                    )
                    for seed in sc["seeds"]
                ]
                finalists.append(
                    {
                        "mean_score": float(np.mean([r["score"] for r in rows])),
                        "std_score": float(np.std([r["score"] for r in rows])),
                        "runs": rows,
                    }
                )
    winner = min(finalists, key=lambda x: (x["mean_score"], x["runs"][0]["parameters"]))
    chosen = min(winner["runs"], key=lambda x: x["score"])
    test_config = copy.deepcopy(config)
    test_config["evaluation"]["split"] = "test"
    final_test = evaluate_checkpoint(
        chosen["checkpoint"], config["cache"], test_config, root / "test.json"
    )
    result = {
        "status": "complete",
        "winner": winner,
        "selected_checkpoint": chosen["checkpoint"],
        "finalists": finalists,
        "test": final_test,
        "claim": "Best tested configuration within the saved data and compute budget",
    }
    write_json(root / "summary.json", result)
    return result
