"""Event-stratified diagnostics for isolated state predictors, never model inputs."""

import hashlib

import numpy as np
import torch

from gymemu.state_data import FIELDS, NATIVE_SCALES
from gymemu.state_training import batches, constant_motion, terminal_metrics


def event_masks(previous, target, terminal):
    """Infer overlapping events from recorded transitions, not native event flags.

    Wall/paddle categories are geometric proxies. Terminal successors are undefined
    and excluded from all state-event categories. Free flight means unchanged
    velocity and bricks; two cancelling native collisions could escape this proxy.
    """
    live = ~terminal
    velocity = np.any(np.abs(target[:, 2:4] - previous[:, 2:4]) > 1e-5, axis=1)
    bricks = np.any(target[:, 7:] != previous[:, 7:], axis=1)
    upward_bounce = (previous[:, 3] > 0) & (target[:, 3] < 0)
    paddle = velocity & upward_bounce & (previous[:, 1] * 255 > 160) & ~bricks
    wall = (
        velocity
        & ~bricks
        & ~paddle
        & ((previous[:, 0] * 160 < 15) | (previous[:, 0] * 160 > 144) | (previous[:, 1] * 255 < 35))
    )
    return {
        "all_live": live,
        "free_flight_proxy": live & ~velocity & ~bricks,
        "velocity_change": live & velocity,
        "brick_change": live & bricks,
        "paddle_collision_proxy": live & paddle,
        "paddle_non_breakthrough": live & paddle & (np.abs(previous[:, 3]) < 0.999),
        "wall_collision_proxy": live & wall,
        "other_velocity_change": live & velocity & ~bricks & ~paddle & ~wall,
        "paddle_reversal": live & (previous[:, 5] * target[:, 5] < -1e-8),
        "paddle_acceleration": live & (np.abs(previous[:, 5] - target[:, 5]) > 1e-5),
        "width_change": live & (np.abs(previous[:, 6] - target[:, 6]) > 1e-5),
        "terminal": terminal,
    }


def sample_training_indices(data, eligible, count, seed, sampling):
    """Uniform or a 50/50 event/ordinary draw, using training labels only."""
    if data.split != "train":
        raise ValueError("Sampling is restricted to training trajectories")
    rng = np.random.default_rng(seed)
    count = min(count, len(eligible))
    if sampling == "uniform":
        return rng.choice(eligible, count, replace=False)
    if sampling != "balanced_events":
        raise ValueError("Unknown sampling strategy")
    a = data.arrays
    masks = event_masks(a["states"][eligible], a["states"][eligible + 1], a["terminal"][eligible])
    event = masks["velocity_change"] | masks["brick_change"] | masks["terminal"]
    if not event.any() or event.all():
        raise ValueError("Balanced sampling requires both event and ordinary transitions")
    n = count // 2
    chosen = np.r_[
        rng.choice(eligible[event], n, replace=n > event.sum()),
        rng.choice(eligible[~event], count - n, replace=count - n > (~event).sum()),
    ]
    return rng.permutation(chosen)


def error_metrics(error):
    if not len(error):
        return {"samples": 0}
    return {
        "samples": len(error),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "p95": float(np.quantile(np.abs(error), 0.95)),
        "within_one": float((np.abs(error) <= 1).mean()),
    }


@torch.no_grad()
def diagnose_target(model, data, config, indices=None, threshold=None):
    """Evaluate unchanged target identities and save concrete largest-error witnesses."""
    if data.split == "test":
        raise ValueError("Development diagnostics must not reuse the reserved test")
    model.eval()
    indices = np.array(data.indices if indices is None else indices)
    predictions = []
    mc = config["model"]
    for ii in batches(indices, 2048):
        batch = data.batch(ii, mc["history"], mc["past_actions"])
        if hasattr(data, "paddle_context"):
            data.paddle_context.attach(ii, batch)
        if hasattr(data, "hidden_inputs"):
            data.hidden_inputs.attach(ii, batch)
        predictions.append(model(batch).numpy())
    prediction = np.concatenate(predictions)
    a = data.arrays
    old, y, terminal = a["states"][indices], a["states"][indices + 1], a["terminal"][indices]
    masks = event_masks(old, y, terminal)
    if hasattr(data, "hidden_inputs"):
        counts = np.rint(data.hidden_inputs.values[indices, 4] * 12)
        masks["paddle_speed_threshold"] = masks["paddle_non_breakthrough"] & np.isin(
            counts, [3, 7, 11]
        )
    masks["first_eight_segment_steps"] = (~terminal) & (indices - a["starts"][indices] < 8)
    result = {
        "split": data.split,
        "cache_identity": data.manifest["identity"],
        "evaluation_identity": hashlib.sha256(indices.tobytes()).hexdigest(),
        "event_labels": "overlapping proxies inferred from successor labels, not predictor inputs",
        "counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "target": model.target,
    }
    if model.target in FIELDS[:6]:
        k = FIELDS.index(model.target)
        error = (prediction - y[:, k]) * NATIVE_SCALES[k]
        baseline = (constant_motion(old)[:, k] - y[:, k]) * NATIVE_SCALES[k]
        result["events"] = {
            name: {
                "model": error_metrics(error[mask]),
                "constant_motion": error_metrics(baseline[mask]),
            }
            for name, mask in masks.items()
            if name != "terminal"
        }
        live = np.flatnonzero(~terminal)
        top = live[np.argsort(np.abs(error[live]))[-20:][::-1]]
        result["worst_examples"] = [
            {
                "episode": int(a["episode_ids"][indices[j]]),
                "step": int(a["steps"][indices[j]]),
                "current_native": (old[j, :6] * NATIVE_SCALES).tolist(),
                "next_native": (y[j, :6] * NATIVE_SCALES).tolist(),
                "prediction_native": float(prediction[j] * NATIVE_SCALES[k]),
                "error_native": float(error[j]),
                "events": [name for name, mask in masks.items() if mask[j]],
            }
            for j in top
        ]
    elif model.target == "terminal":
        if threshold is None:
            raise ValueError("Pass the checkpoint's validation-selected threshold")
        result["terminal"] = terminal_metrics(
            torch.from_numpy(prediction).sigmoid().numpy(), terminal, threshold
        )
    elif model.target == "brick_grid":
        result["events"] = {}
        for name, mask in masks.items():
            if name == "terminal":
                continue
            before, after, pred = old[mask, 7:] >= 0.5, y[mask, 7:] >= 0.5, prediction[mask] >= 0
            true, guessed = before & ~after, before & ~pred
            result["events"][name] = terminal_metrics(
                guessed.astype(float).ravel(), true.ravel(), 0.5
            )
    else:
        live = ~terminal
        change = masks["width_change"]
        correct = (prediction >= 0) == (y[:, 6] < 0.875)
        result["accuracy"] = float(correct[live].mean())
        result["change_accuracy"] = float(correct[change].mean()) if change.any() else None
    return result
