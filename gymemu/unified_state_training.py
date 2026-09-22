"""Supervised labels and masked objectives for the shared state MLP experiment."""

import numpy as np
import torch
from torch.nn import functional as F

from gymemu.models.unified_state import UnifiedStateMLP


def transition_values(source, target, terminal):
    """Read state labels only on continuing rows; terminal successors may be NaN."""
    n = len(source)
    if (
        source.shape != (n, 119)
        or target.shape != (n, 117)
        or terminal.shape != (n,)
        or terminal.dtype != np.bool_
        or not np.isfinite(target[~terminal]).all()
    ):
        raise ValueError("Expected aligned current state, masked next state and boolean stop")
    current, following = source[~terminal], target[~terminal]
    values = dict(
        dx=following[:, 0] - current[:, 0],
        dy=following[:, 1] - current[:, 1] - current[:, 8] / 8,
        vx=following[:, 2],
        vy=following[:, 3],
        charge=following[:, 115] - current[:, 6],
        paddle_x=following[:, 116] - current[:, 4],
    )
    bricks, next_bricks = current[:, 10:118], following[:, 4:112]
    removed = (bricks > 0) & (next_bricks == 0)
    if (
        not np.isin(bricks, [0, 1]).all()
        or not np.isin(next_bricks, [0, 1]).all()
        or (next_bricks > bricks).any()
        or (removed.sum(1) > 1).any()
    ):
        raise ValueError("Brick labels require no change or one occupied-cell removal")
    values.update(
        bricks=np.where(removed.any(1), removed.argmax(1) + 1, 0),
        contact=following[:, 112],
        count=following[:, 113] - current[:, 7],
        width=(following[:, 114] - 12) / 4,
    )
    for name in ("contact", "count", "width"):
        if not np.isin(values[name], [0, 1]).all():
            raise ValueError(f"Expected binary {name} target")
    return values


def fit_vocabulary(source, target, terminal):
    """Call on training episodes only; vocabulary is checkpoint metadata."""
    values = transition_values(source, target, terminal)
    return {name: np.unique(values[name]).tolist() for name in UnifiedStateMLP.vocabulary_names}


def encode_targets(source, target, terminal, vocabulary):
    values = transition_values(source, target, terminal)
    labels = np.full((len(source), len(UnifiedStateMLP.names)), -100, dtype=np.int64)
    for column, name in enumerate(UnifiedStateMLP.names):
        if name == "terminated":
            labels[:, column] = terminal
            continue
        value = values[name]
        if name in vocabulary:
            classes = np.asarray(vocabulary[name])
            index = np.searchsorted(classes, value)
            if (index >= len(classes)).any() or not np.array_equal(classes[index], value):
                raise ValueError(f"Target outside training vocabulary: {name}")
            value = index
        labels[~terminal, column] = value.astype(np.int64)
    return labels


def classification_losses(logits, labels):
    """Equal task weights; each state objective averages only continuing rows."""
    return {
        name: F.cross_entropy(logits[name], labels[:, j], ignore_index=-100, reduction="sum")
        / (labels[:, j] != -100).sum().clamp(min=1)
        for j, name in enumerate(UnifiedStateMLP.names)
    }


def joint_loss(logits, labels):
    return torch.stack(list(classification_losses(logits, labels).values())).mean()
