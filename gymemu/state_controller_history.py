"""Source-time inputs for isolated controller-state inference diagnostics."""

import numpy as np
import torch


def controller_history_inputs(context, indices, length):
    """H observations ending now and H preceding actions, clipped at life boundaries.

    The context mapping contains arrays from RecordedPaddleContext. Current and
    future actions, future observations and reconstructed hidden labels are absent.
    """
    if length < 1:
        raise ValueError("History length must be positive")
    source = context["source_rows"][np.asarray(indices)]
    rows = source[:, None] - np.arange(length - 1, -1, -1)
    lower = context["starts"][source, None]
    safe = np.maximum(rows, lower)
    valid = (rows >= lower) & context["valid"][safe]
    observed = context["states"][safe].copy()
    observed[~valid] = 0
    observed = np.rint(observed * np.array([160, 160, 16], np.float32))
    preceding = rows - 1
    actions = context["actions"][np.maximum(preceding, lower)].copy()
    actions[preceding < lower] = 3
    inputs = np.concatenate([observed, valid[..., None], actions[..., None]], axis=-1)
    complete = valid.all(1) & (preceding[:, 0] >= lower[:, 0])
    return torch.from_numpy(inputs.astype(np.float32)), complete
