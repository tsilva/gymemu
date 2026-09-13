"""Explicit optimizer registry; YAML selects settings, never arbitrary Python targets."""

import math

import torch

OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}


def validate_optimizer(spec):
    if spec["kind"] not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {spec['kind']!r}; choose {sorted(OPTIMIZERS)}")
    betas = spec["betas"]
    if len(betas) != 2 or any(type(b) not in (float, int) or not 0 <= b < 1 for b in betas):
        raise ValueError("optimizer.betas must contain two numbers in [0, 1)")
    for key in ("eps", "weight_decay"):
        value = spec[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f"optimizer.{key} must be finite and nonnegative")
    if type(spec["amsgrad"]) is not bool:
        raise ValueError("optimizer.amsgrad must be a boolean")


def build_optimizer(parameters, spec, learning_rate):
    validate_optimizer(spec)
    return OPTIMIZERS[spec["kind"]](
        parameters,
        lr=learning_rate,
        betas=tuple(spec["betas"]),
        eps=spec["eps"],
        weight_decay=spec["weight_decay"],
        amsgrad=spec["amsgrad"],
    )
