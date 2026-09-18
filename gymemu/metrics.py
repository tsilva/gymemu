"""Metric schema v3: short semantic paths, explicit axes, latest-value summaries.

Each measurement has one canonical path; legacy aliases are no longer emitted.
"""

import re

SCHEMA_VERSION = 3
# Template: (display label, unit, cadence). Dimensions are bounded path segments.
METRICS = {
    "train/step": ("Optimizer step", "updates", "log"),
    "eval/step": ("Evaluated step", "updates", "epoch"),
    "train/samples": ("Training samples", "samples", "log"),
    "train/objective": ("Training objective", "name", "epoch"),
    "eval/rate": ("Validation throughput", "samples/s", "epoch"),
    "train/{stage}/{metric}/epoch": ("{stage} training {metric}", "metric", "epoch"),
    "eval/{stage}/{metric}": ("{stage} validation {metric}", "metric", "epoch"),
    "train/{stage}/curriculum/{metric}": ("{stage} curriculum {metric}", "metric", "epoch"),
    "train/stage": ("Stage", "name", "log"),
    "train/epoch": ("Epoch", "epochs", "log"),
    "train/lr": ("Learning rate", "rate", "log"),
    "train/horizon": ("Training horizon", "frames", "log"),
    "train/rate": ("Throughput", "samples/s", "epoch"),
    "train/loss/mean": ("Window loss", "objective", "log"),
    "train/loss/max": ("Peak batch loss", "objective", "log"),
    "train/{stage}/loss/epoch": ("{stage} training loss", "objective", "epoch"),
    "eval/{stage}/loss": ("{stage} validation loss", "objective", "epoch"),
    "eval/mse": ("Held-out RGB MSE", "MSE", "epoch"),
    "train/grad/peak/step": ("Peak gradient step", "updates", "log"),
    "train/rgb/mse": ("Training RGB error", "MSE", "log"),
    "train/ball/mse": ("Training ball penalty", "MSE", "log"),
    "train/ball/coverage": ("Training ball coverage", "fraction", "log"),
    "train/grad/norm/max": ("Peak gradient norm", "L2", "every update → log"),
    "train/grad/norm/mean": ("Mean gradient norm", "L2", "every update → log"),
    "train/grad/norm/nonfinite/fraction": (
        "Nonfinite gradient norms",
        "fraction",
        "every update → log",
    ),
    "train/grad/weights/norm/min": ("Minimum weight gradient", "L2", "every update → log"),
    "train/grad/weights/zero/fraction": ("Dead weight updates", "fraction", "every update → log"),
    "train/grad/{layer}/norm": ("{layer} gradient", "L2", "sample"),
    "train/grad/{layer}/missing": ("{layer} missing gradient", "flag", "sample"),
    "train/grad/{layer}/nonzero/fraction": ("{layer} active gradients", "fraction", "sample"),
    "train/grad/{layer}/nonfinite/fraction": ("{layer} nonfinite gradients", "fraction", "sample"),
    "train/weight/{layer}/norm": ("{layer} weights", "L2", "sample"),
    "train/update/{layer}/ratio": ("{layer} relative update", "ratio", "sample"),
    "train/act/{layer}/abs/max": ("{layer} peak logits", "absolute", "sample"),
    "train/act/{layer}/sat/fraction": ("{layer} sigmoid saturation", "fraction", "sample"),
    "train/act/{layer}/nonfinite/fraction": ("{layer} nonfinite logits", "fraction", "sample"),
    "train/relu/{layer}/abs/max": ("{layer} peak activation", "absolute", "sample"),
    "train/relu/{layer}/zero/fraction": ("{layer} zero activations", "fraction", "sample"),
    "train/relu/{layer}/nonfinite/fraction": (
        "{layer} nonfinite activations",
        "fraction",
        "sample",
    ),
    "probe/mse/h{horizon}": ("RGB MSE at step {horizon}", "MSE", "probe"),
    "probe/count/h{horizon}": ("Targets at step {horizon}", "frames", "probe"),
    "probe/mse/mean": ("Rollout RGB MSE", "MSE", "probe"),
    "probe/teacher/mse": ("Clean-history RGB MSE", "MSE", "probe"),
    "probe/early/mse": ("Early-start RGB MSE", "MSE", "probe"),
    "probe/regular/mse": ("Regular-start RGB MSE", "MSE", "probe"),
    "probe/drift/ratio": ("Rollout / clean error", "ratio", "probe"),
    "probe/fg/mse": ("Foreground MSE", "MSE", "probe"),
    "probe/ball/mse": ("Visible-ball region MSE", "MSE", "probe"),
    "probe/ball/mse/h{horizon}": ("Ball MSE at step {horizon}", "MSE", "probe"),
    "probe/ball/coverage": ("Ball detector coverage", "fraction", "probe"),
    "probe/ball/count": ("Detected ball targets", "frames", "probe"),
    "probe/ball/reappear/mse": ("Ball reappearance MSE", "MSE", "probe"),
    "probe/ball/reappear/count": ("Ball reappearances", "frames", "probe"),
    "probe/pixel/nonfinite/fraction": ("Nonfinite predictions", "fraction", "probe"),
    "probe/pixel/saturated/fraction": ("Saturated predictions", "fraction", "probe"),
    "probe/pixel/wrong_sat/fraction": ("Incorrect saturated pixels", "fraction", "probe"),
    "probe/pixel/std": ("Spatial RGB variation", "std", "probe"),
    "probe/motion/mse": ("Motion error", "MSE", "probe"),
    "probe/motion/pred": ("Predicted motion", "absolute change", "probe"),
    "probe/motion/target": ("Recorded motion", "absolute change", "probe"),
    "probe/frames": ("Fixed rollouts", "image", "probe"),
}


def pattern(template):
    value = re.escape(template)
    for dimension in ("stage", "layer", "metric"):
        value = value.replace(re.escape("{" + dimension + "}"), r"[A-Za-z0-9_.-]+")
    return "^" + value.replace(re.escape("{horizon}"), r"[1-9][0-9]*") + "$"


PATTERNS = [(re.compile(pattern(template)), definition) for template, definition in METRICS.items()]


def validate_metrics(metrics):
    for name in metrics:
        if not any(regex.fullmatch(name) for regex, _ in PATTERNS):
            raise ValueError(f"Unregistered metric: {name}")
