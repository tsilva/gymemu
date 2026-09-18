"""Compatibility for the original argparse training commands."""

import argparse
from pathlib import Path

from gymemu.config import compose_config
from gymemu.data import DEFAULT_DATASET, DEFAULT_HISTORY
from gymemu.engine import train
from gymemu.runtime import positive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET, help="Hub ID or local snapshot folder"
    )
    parser.add_argument("--revision", help="Hub revision; the default dataset uses a pinned commit")
    parser.add_argument("--output", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--history", type=positive, default=DEFAULT_HISTORY)
    parser.add_argument("--width", type=positive, default=32)
    parser.add_argument("--epochs", type=positive, default=10)
    parser.add_argument("--batch-size", type=positive, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--threads", type=positive, help="Optional PyTorch CPU thread limit")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--workers", type=int, default=0, help="Parallel image-loader processes")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--checkpoint-seconds",
        type=positive,
        default=600,
        help="Interval for atomic latest.pt inference snapshots",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="heldout")
    parser.add_argument(
        "--limit-episodes", type=positive, help="First N episodes per split for smokes"
    )
    parser.add_argument("--train-batches", type=positive, help="Optional per-epoch smoke limit")
    parser.add_argument("--eval-batches", type=positive, help="Optional evaluation smoke limit")
    parser.add_argument("--env-id", help="Canonical environment ID for W&B project naming")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    parser.add_argument("--r2", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    cfg = compose_config([] if args.dataset == DEFAULT_DATASET else ["game=custom"])
    cfg.output = str(args.output)
    cfg.game.dataset = args.dataset
    cfg.game.revision = args.revision
    if args.env_id is not None:
        cfg.game.env_id = args.env_id
    if args.wandb_mode is not None:
        cfg.wandb.mode = args.wandb_mode
    if args.r2 is not None:
        cfg.r2.enabled = args.r2
    if args.dataset != DEFAULT_DATASET:
        cfg.game.name = "custom"
        cfg.game.key_actions = {}
    cfg.history = args.history
    cfg.model.width = args.width
    cfg.seed = args.seed
    cfg.game.train_split = args.train_split
    cfg.game.eval_split = args.eval_split
    for name in [
        "epochs",
        "batch_size",
        "learning_rate",
        "device",
        "threads",
        "workers",
        "precision",
        "checkpoint_seconds",
        "limit_episodes",
        "train_batches",
        "eval_batches",
    ]:
        cfg.trainer[name] = getattr(args, name)
    try:
        return train(cfg)
    except ValueError as error:
        if "Output directory" in str(error) or "must" in str(error):
            parser.error(str(error))
        raise
