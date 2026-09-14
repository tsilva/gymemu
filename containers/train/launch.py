"""Common command interface for Docker, Runpod and dstack."""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(arguments):
    """Prepare the configured cache once, then execute the ordinary training CLI."""
    import torch

    from gymemu.config import compose_config
    from gymemu.data import Frames, resolve_dataset

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--recipe")
    selected, overrides = parser.parse_known_args(arguments)
    cfg = compose_config(overrides, recipe=selected.recipe)
    if cfg.trainer.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; check GPU allocation and the host NVIDIA driver"
            )
        if cfg.trainer.precision == "bf16" and not torch.cuda.is_bf16_supported(
            including_emulation=False
        ):
            raise RuntimeError("This recipe requires a GPU with native bfloat16 support")
    if cfg.trainer.frame_cache:
        from gymemu.cache import build_cache

        root, _ = resolve_dataset(cfg.game.dataset, cfg.game.revision)
        cache = Path(cfg.trainer.frame_cache).expanduser()
        if not cache.exists():
            build_cache(root, cache, workers=max(1, cfg.trainer.workers))
        # Reuse requires a valid, complete cache tied to the exact source frames.
        Frames(root, compact=True, cache=cache)
    execute("train", arguments)


def execute(command, arguments):
    scripts = {
        "train": "train.py",
        "cache": "cache_frames.py",
        "play": "play.py",
        "upload": "upload_checkpoints.py",
        "smoke": "containers/train/smoke.py",
    }
    if command not in scripts:
        raise SystemExit("Usage: gymemu-container {run|train|cache|play|upload|smoke} [arguments]")
    os.execv(sys.executable, [sys.executable, str(ROOT / scripts[command]), *arguments])


def main():
    args = sys.argv[1:] or ["smoke", "--device", "cpu"]
    command, arguments = args[0], args[1:]
    if command == "run":
        # Give every invocation its own output, including when several jobs start together.
        if not any(arg.startswith("output=") for arg in arguments):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            arguments.append(f"output={Path.cwd() / 'runs' / stamp}")
        run(arguments)
    else:
        execute(command, arguments)


if __name__ == "__main__":
    main()
