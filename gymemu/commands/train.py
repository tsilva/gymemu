"""Train a configured emulator approach. Use --multirun for Hydra sweeps."""

import sys

import hydra

# Preserve imports used by existing scripts and version-1 checkpoint consumers.
from gymemu.checkpoints import load_model, save_model  # noqa: F401
from gymemu.config import CONFIG_DIR
from gymemu.data import Frames, Windows, frame_stack, read_episodes  # noqa: F401
from gymemu.engine import train
from gymemu.legacy import main as legacy_main
from gymemu.models.direct import Autoencoder  # noqa: F401
from gymemu.recipes import recipe_arguments
from gymemu.runtime import device_for, positive  # noqa: F401


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="config")
def hydra_main(cfg):
    train(cfg)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(arg == "--resume" or arg.startswith("--resume=") for arg in arguments):
        import argparse

        from gymemu.training_state import resume_config

        parser = argparse.ArgumentParser(description="Continue a saved training run")
        parser.add_argument("--resume", required=True)
        selected, overrides = parser.parse_known_args(arguments)
        try:
            cfg = resume_config(selected.resume, overrides)
        except ValueError as error:
            parser.error(str(error))
        return train(cfg)
    # Keep existing --dataset/--output commands working. New runs use key=value overrides.
    legacy_flags = {
        "--dataset",
        "--revision",
        "--output",
        "--history",
        "--width",
        "--epochs",
        "--batch-size",
        "--learning-rate",
        "--device",
        "--threads",
        "--seed",
        "--workers",
        "--precision",
        "--checkpoint-seconds",
        "--train-split",
        "--eval-split",
        "--limit-episodes",
        "--train-batches",
        "--eval-batches",
        "--env-id",
        "--wandb-mode",
        "--r2",
        "--no-r2",
    }
    if any(arg.split("=", 1)[0] in legacy_flags for arg in arguments):
        return legacy_main(arguments)
    try:
        arguments = recipe_arguments(arguments)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    original = sys.argv
    try:
        sys.argv = [original[0], *arguments]
        return hydra_main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
