"""Train a configured emulator approach. Use --multirun for Hydra sweeps."""

import sys

import hydra

# Preserve imports used by existing scripts and version-1 checkpoint consumers.
from gymemu.checkpoints import load_model, save_model  # noqa: F401
from gymemu.data import Frames, Windows, frame_stack, read_episodes  # noqa: F401
from gymemu.engine import train
from gymemu.legacy import main as legacy_main
from gymemu.models.direct import Autoencoder  # noqa: F401
from gymemu.runtime import device_for, positive  # noqa: F401


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def hydra_main(cfg):
    train(cfg)


def main(argv=None):
    if argv is not None:
        return legacy_main(argv)
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
    }
    if any(arg.split("=", 1)[0] in legacy_flags for arg in sys.argv[1:]):
        return legacy_main()
    return hydra_main()


if __name__ == "__main__":
    main()
