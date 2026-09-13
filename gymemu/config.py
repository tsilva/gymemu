"""Hydra composition, also usable from tests and notebooks without changing CWD."""

from pathlib import Path

from hydra import compose, initialize_config_dir

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def compose_config(overrides=()):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=list(overrides))
