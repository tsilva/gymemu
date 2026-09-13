"""Hydra composition, also usable from tests and notebooks without changing CWD."""

from pathlib import Path

from hydra import compose, initialize_config_dir

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def compose_config(overrides=(), *, recipe=None):
    from gymemu.recipes import recipe_path

    path = recipe_path(recipe) if recipe is not None else CONFIG_DIR / "config.yaml"
    with initialize_config_dir(version_base="1.3", config_dir=str(path.parent)):
        return compose(config_name=path.stem, overrides=list(overrides))
