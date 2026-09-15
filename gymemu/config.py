"""Hydra composition, also usable from tests and notebooks without changing CWD."""

from hydra import compose, initialize_config_dir

from gymemu.resources import RESOURCE_ROOT

CONFIG_DIR = RESOURCE_ROOT / "configs"


def compose_config(overrides=(), *, recipe=None):
    from gymemu.recipes import recipe_path

    path = recipe_path(recipe) if recipe is not None else CONFIG_DIR / "config.yaml"
    with initialize_config_dir(version_base="1.3", config_dir=str(path.parent)):
        return compose(config_name=path.stem, overrides=list(overrides))
