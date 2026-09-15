"""Locate bundled resources in a checkout or an installed wheel."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CHECKOUT_DIR = PACKAGE_DIR.parent
IS_CHECKOUT = (CHECKOUT_DIR / "pyproject.toml").is_file() and (
    CHECKOUT_DIR / "configs/config.yaml"
).is_file()
RESOURCE_ROOT = CHECKOUT_DIR if IS_CHECKOUT else PACKAGE_DIR
