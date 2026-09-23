"""Resolve R2 secrets from the process environment or a local Keychain profile."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

CREDENTIAL_PREFIX = "GYMEMU_MODELS_R2"
PUBLIC_CREDENTIAL_PREFIX = "GYMEMU_PUBLIC_R2"


def r2_credentials(*, public=False):
    prefix = PUBLIC_CREDENTIAL_PREFIX if public else CREDENTIAL_PREFIX
    config_variable = "GYMEMU_PUBLIC_R2_CONFIG" if public else "GYMEMU_R2_CONFIG"
    default_profile = "~/.config/gymemu/public-r2.toml" if public else "~/.config/gymemu/r2.toml"
    names = ("ENDPOINT_URL", "ACCESS_KEY_ID", "SECRET_ACCESS_KEY")
    # Treat an explicit environment as one complete credential set. Never combine
    # an overridden endpoint or key with credentials from another account's profile.
    if any(f"{prefix}_{name}" in os.environ for name in names):
        return {name: os.environ.get(f"{prefix}_{name}", "").strip() for name in names}
    path = Path(os.environ.get(config_variable, default_profile)).expanduser()
    if not path.exists():
        return dict.fromkeys(names, "")
    if path.stat().st_mode & 0o022:
        raise ValueError("The Gymemu R2 profile must not be writable by other users")
    profile = tomllib.loads(path.read_text())
    if sys.platform != "darwin":
        raise ValueError("The R2 Keychain profile requires macOS; export R2 environment variables")
    references = profile["keychain"]
    values = {"ENDPOINT_URL": profile["endpoint_url"]}
    for name in ("access_key_id", "secret_access_key"):
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                references[name],
                "-a",
                references["account"],
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode:
            raise ValueError(f"The Gymemu R2 {name} is unavailable in macOS Keychain")
        values[name.upper()] = result.stdout.strip()
    return values
