"""Portable inference bundles for one or several models, with v1 compatibility."""

from pathlib import Path

import torch

from gymemu.approaches import build_approach
from gymemu.models.direct import Autoencoder


def load_model(checkpoint: Path, device: torch.device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    version = config.get("format_version")
    if version == 1:
        model = Autoencoder(
            config["history"], len(config["action_values"]), config["shape"], config["width"]
        )
    elif version == 2:
        model = build_approach(
            config["approach"], config["history"], len(config["action_values"]), config["shape"]
        )
    else:
        raise ValueError(f"Unsupported checkpoint format {version!r}")
    if config.get("action_history", 1) != getattr(model, "action_history", 1):
        raise ValueError("Checkpoint action-history contract differs from its model")
    if tuple(config.get("state_fields", ())) != tuple(getattr(model, "state_fields", ())):
        raise ValueError("Checkpoint state-fields contract differs from its model")
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval(), config


def save_model(path, model, config):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "config": config,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        temporary,
    )
    temporary.replace(path)
