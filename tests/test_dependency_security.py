from importlib.metadata import version
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, UnidentifiedImageError

from game_config import BREAKOUT_CONFIG
from preprocessing import preprocess_frame
from spatial_model import SpatialLatentWorldModel, load_spatial_model_state_dict


def _numeric_version(distribution: str) -> tuple[int, ...]:
    release = version(distribution).split("+", maxsplit=1)[0]
    return tuple(int(part) for part in release.split(".") if part.isdigit())


@pytest.mark.parametrize(
    ("distribution", "minimum"),
    [
        ("aiohttp", (3, 14, 3)),
        ("idna", (3, 15)),
        ("pillow", (12, 3, 0)),
        ("pygments", (2, 20, 0)),
        ("python-dotenv", (1, 2, 2)),
        ("setuptools", (83, 0, 0)),
        ("torch", (2, 13, 0)),
        ("urllib3", (2, 7, 0)),
        ("wandb", (0, 28, 1)),
    ],
)
def test_audited_dependency_floors(distribution: str, minimum: tuple[int, ...]) -> None:
    assert _numeric_version(distribution) >= minimum


def test_image_boundary_rejects_invalid_data_and_accepts_a_valid_frame() -> None:
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(b"not an image")).verify()

    with pytest.raises(ValueError, match="Unsupported image shape"):
        preprocess_frame(np.zeros((4, 4, 2), dtype=np.uint8), BREAKOUT_CONFIG, (8, 8))

    output = preprocess_frame(np.zeros((12, 12, 3), dtype=np.uint8), BREAKOUT_CONFIG, (8, 8))
    assert output.shape == (8, 8)
    assert output.dtype == np.uint8


def test_torch_runtime_executes_a_legitimate_world_model() -> None:
    model = SpatialLatentWorldModel(
        history_length=2,
        n_actions=4,
        latent_channels=8,
        refine_blocks=1,
    )
    history = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    action = torch.nn.functional.one_hot(torch.tensor([1]), num_classes=4).float()

    with torch.no_grad():
        predicted_frame = model(history, action)

    assert predicted_frame.shape == (1, 1, 8, 8)
    assert torch.isfinite(predicted_frame).all()


def test_torch_upgrade_preserves_committed_checkpoint_compatibility() -> None:
    checkpoint_path = Path(
        "models/modal_real_20260310/"
        "tsilva__gymrec__BreakoutNoFrameskip_dash_v4_stack4_unroll8_"
        "train_ready-spatial-dynamics.pt"
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = SpatialLatentWorldModel(
        history_length=4,
        n_actions=4,
        latent_channels=32,
        refine_blocks=4,
    )

    compatibility = load_spatial_model_state_dict(model, state_dict)
    assert compatibility["missing_keys"] == []
    assert compatibility["unexpected_keys"] == []

    history = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    action = torch.nn.functional.one_hot(torch.tensor([1]), num_classes=4).float()
    with torch.no_grad():
        predicted_frame = model(history, action)

    assert predicted_frame.shape == (1, 1, 8, 8)
    assert torch.isfinite(predicted_frame).all()
