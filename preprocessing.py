from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

from game_config import GameConfig


def _to_pil_image(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image

    image_array = np.asarray(image)
    if image_array.ndim == 2:
        return Image.fromarray(image_array.astype(np.uint8), mode="L")
    if image_array.ndim == 3 and image_array.shape[2] in (3, 4):
        return Image.fromarray(image_array.astype(np.uint8))
    if image_array.ndim == 3 and image_array.shape[2] == 1:
        return Image.fromarray(image_array[:, :, 0].astype(np.uint8), mode="L")
    raise ValueError(f"Unsupported image shape: {image_array.shape}")


def crop_game_area(image: Image.Image | np.ndarray, game_config: GameConfig) -> Image.Image:
    pil_image = _to_pil_image(image).convert("RGB")
    width, height = pil_image.size
    crop_top = min(max(game_config.crop_top, 0), height)
    return pil_image.crop((0, crop_top, width, height))


def has_valid_black_background(image: Image.Image | np.ndarray, game_config: GameConfig) -> bool:
    game_area = crop_game_area(image, game_config)
    image_array = np.asarray(game_area, dtype=np.uint8)
    height, width = image_array.shape[:2]
    top = int(height * game_config.background_check_top_ratio)
    bottom = int(height * game_config.background_check_bottom_ratio)
    left = int(width * game_config.background_check_left_ratio)
    right = int(width * game_config.background_check_right_ratio)

    background_region = image_array[top:bottom, left:right]
    if background_region.size == 0:
        return False
    background_pixels = np.all(
        background_region <= game_config.background_tolerance, axis=2
    )
    return float(background_pixels.mean()) >= game_config.background_threshold


def preprocess_frame(
    image: Image.Image | np.ndarray,
    game_config: GameConfig,
    target_size: tuple[int, int],
) -> np.ndarray:
    game_area = crop_game_area(image, game_config).convert("L")
    resized = game_area.resize(target_size, Image.NEAREST)
    grayscale = np.asarray(resized, dtype=np.uint8)
    binary = (grayscale >= game_config.binarize_threshold).astype(np.uint8)
    return binary


def encode_action(action_value: int | Sequence[int], game_config: GameConfig) -> np.ndarray:
    if isinstance(action_value, np.ndarray):
        if action_value.ndim == 0:
            action_id = int(action_value)
        else:
            action_id = int(action_value.reshape(-1)[0])
    elif isinstance(action_value, Sequence) and not isinstance(action_value, (str, bytes)):
        if len(action_value) == 0:
            action_id = 0
        elif len(action_value) == game_config.n_actions and all(
            value in (0, 1, 0.0, 1.0, False, True) for value in action_value
        ):
            return np.asarray(action_value, dtype=np.float32)
        else:
            action_id = int(action_value[0])
    else:
        action_id = int(action_value)

    if not 0 <= action_id < game_config.n_actions:
        raise ValueError(
            f"Action id {action_id} is outside [0, {game_config.n_actions - 1}]"
        )

    action = np.zeros(game_config.n_actions, dtype=np.float32)
    action[action_id] = 1.0
    return action
