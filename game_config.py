from dataclasses import dataclass


@dataclass(frozen=True)
class KeyBinding:
    pygame_key: str
    label: str
    action_id: int


@dataclass(frozen=True)
class GameConfig:
    name: str
    dataset_id: str
    n_actions: int
    crop_top: int
    background_tolerance: int
    background_threshold: float
    background_check_top_ratio: float
    background_check_bottom_ratio: float
    background_check_left_ratio: float
    background_check_right_ratio: float
    binarize_threshold: int
    key_bindings: tuple[KeyBinding, ...]


BREAKOUT_CONFIG = GameConfig(
    name="breakout",
    dataset_id="tsilva/gymrec__BreakoutNoFrameskip_dash_v4",
    n_actions=4,
    crop_top=18,
    background_tolerance=12,
    background_threshold=0.90,
    background_check_top_ratio=0.30,
    background_check_bottom_ratio=0.90,
    background_check_left_ratio=0.15,
    background_check_right_ratio=0.85,
    binarize_threshold=64,
    key_bindings=(
        KeyBinding("K_SPACE", "FIRE", 1),
        KeyBinding("K_RIGHT", "RIGHT", 2),
        KeyBinding("K_LEFT", "LEFT", 3),
    ),
)


GAME_CONFIGS = {
    BREAKOUT_CONFIG.name: BREAKOUT_CONFIG,
}


def infer_game_config(dataset_id: str | None = None, game: str | None = None) -> GameConfig:
    if game:
        normalized = game.strip().lower()
        if normalized in GAME_CONFIGS:
            return GAME_CONFIGS[normalized]
        raise ValueError(f"Unsupported game '{game}'. Available: {', '.join(GAME_CONFIGS)}")

    if dataset_id:
        lowered = dataset_id.lower()
        if "breakout" in lowered:
            return BREAKOUT_CONFIG

    return BREAKOUT_CONFIG
