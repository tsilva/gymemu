"""Explicit model registry. Checkpoints never import arbitrary Python targets."""

from gymemu.models.action_history import ActionHistoryAutoencoder
from gymemu.models.ball_state import BallStateAutoencoder
from gymemu.models.direct import Autoencoder
from gymemu.models.latent import FrameCodec, LatentDynamics

MODELS = {
    "direct_cnn": Autoencoder,
    "action_history_cnn": ActionHistoryAutoencoder,
    "ball_state_cnn": BallStateAutoencoder,
    "frame_codec": FrameCodec,
    "latent_cnn": LatentDynamics,
}


def build_model(spec, **dimensions):
    options = dict(spec)
    kind = options.pop("kind")
    if kind not in MODELS:
        raise ValueError(f"Unknown model {kind!r}; registered models: {sorted(MODELS)}")
    return MODELS[kind](**dimensions, **options)
