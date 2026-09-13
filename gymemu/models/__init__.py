"""Explicit model registry. Checkpoints never import arbitrary Python targets."""

from gymemu.models.direct import Autoencoder
from gymemu.models.latent import FrameCodec, LatentDynamics

MODELS = {
    "direct_cnn": Autoencoder,
    "frame_codec": FrameCodec,
    "latent_cnn": LatentDynamics,
}


def build_model(spec, **dimensions):
    options = dict(spec)
    kind = options.pop("kind")
    if kind not in MODELS:
        raise ValueError(f"Unknown model {kind!r}; registered models: {sorted(MODELS)}")
    return MODELS[kind](**dimensions, **options)
