"""Explicit model registry. Checkpoints never import arbitrary Python targets."""

from gymemu.models.action_history import ActionHistoryAutoencoder
from gymemu.models.ball_acceleration import BallAcceleration, BallAccelerationSpatial
from gymemu.models.ball_direction_geometry import BallDirectionGeometry
from gymemu.models.ball_horizontal_router import BallHorizontalRouter
from gymemu.models.ball_horizontal_velocity import BallHorizontalVelocity
from gymemu.models.ball_position import BallPosition
from gymemu.models.ball_state import BallStateAutoencoder
from gymemu.models.ball_velocity import BallVelocityMLP
from gymemu.models.ball_vertical_velocity import BallVerticalVelocity
from gymemu.models.brick_layout import BrickLayout
from gymemu.models.controller_history import ControllerHistoryMLP
from gymemu.models.direct import Autoencoder
from gymemu.models.latent import FrameCodec, LatentDynamics
from gymemu.models.paddle import PaddlePositionCNN, PaddleStateCNN
from gymemu.models.paddle_transition import PaddleTransitionMLP
from gymemu.models.state_context import HiddenStateProbe, PaddleContextProbe
from gymemu.models.state_dynamics import StateGRU, StateMLP

MODELS = {
    "direct_cnn": Autoencoder,
    "action_history_cnn": ActionHistoryAutoencoder,
    "ball_state_cnn": BallStateAutoencoder,
    "ball_velocity_mlp": BallVelocityMLP,
    "ball_direction_geometry": BallDirectionGeometry,
    "ball_horizontal_velocity": BallHorizontalVelocity,
    "ball_horizontal_router": BallHorizontalRouter,
    "ball_acceleration": BallAcceleration,
    "ball_acceleration_spatial": BallAccelerationSpatial,
    "ball_vertical_velocity": BallVerticalVelocity,
    "ball_position": BallPosition,
    "brick_layout": BrickLayout,
    "frame_codec": FrameCodec,
    "latent_cnn": LatentDynamics,
    "paddle_state_cnn": PaddleStateCNN,
    "paddle_position_cnn": PaddlePositionCNN,
    "state_mlp": StateMLP,
    "state_paddle_context_probe": PaddleContextProbe,
    "state_hidden_probe": HiddenStateProbe,
    "paddle_transition_mlp": PaddleTransitionMLP,
    "controller_history_mlp": ControllerHistoryMLP,
    "state_gru": StateGRU,
}


def build_model(spec, **dimensions):
    options = dict(spec)
    kind = options.pop("kind")
    if kind not in MODELS:
        raise ValueError(f"Unknown model {kind!r}; registered models: {sorted(MODELS)}")
    return MODELS[kind](**dimensions, **options)
