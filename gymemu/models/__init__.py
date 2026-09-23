"""Explicit model registry. Checkpoints never import arbitrary Python targets."""

from gymemu.models.action_history import ActionHistoryAutoencoder
from gymemu.models.ball_acceleration import BallAcceleration, BallAccelerationSpatial
from gymemu.models.ball_bricks import BallBricks
from gymemu.models.ball_bricks_contact import BallBricksContact
from gymemu.models.ball_bricks_contact_count import BallBricksContactCount
from gymemu.models.ball_direction_geometry import BallDirectionGeometry
from gymemu.models.ball_horizontal_router import BallHorizontalRouter
from gymemu.models.ball_horizontal_velocity import BallHorizontalVelocity
from gymemu.models.ball_life_termination import BallLifeTermination
from gymemu.models.ball_motion import BallMotion
from gymemu.models.ball_paddle_charge import BallPaddleCharge
from gymemu.models.ball_paddle_position import BallPaddlePosition
from gymemu.models.ball_paddle_width import BallPaddleWidth
from gymemu.models.ball_position import BallPosition
from gymemu.models.ball_state import BallStateAutoencoder
from gymemu.models.ball_velocity import BallVelocityMLP
from gymemu.models.ball_vertical_velocity import BallVerticalVelocity
from gymemu.models.brick_contact import BrickContact
from gymemu.models.brick_layout import BrickLayout
from gymemu.models.controller_history import ControllerHistoryMLP
from gymemu.models.direct import Autoencoder
from gymemu.models.horizontal_ball_pair import HorizontalBallPair
from gymemu.models.latent import FrameCodec, LatentDynamics
from gymemu.models.life_termination import LifeTermination
from gymemu.models.paddle import PaddlePositionCNN, PaddleStateCNN
from gymemu.models.paddle_hit_count import PaddleHitCount
from gymemu.models.paddle_transition import PaddleTransitionMLP
from gymemu.models.paddle_vertical_pair import PaddleVerticalPair
from gymemu.models.paddle_width import PaddleWidth
from gymemu.models.state_context import HiddenStateProbe, PaddleContextProbe
from gymemu.models.state_dynamics import StateGRU, StateMLP
from gymemu.models.state_renderer import StateRenderer
from gymemu.models.unified_state import UnifiedStateMLP
from gymemu.models.vertical_ball_pair import VerticalBallPair
from gymemu.models.vertical_collision_timing import VerticalCollisionTiming
from gymemu.models.vertical_displacement_refinement import VerticalDisplacementRefinement

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
    "brick_contact": BrickContact,
    "paddle_hit_count": PaddleHitCount,
    "paddle_width": PaddleWidth,
    "life_termination": LifeTermination,
    "vertical_ball_pair": VerticalBallPair,
    "horizontal_ball_pair": HorizontalBallPair,
    "ball_motion": BallMotion,
    "ball_paddle_width": BallPaddleWidth,
    "ball_paddle_charge": BallPaddleCharge,
    "ball_paddle_position": BallPaddlePosition,
    "ball_life_termination": BallLifeTermination,
    "ball_bricks": BallBricks,
    "ball_bricks_contact": BallBricksContact,
    "ball_bricks_contact_count": BallBricksContactCount,
    "paddle_vertical_pair": PaddleVerticalPair,
    "vertical_collision_timing": VerticalCollisionTiming,
    "vertical_displacement_refinement": VerticalDisplacementRefinement,
    "frame_codec": FrameCodec,
    "latent_cnn": LatentDynamics,
    "paddle_state_cnn": PaddleStateCNN,
    "paddle_position_cnn": PaddlePositionCNN,
    "state_mlp": StateMLP,
    "unified_state_mlp": UnifiedStateMLP,
    "state_renderer": StateRenderer,
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
