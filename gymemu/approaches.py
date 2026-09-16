"""Approaches own model composition, objectives, and stage-specific trainable parameters."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from gymemu.ball_regions import ball_region_mask
from gymemu.models import build_model


class Approach(nn.Module):
    action_history = 1
    state_fields = ()
    training_rollout_steps = 0
    training_future_steps = 0
    objectives = ()
    predictive_objectives = ()

    def configure_training(self, *, compile=False):
        """Configure optional execution helpers without changing saved model weights."""

    def begin_epoch(self, epoch):
        """Update training curricula outside compiled computation; return log metadata."""
        return {}

    def reset_epoch_metrics(self):
        """Reset optional diagnostics before each training or validation pass."""

    def epoch_metrics(self):
        """Return optional scalar diagnostics, separate from comparison metrics."""
        return {}

    def interval_metrics(self):
        """Optional objective diagnostics since the last logging interval."""
        return {}

    def validate_stages(self, stages):
        for stage in stages:
            if stage["objective"] not in self.objectives:
                raise ValueError(f"Unsupported objective: {stage['objective']}")
        if stages[-1]["objective"] not in self.predictive_objectives:
            raise ValueError("The final stage must train next-frame prediction")

    def prepare_stage(self, objective):
        if objective not in self.objectives:
            raise ValueError(f"Unsupported objective {objective!r} for {type(self).__name__}")
        self.objective = objective
        self.zero_grad(set_to_none=True)
        self.requires_grad_(True)
        self.train()
        return [p for p in self.parameters() if p.requires_grad]

    def loss(self, history, action, target):
        return F.mse_loss(self(history, action).float(), target.float())

    def evaluate(self, history, action, target):
        return self.loss(history, action, target), self(history, action)


class DirectApproach(Approach):
    objectives = predictive_objectives = ("next_frame",)

    def __init__(self, spec, history, actions, shape):
        super().__init__()
        self.predictor = build_model(
            spec["predictor"], history=history, actions=actions, shape=shape
        )

    def validate_stages(self, stages):
        super().validate_stages(stages)
        if getattr(self.predictor, "action_history", 1) != self.action_history:
            raise ValueError("Predictor action history requires a matching approach input contract")

    def forward(self, history, action):
        return self.predictor(history, action)

    def evaluate(self, history, action, target):
        prediction = self(history, action)
        return F.mse_loss(prediction.float(), target.float()), prediction


class ActionHistoryApproach(DirectApproach):
    def __init__(self, spec, history, actions, shape):
        super().__init__(spec, history, actions, shape)
        if not hasattr(self.predictor, "action_history"):
            raise ValueError("direct_actions requires a predictor declaring action_history")
        self.action_history = self.predictor.action_history


class BallRegionApproach(ActionHistoryApproach):
    objectives = predictive_objectives = ("next_frame_ball_region",)

    def __init__(
        self,
        spec,
        history,
        actions,
        shape,
        *,
        sprite_height,
        sprite_width,
        padding=4,
        ball_region_weight=0.3,
    ):
        super().__init__(spec, history, actions, shape)
        if shape[0] != 3:
            raise ValueError("Ball-region detection requires RGB frames")
        for name, value in (("sprite_height", sprite_height), ("sprite_width", sprite_width)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(padding) is not int or padding < 0:
            raise ValueError("padding must be a nonnegative integer")
        if (
            type(ball_region_weight) not in (int, float)
            or not math.isfinite(ball_region_weight)
            or ball_region_weight < 0
        ):
            raise ValueError("ball_region_weight must be finite and nonnegative")
        self.sprite_height, self.sprite_width = sprite_height, sprite_width
        self.padding, self.ball_region_weight = padding, ball_region_weight
        self.register_buffer("_region_totals", torch.zeros(4), persistent=False)
        self.register_buffer("_region_previous", torch.zeros(4), persistent=False)

    def reset_epoch_metrics(self):
        self._region_totals.zero_()
        self._region_previous.zero_()

    def epoch_metrics(self):
        samples, detected, rgb, region = self._region_totals.tolist()
        return {
            "ball_detection_coverage": detected / max(samples, 1),
            "rgb_mse": rgb / max(samples, 1),
            "ball_region_mse": region / max(samples, 1),
        }

    def interval_metrics(self):
        samples, detected, rgb, region = (self._region_totals - self._region_previous).tolist()
        self._region_previous.copy_(self._region_totals)
        if not samples:
            return {}
        return {
            "train/rgb/mse": rgb / samples,
            "train/ball/mse": region / samples,
            "train/ball/coverage": detected / samples,
        }

    def joint_loss(self, prediction, target, *, valid=None, reduction="mean"):
        mask, available = ball_region_mask(
            target,
            sprite_height=self.sprite_height,
            sprite_width=self.sprite_width,
            padding=self.padding,
        )
        error = (prediction.float() - target.float()).square().mean(dim=1, keepdim=True)
        rgb = error.flatten(1).mean(dim=1)
        # Each detected region gets its own mean, including clipped edge regions.
        # Undetected samples contribute zero; full-batch averaging is partition invariant.
        region = (error * mask).flatten(1).sum(dim=1) / mask.flatten(1).sum(dim=1).clamp_min(1)
        if valid is None:
            valid = torch.ones_like(rgb)
        else:
            valid = valid.to(rgb.dtype)
        with torch.no_grad():
            self._region_totals.add_(
                torch.stack(
                    (
                        valid.sum(),
                        (available * valid).sum().float(),
                        (rgb.detach() * valid).sum(),
                        (region.detach() * valid).sum(),
                    )
                )
            )
        losses = (rgb + self.ball_region_weight * region) * valid
        return losses if reduction == "none" else losses.sum() / valid.sum().clamp_min(1)

    def loss(self, history, action, target):
        return self.joint_loss(self(history, action), target)

    def evaluate(self, history, action, target):
        prediction = self(history, action)
        return self.joint_loss(prediction, target), prediction


class BallStateApproach(ActionHistoryApproach):
    state_fields = ("ball_x_normalized", "ball_y_normalized")
    objectives = predictive_objectives = ("next_frame_and_ball",)

    def __init__(self, spec, history, actions, shape, *, coordinate_loss_weight=0.01):
        super().__init__(spec, history, actions, shape)
        if (
            type(coordinate_loss_weight) not in (int, float)
            or not math.isfinite(coordinate_loss_weight)
            or coordinate_loss_weight <= 0
        ):
            raise ValueError("coordinate_loss_weight must be finite and positive")
        self.coordinate_loss_weight = coordinate_loss_weight

    def forward(self, history, action, state_history):
        return self.predictor(history, action, state_history)

    def predict_step(self, history, action, state_history):
        rgb, coordinates = self(history, action, state_history)
        return rgb, torch.cat((coordinates, torch.ones_like(coordinates[:, :1])), dim=1)

    def joint_loss(self, prediction, coordinates, target, state_target):
        error = (coordinates.float() - state_target[:, :-1].float()).square().mean(dim=1)
        available = state_target[:, -1].float()
        # Average over the full batch so aggregation stays independent of batch boundaries.
        state_mse = (error * available).mean()
        rgb_mse = F.mse_loss(prediction.float(), target.float())
        return rgb_mse + self.coordinate_loss_weight * state_mse

    def loss(self, history, action, target, state_history, state_target):
        prediction, coordinates = self(history, action, state_history)
        return self.joint_loss(prediction, coordinates, target, state_target)

    def evaluate(self, history, action, target, state_history, state_target):
        prediction, coordinates = self(history, action, state_history)
        return self.joint_loss(prediction, coordinates, target, state_target), prediction


class ScheduledSamplingApproach(ActionHistoryApproach):
    def __init__(
        self,
        spec,
        history,
        actions,
        shape,
        *,
        rollout_steps,
        schedule,
        feedback_dtype="fp32",
        selective_threshold=0.0,
        **objective_options,
    ):
        super().__init__(spec, history, actions, shape, **objective_options)
        if type(rollout_steps) is not int or not 1 <= rollout_steps <= history:
            raise ValueError("rollout_steps must be between 1 and the RGB history length")
        if set(schedule) != {"warmup_epochs", "ramp_epochs", "max_probability"}:
            raise ValueError("Schedule needs warmup_epochs, ramp_epochs, and max_probability")
        warmup, ramp, maximum = (
            schedule["warmup_epochs"],
            schedule["ramp_epochs"],
            schedule["max_probability"],
        )
        if type(warmup) is not int or warmup < 0 or type(ramp) is not int or ramp < 1:
            raise ValueError("warmup_epochs must be nonnegative and ramp_epochs positive")
        if type(maximum) not in (float, int) or not math.isfinite(maximum) or not 0 <= maximum <= 1:
            raise ValueError("max_probability must be between 0 and 1")
        self.history_length = history
        self.training_rollout_steps = rollout_steps
        self.schedule = dict(schedule)
        if feedback_dtype not in ("fp32", "autocast"):
            raise ValueError("feedback_dtype must be fp32 or autocast")
        if (
            type(selective_threshold) not in (int, float)
            or not math.isfinite(selective_threshold)
            or not 0 <= selective_threshold <= 1
        ):
            raise ValueError("selective_threshold must be between 0 and 1")
        self.feedback_dtype = feedback_dtype
        self.selective_threshold = selective_threshold
        self.sampling_probability = 0.0
        self.selective_enabled = False
        self._predict_feedback = self._selected_prediction
        # A tensor lets compiled computation see new probabilities without recompiling
        # for each epoch. This is training state, not an inference weight.
        self.register_buffer("prediction_probability", torch.tensor(0.0), persistent=False)
        self.sampling_enabled = False

    def begin_epoch(self, epoch):
        progress = min(
            1.0, max(0.0, (epoch - self.schedule["warmup_epochs"]) / self.schedule["ramp_epochs"])
        )
        probability = self.schedule["max_probability"] * progress
        self.prediction_probability.fill_(probability)
        self.sampling_enabled = probability > 0
        self.sampling_probability = probability
        self.selective_enabled = 0 < probability < self.selective_threshold
        return {"prediction_probability": probability, "rollout_steps": self.training_rollout_steps}

    def _model_actions(self, actions):
        return actions[:, 0] if self.action_history == 1 else actions

    def configure_training(self, *, compile=False):
        self._predict_feedback = (
            torch.compile(self._selected_prediction, dynamic=True)
            if compile and self.selective_threshold > 0
            else self._selected_prediction
        )

    def _feedback_history(self, recorded):
        device = recorded.device.type
        if self.feedback_dtype == "autocast" and torch.is_autocast_enabled(device):
            return recorded.to(torch.get_autocast_dtype(device))
        return recorded

    def mixed_history(self, recorded, actions, *, masks=None):
        """Generate each replacement from earlier context, never from its target image."""
        recorded = self._feedback_history(recorded)
        length = self.history_length
        context = recorded[:, :length]
        with torch.no_grad():
            for step in range(self.training_rollout_steps):
                tokens = actions[:, step]
                valid = tokens[:, -1] >= 0  # Negative positions precede the episode.
                predicted = self(context, self._model_actions(tokens.clamp_min(0))).to(
                    recorded.dtype
                )
                choose_prediction = (
                    (
                        torch.rand(len(recorded), device=recorded.device)
                        < self.prediction_probability
                    )
                    if masks is None
                    else masks[step].to(recorded.device)
                ) & valid
                frame = torch.where(
                    choose_prediction[:, None, None, None], predicted, recorded[:, length + step]
                )
                context = torch.cat((context[:, 1:], frame[:, None]), dim=1)
        return context

    def _selected_prediction(self, timeline, actions, rows, steps):
        offsets = torch.arange(self.history_length, device=timeline.device)
        context = timeline[rows[:, None], steps[:, None] + offsets]
        tokens = actions[rows, steps]
        return self(context, self._model_actions(tokens.clamp_min(0)))

    @torch.compiler.disable
    def selective_history(self, recorded, actions, *, masks=None):
        """Batch by replacement rank, preserving each example's dependencies.

        First selected frames are independent, even at different time positions.
        Generate them together, then second selections, and so on. CPU Bernoulli
        masks avoid CUDA nonzero synchronization. Pad inference batches to multiples
        of 16, discarding duplicate padding rows before updating the timeline.
        """
        recorded = self._feedback_history(recorded)
        length, steps_count = self.history_length, self.training_rollout_steps
        timeline = recorded.detach().clone()
        if masks is None:
            masks = torch.rand(steps_count, len(recorded)) < self.sampling_probability
        positions = torch.arange(steps_count).expand(len(recorded), -1)
        positions = positions.masked_fill(~masks.T, steps_count).sort(dim=1).values
        with torch.no_grad():
            for rank in range(int(masks.sum(dim=0).max())):
                rows = (positions[:, rank] < steps_count).nonzero().flatten()
                steps = positions[rows, rank]
                count = len(rows)
                padded = ((count + 15) // 16) * 16
                rows = F.pad(rows, (0, padded - count), value=int(rows[0])).to(recorded.device)
                steps = F.pad(steps, (0, padded - count), value=int(steps[0])).to(recorded.device)
                prediction = self._predict_feedback(timeline, actions, rows, steps).to(
                    recorded.dtype
                )
                rows, steps = rows[:count], steps[:count]
                valid = actions[rows, steps, -1] >= 0
                timeline[rows, length + steps] = torch.where(
                    valid[:, None, None, None],
                    prediction[:count],
                    timeline[rows, length + steps],
                )
        return timeline[:, -length:]

    def loss(self, history, action, target):
        if self.sampling_enabled:
            if self.selective_enabled:
                context = self.selective_history(self._feedback_history(history), action)
            else:
                context = self.mixed_history(history, action)
        else:
            context = history[:, -self.history_length :]
        prediction = self(context, self._model_actions(action[:, -1]))
        return self.prediction_loss(prediction, target)

    def prediction_loss(self, prediction, target):
        return F.mse_loss(prediction.float(), target.float())


class ScheduledBallRegionApproach(ScheduledSamplingApproach, BallRegionApproach):
    """Reuse sampled context generation and the target-only ball objective."""

    def prediction_loss(self, prediction, target):
        return self.joint_loss(prediction, target)


class AutoregressiveBallRegionApproach(BallRegionApproach):
    """Supervise every future frame, differentiating through RGB feedback."""

    def __init__(
        self,
        spec,
        history,
        actions,
        shape,
        *,
        rollout_steps,
        rollout_schedule=None,
        feedback_dtype="fp32",
        **options,
    ):
        super().__init__(spec, history, actions, shape, **options)
        if feedback_dtype not in ("fp32", "autocast"):
            raise ValueError("feedback_dtype must be fp32 or autocast")
        self.feedback_dtype = feedback_dtype
        if type(rollout_steps) is not int or rollout_steps < 1:
            raise ValueError("rollout_steps must be a positive integer")
        schedule = list(rollout_schedule) if rollout_schedule is not None else [rollout_steps]
        if (
            not schedule
            or any(type(n) is not int or not 1 <= n <= rollout_steps for n in schedule)
            or schedule != sorted(schedule)
            or schedule[-1] != rollout_steps
        ):
            raise ValueError(
                "rollout_schedule must increase to rollout_steps with positive lengths"
            )
        self.training_future_steps = rollout_steps
        self.rollout_schedule = schedule
        self.active_steps = schedule[0]

    def begin_epoch(self, epoch):
        self.active_steps = self.rollout_schedule[
            min(max(epoch - 1, 0), len(self.rollout_schedule) - 1)
        ]
        return {"rollout_steps": self.active_steps, "prediction_probability": 1.0}

    def loss(self, history, action, target):
        context = history
        if self.feedback_dtype == "autocast" and torch.is_autocast_enabled(history.device.type):
            context = history.to(torch.get_autocast_dtype(history.device.type))
        total = history.new_zeros(len(history), dtype=torch.float32)
        counts = torch.zeros_like(total)
        for step in range(self.active_steps):
            tokens = action[:, step]
            valid = tokens[:, -1] >= 0
            tokens = tokens.clamp_min(0)
            model_action = tokens[:, 0] if self.action_history == 1 else tokens
            prediction = self(context, model_action)
            total = total + self.joint_loss(
                prediction, target[:, step], valid=valid, reduction="none"
            )
            counts = counts + valid
            # No detach: later losses differentiate through all earlier predictions.
            context = torch.cat((context[:, 1:], prediction[:, None].to(context.dtype)), dim=1)
        # Each sampled start has equal weight, independent of its remaining episode length.
        return (total / counts.clamp_min(1)).mean()


class LatentApproach(Approach):
    objectives = ("reconstruction", "latent_prediction")
    predictive_objectives = ("latent_prediction",)

    def validate_stages(self, stages):
        super().validate_stages(stages)
        if [stage["objective"] for stage in stages] != ["reconstruction", "latent_prediction"]:
            raise ValueError("The latent approach requires reconstruction then latent_prediction")

    def __init__(self, spec, history, actions, shape):
        super().__init__()
        self.shape = tuple(shape)
        self.codec = build_model(spec["codec"], channels=shape[0])
        self.dynamics = build_model(
            spec["dynamics"],
            history=history,
            actions=actions,
            latent_channels=self.codec.latent_channels,
        )
        self.objective = "reconstruction"

    def prepare_stage(self, objective):
        super().prepare_stage(objective)
        self.codec.requires_grad_(objective == "reconstruction")
        self.dynamics.requires_grad_(objective == "latent_prediction")
        self.train()
        return [p for p in self.parameters() if p.requires_grad]

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "codec"):
            self.codec.train(mode and self.objective == "reconstruction")
            self.dynamics.train(mode and self.objective == "latent_prediction")
        return self

    def predicted_latents(self, history, action):
        batch, length = history.shape[:2]
        encoded = self.codec.encode(history.flatten(0, 1))
        return self.dynamics(encoded.reshape(batch, length, *encoded.shape[1:]), action)

    def forward(self, history, action):
        return self.codec.decode(self.predicted_latents(history, action), self.shape)

    def loss(self, history, action, target):
        if self.objective == "reconstruction":
            # Only training targets enter representation fitting, never held-out images.
            return F.mse_loss(self.codec(target).float(), target.float())
        with torch.no_grad():
            latent_target = self.codec.encode(target)
        return F.mse_loss(self.predicted_latents(history, action).float(), latent_target.float())


APPROACHES = {
    "direct": DirectApproach,
    "direct_actions": ActionHistoryApproach,
    "ball_region": BallRegionApproach,
    "ball_state": BallStateApproach,
    "latent": LatentApproach,
    "scheduled_actions": ScheduledSamplingApproach,
    "scheduled_ball_region": ScheduledBallRegionApproach,
    "autoregressive_ball_region": AutoregressiveBallRegionApproach,
}


def build_approach(spec, history, actions, shape):
    kind = spec["kind"]
    if kind not in APPROACHES:
        raise ValueError(f"Unknown approach {kind!r}; registered approaches: {sorted(APPROACHES)}")
    return APPROACHES[kind](spec["models"], history, actions, shape, **spec.get("options", {}))
