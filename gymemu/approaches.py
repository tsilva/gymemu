"""Approaches own model composition, objectives, and stage-specific trainable parameters."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from gymemu.models import build_model


class Approach(nn.Module):
    action_history = 1
    training_rollout_steps = 0
    objectives = ()
    predictive_objectives = ()

    def begin_epoch(self, epoch):
        """Update training curricula outside compiled computation; return log metadata."""
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


class ScheduledSamplingApproach(ActionHistoryApproach):
    def __init__(self, spec, history, actions, shape, *, rollout_steps, schedule):
        super().__init__(spec, history, actions, shape)
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
        return {"prediction_probability": probability, "rollout_steps": self.training_rollout_steps}

    def _model_actions(self, actions):
        return actions[:, 0] if self.action_history == 1 else actions

    def mixed_history(self, recorded, actions):
        """Generate each replacement from earlier context, never from its target image."""
        length = self.history_length
        context = recorded[:, :length]
        with torch.no_grad():
            for step in range(self.training_rollout_steps):
                tokens = actions[:, step]
                valid = tokens[:, -1] >= 0  # Negative positions precede the episode.
                predicted = self(context, self._model_actions(tokens.clamp_min(0))).float()
                choose_prediction = (
                    torch.rand(len(recorded), device=recorded.device) < self.prediction_probability
                ) & valid
                frame = torch.where(
                    choose_prediction[:, None, None, None], predicted, recorded[:, length + step]
                )
                context = torch.cat((context[:, 1:], frame[:, None]), dim=1)
        return context

    def loss(self, history, action, target):
        if self.sampling_enabled:
            context = self.mixed_history(history, action)
        else:
            context = history[:, -self.history_length :]
        prediction = self(context, self._model_actions(action[:, -1]))
        return F.mse_loss(prediction.float(), target.float())


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
    "latent": LatentApproach,
    "scheduled_actions": ScheduledSamplingApproach,
}


def build_approach(spec, history, actions, shape):
    kind = spec["kind"]
    if kind not in APPROACHES:
        raise ValueError(f"Unknown approach {kind!r}; registered approaches: {sorted(APPROACHES)}")
    return APPROACHES[kind](spec["models"], history, actions, shape, **spec.get("options", {}))
