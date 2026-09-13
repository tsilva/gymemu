"""Approaches own model composition, objectives, and stage-specific trainable parameters."""

import torch
from torch import nn
from torch.nn import functional as F

from gymemu.models import build_model


class Approach(nn.Module):
    objectives = ()
    predictive_objectives = ()

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

    def forward(self, history, action):
        return self.predictor(history, action)

    def evaluate(self, history, action, target):
        prediction = self(history, action)
        return F.mse_loss(prediction.float(), target.float()), prediction


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


APPROACHES = {"direct": DirectApproach, "latent": LatentApproach}


def build_approach(spec, history, actions, shape):
    kind = spec["kind"]
    if kind not in APPROACHES:
        raise ValueError(f"Unknown approach {kind!r}; registered approaches: {sorted(APPROACHES)}")
    return APPROACHES[kind](spec["models"], history, actions, shape)
