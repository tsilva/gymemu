"""Source-state routing between full-field and frozen near-paddle vx predictors."""

import torch
from torch import nn

from gymemu.models.ball_horizontal_velocity import BallHorizontalVelocity
from gymemu.models.ball_velocity import BallVelocityMLP


class BallHorizontalRouter(nn.Module):
    """Predict native vx over one recorded two-native-frame transition.

    Both experts are learned. The gate uses only the current RAM y and vy,
    matching the source region used to train the near-paddle expert.
    """

    def __init__(self, global_model, paddle_model):
        super().__init__()
        self.global_model = BallVelocityMLP(**global_model)
        if self.global_model.source_size != 118 or self.global_model.objective != "classification":
            raise ValueError("Router requires a full-field 118-input vx classifier")
        self.paddle_model = BallHorizontalVelocity(**paddle_model)
        self.paddle_model.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.paddle_model.eval()
        return self

    @staticmethod
    def paddle_region(source):
        if source.ndim != 2 or source.shape[1] != 118:
            raise ValueError("Expected 118 source features")
        return (source[:, 1] >= 160) & (source[:, 1] <= 183) & (source[:, 3] > 0)

    def forward(self, source):
        """Return native vx values; train logits through global_model outside the gate."""
        gate = self.paddle_region(source)
        result = torch.empty(
            len(source), device=source.device, dtype=self.global_model.classes.dtype
        )
        if gate.any():
            with torch.no_grad():
                result[gate] = self.paddle_model.predict(source[gate, :9])
        if (~gate).any():
            result[~gate] = self.global_model.predict(source[~gate])
        return result

    def predict(self, source):
        return self(source)
