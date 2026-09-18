"""Learn horizontal speed while retaining a frozen learned direction predictor."""

import torch
from torch import nn

from gymemu.models.ball_direction_geometry import BallDirectionGeometry


class BallHorizontalVelocity(nn.Module):
    """Near-paddle velocity from separately learned direction and magnitude."""

    def __init__(self, direction, width=128, depth=2, activation="relu"):
        super().__init__()
        if min(width, depth) < 1 or activation not in ("relu", "silu"):
            raise ValueError("Invalid speed network dimensions or activation")
        self.direction = BallDirectionGeometry(**direction)
        self.direction.requires_grad_(False).eval()
        self.register_buffer("speeds", torch.tensor([0.5, 1.0, 1.5, 2.0]))
        inputs = self.direction.network[0].in_features
        layers = []
        for _ in range(depth):
            layers.extend(
                [nn.Linear(inputs, width), nn.ReLU() if activation == "relu" else nn.SiLU()]
            )
            inputs = width
        layers.append(nn.Linear(inputs, len(self.speeds)))
        self.network = nn.Sequential(*layers)

    def train(self, mode=True):
        super().train(mode)
        self.direction.eval()
        return self

    def encode(self, source):
        with torch.no_grad():
            return self.direction.encode(source)

    def forward(self, source):
        """Return speed logits for the isolated cross-entropy objective."""
        return self.network(self.encode(source))

    def predict_speed(self, source):
        return self.speeds[self(source).argmax(-1)]

    def predict(self, source):
        with torch.no_grad():
            sign = self.direction.predict(source)
        return sign * self.predict_speed(source)
