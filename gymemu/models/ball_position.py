"""Discrete ball displacement, with frozen learned velocities and source-only routing."""

import copy
import math

import torch
from torch import nn

from gymemu.models.ball_vertical_velocity import BallVerticalVelocity, _mlp


class BallPosition(nn.Module):
    """Predict one coordinate from the existing 118-value current-state contract.

    The y coordinate includes the fractional eighth-pixel component in RAM's
    coordinate system. Class values are observed displacements, not positions.
    No native collision rules or recorded successor values run during inference.
    """

    def __init__(
        self,
        vertical,
        axis,
        values,
        upper_width=128,
        upper_depth=2,
        paddle_width=256,
        paddle_depth=3,
        flight_width=128,
        flight_depth=2,
    ):
        super().__init__()
        if axis not in ("x", "y"):
            raise ValueError("Expected position axis x or y")
        if (
            not values
            or len(set(values)) != len(values)
            or any(not math.isfinite(v) or v * 8 != round(v * 8) for v in values)
        ):
            raise ValueError("Expected unique finite eighth-pixel displacements")
        self.axis = axis
        self.vertical = BallVerticalVelocity(**vertical)
        self.vertical.requires_grad_(False).eval()
        self.register_buffer("classes", torch.tensor(values, dtype=torch.float32))
        self.cell_network = copy.deepcopy(self.vertical.cell_network)
        self.cell_network.requires_grad_(True)
        cell_width = self.cell_network[-2].out_features
        self.upper_network = _mlp(cell_width + 71, upper_width, upper_depth, len(values))
        self.paddle_network = _mlp(
            self.vertical.paddle_network[0].in_features, paddle_width, paddle_depth, len(values)
        )
        self.flight_network = _mlp(
            31 if axis == "x" else 1, flight_width, flight_depth, len(values)
        )

    def train(self, mode=True):
        super().train(mode)
        self.vertical.eval()
        return self

    @staticmethod
    def regions(source):
        return BallVerticalVelocity.regions(source)

    def paddle_encode(self, source):
        with torch.no_grad():
            return self.vertical.paddle_encode(source)

    def upper_encode(self, source):
        return self.vertical.upper_encode(source)

    def upper_forward_encoded(self, cells, numbers, occupied):
        features = self.cell_network(cells) * occupied[..., None]
        return self.upper_network(torch.cat([features.amax(dim=1), numbers], dim=1))

    def flight_encode(self, source):
        if self.axis == "y":
            return source[:, 3:4] / 3.375
        x, vx = source[:, 0], source[:, 2]
        parts = [torch.stack([x / 160, vx / 2, (x + vx) / 160], dim=1)]
        for value, width in ((x * 8, 11), (vx * 8 + 16, 6), ((x + vx) * 8, 11)):
            bits = (value.round().long()[:, None] >> torch.arange(width, device=source.device)) & 1
            parts.append(bits.float() * 2 - 1)
        return torch.cat(parts, dim=1)

    def forward(self, source):
        upper, paddle, flight = self.regions(source)
        logits = torch.empty(
            (len(source), len(self.classes)), device=source.device, dtype=self.classes.dtype
        )
        if upper.any():
            logits[upper] = self.upper_forward_encoded(*self.upper_encode(source[upper]))
        if paddle.any():
            logits[paddle] = self.paddle_network(self.paddle_encode(source[paddle]))
        if flight.any():
            logits[flight] = self.flight_network(self.flight_encode(source[flight]))
        return logits

    def predict_displacement(self, source):
        return self.classes[self(source).argmax(-1)]

    def predict(self, source):
        current = source[:, 0] if self.axis == "x" else source[:, 1] + source[:, 8] / 8
        return current + self.predict_displacement(source)

    def predict_y_parts(self, source):
        """Return integer RAM y and its fractional remainder in eighth pixels."""
        if self.axis != "y":
            raise ValueError("Only the y model has fractional y components")
        y = self.predict(source)
        integer = y.floor()
        return torch.stack([integer, ((y - integer) * 8).round()], dim=1)

    def predict_velocities(self, source):
        with torch.no_grad():
            return torch.stack(
                [self.vertical.predict_horizontal(source), self.vertical.predict(source)], dim=1
            )
