"""Discrete vertical velocity with frozen horizontal dynamics and source-only routing."""

import copy

import torch
from torch import nn

from gymemu.models.ball_acceleration import BallAccelerationSpatial


def _mlp(inputs, width, depth, outputs):
    if min(width, depth) < 1:
        raise ValueError("Invalid vertical network dimensions")
    layers = []
    for _ in range(depth):
        layers.extend([nn.Linear(inputs, width), nn.ReLU()])
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class BallVerticalVelocity(nn.Module):
    """Learn vy from the existing 118 source values, preserving learned vx.

    The class vocabulary is derived from training targets. No native transitions
    or target/event-dependent routing run during inference.
    """

    def __init__(
        self,
        horizontal,
        values,
        upper_width=128,
        upper_depth=2,
        paddle_width=256,
        paddle_depth=3,
    ):
        super().__init__()
        if not values or len(set(values)) != len(values):
            raise ValueError("Expected unique vertical velocity values")
        self.horizontal = BallAccelerationSpatial(**horizontal)
        self.horizontal.requires_grad_(False).eval()
        self.register_buffer("classes", torch.tensor(values, dtype=torch.float32))
        self.cell_network = copy.deepcopy(self.horizontal.cell_network)
        self.cell_network.requires_grad_(True)
        cell_width = self.cell_network[-2].out_features
        self.upper_network = _mlp(cell_width + 71, upper_width, upper_depth, len(values))
        # Keep charge available even when the frozen intermediate-paddle estimate errs.
        paddle_features = self.horizontal.base.paddle_model.direction.network[0].in_features
        self.paddle_network = _mlp(paddle_features + 13, paddle_width, paddle_depth, len(values))
        self.flight_network = _mlp(1, 32, 1, len(values))

    def train(self, mode=True):
        super().train(mode)
        self.horizontal.eval()
        return self

    @staticmethod
    def regions(source):
        if source.ndim != 2 or source.shape[1] != 118:
            raise ValueError("Expected 118 source features")
        paddle = (source[:, 1] >= 160) & (source[:, 1] <= 183) & (source[:, 3] > 0)
        upper = source[:, 1] <= 100
        return upper, paddle, ~(upper | paddle)

    def paddle_encode(self, source):
        with torch.no_grad():
            features = self.horizontal.base.paddle_model.direction.encode(source[:, :9])
        charge = source[:, 6].round().long()
        bits = (charge[:, None] >> torch.arange(12, device=source.device)) & 1
        return torch.cat([features, source[:, 6:7] / 3856, bits.float() * 2 - 1], dim=1)

    def upper_encode(self, source):
        cells, numbers, occupied = self.horizontal.encode(source)
        raw = source[:, self.horizontal.columns]
        integers = (raw * self.horizontal.multipliers + self.horizontal.offsets).round().long()
        parts = [numbers]
        for column, width in enumerate(self.horizontal.bit_widths):
            bits = (integers[:, column, None] >> torch.arange(width, device=source.device)) & 1
            parts.append(bits.float() * 2 - 1)
        x, x1 = source[:, 0], source[:, 0] + source[:, 2]
        y = source[:, 1] + source[:, 8] / 8
        y1 = y + source[:, 3]
        parts.append(
            torch.stack(
                [
                    x / 160,
                    x1 / 160,
                    y / 255,
                    y1 / 255,
                    x - x.floor(),
                    x1 - x1.floor(),
                    y - y.floor(),
                    y1 - y1.floor(),
                ],
                dim=1,
            )
        )
        for value in (x1, y1):
            integer = (value * 8).round().long()
            bits = (integer[:, None] >> torch.arange(11, device=source.device)) & 1
            parts.append(bits.float() * 2 - 1)
        return cells, torch.cat(parts, dim=1), occupied

    def upper_forward_encoded(self, cells, numbers, occupied):
        features = self.cell_network(cells) * occupied[..., None]
        return self.upper_network(torch.cat([features.amax(dim=1), numbers], dim=1))

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
            logits[flight] = self.flight_network(source[flight, 3:4] / 3.375)
        return logits

    def predict(self, source):
        return self.classes[self(source).argmax(-1)]

    def predict_horizontal(self, source):
        with torch.no_grad():
            return self.horizontal.predict(source)
