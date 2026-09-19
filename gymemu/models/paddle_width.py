"""Predict recorded paddle width from current vertical ball state and width."""

import torch
from torch import nn

from gymemu.models.ball_vertical_velocity import _mlp


class PaddleWidth(nn.Module):
    """Classify next width without running a native ceiling-collision rule.

    Inputs use the shared physical 118-value state contract. Integer and
    fractional y form one eighth-pixel coordinate. No other state field or
    predicted successor is used. Life resets remain external boundaries.
    """

    def __init__(self, width=64, depth=2, proposal=False):
        super().__init__()
        self.proposal = proposal
        self.register_buffer("classes", torch.tensor([12.0, 16.0]))
        self.network = _mlp(33 if proposal else 21, width, depth, 2)

    def encode(self, source):
        if source.ndim != 2 or source.shape[1] != 118:
            raise ValueError("Expected 118 source features")
        y = source[:, 1] + source[:, 8] / 8
        vy = source[:, 3]
        paddle_width = source[:, 5]
        parts = [torch.stack([y / 255, vy / 3.375, paddle_width / 16], dim=1)]
        for values, bits in (((y * 8).round().long(), 11), ((vy * 8 + 27).round().long(), 6)):
            parts.append(
                ((values[:, None] >> torch.arange(bits, device=source.device)) & 1).float() * 2 - 1
            )
        parts.append((paddle_width == 12).to(source.dtype)[:, None])
        if self.proposal:
            # A constant-velocity proposal is a feature, not an executed transition.
            y1 = y + vy
            bits = (y1 * 8).round().long()[:, None] >> torch.arange(11, device=source.device)
            parts.extend([y1[:, None] / 255, (bits & 1).float() * 2 - 1])
        return torch.cat(parts, dim=1)

    def forward_encoded(self, encoded):
        return self.network(encoded)

    def forward(self, source):
        return self.forward_encoded(self.encode(source))

    def predict(self, source):
        return self.classes[self(source).argmax(dim=1)]
