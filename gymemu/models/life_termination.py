"""Predict life loss from the current vertical ball state."""

import torch
from torch import nn

from gymemu.models.ball_vertical_velocity import _mlp


class LifeTermination(nn.Module):
    """Binary stop head over the shared physical 118-value source state.

    A constant-velocity proposal is an input feature. The network learns the
    decision; no native collision or termination rule executes at inference.
    """

    def __init__(self, width=64, depth=2):
        super().__init__()
        self.network = _mlp(31, width, depth, 2)

    def encode(self, source):
        if source.ndim != 2 or source.shape[1] != 118:
            raise ValueError("Expected 118 source features")
        y = source[:, 1] + source[:, 8] / 8
        vy = source[:, 3]
        parts = [torch.stack([y / 255, vy / 3.375], dim=1)]
        for values, bits in (((y * 8).round().long(), 11), ((vy * 8 + 27).round().long(), 6)):
            parts.append(
                ((values[:, None] >> torch.arange(bits, device=source.device)) & 1).float() * 2 - 1
            )
        # This proposal does not decide whether a life ends.
        y1 = y + vy
        bits = (y1 * 8).round().long()[:, None] >> torch.arange(11, device=source.device)
        parts.extend([y1[:, None] / 255, (bits & 1).float() * 2 - 1])
        return torch.cat(parts, dim=1)

    def forward_encoded(self, encoded):
        return self.network(encoded)

    def forward(self, source):
        return self.forward_encoded(self.encode(source))

    def predict(self, source):
        return self(source).argmax(dim=1).bool()
