"""Learn paddle hits and decode a bounded count update."""

import torch
from torch import nn

from gymemu.models.ball_vertical_velocity import BallVerticalVelocity, _mlp


class PaddleHitCount(nn.Module):
    """Predict a hit from current geometry, then retain or increment the count.

    The hit classifier excludes the current count from its features. The broad
    source-only paddle region must cover every hit in the audited cadence.
    Life reset and termination remain the caller's responsibility.
    """

    def __init__(self, vertical, width=256, depth=3):
        super().__init__()
        self.vertical = BallVerticalVelocity(**vertical)
        self.vertical.requires_grad_(False).eval()
        self.network = _mlp(96, width, depth, 2)

    def train(self, mode=True):
        super().train(mode)
        self.vertical.eval()
        return self

    def encode(self, source):
        self.vertical.regions(source)
        with torch.no_grad():
            features = self.vertical.paddle_encode(source)
            # The eighth scalar in the established geometry is prior hit count.
            return torch.cat([features[:, :7], features[:, 8:]], dim=1)

    def forward_encoded(self, encoded):
        return self.network(encoded)

    def forward(self, source):
        paddle = self.vertical.regions(source)[1]
        logits = source.new_zeros((len(source), 2))
        logits[:, 1] = -torch.inf
        if paddle.any():
            logits[paddle] = self.forward_encoded(self.encode(source[paddle]))
        return logits

    def predict_hit(self, source):
        return self(source).argmax(dim=1)

    def predict(self, source):
        return (source[:, 7].round().long() + self.predict_hit(source)).clamp(0, 12)
