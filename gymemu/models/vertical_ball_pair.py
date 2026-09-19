"""Joint vertical position/velocity prediction from one immutable source state."""

import torch
from torch import nn

from gymemu.models.ball_position import BallPosition
from gymemu.models.ball_vertical_velocity import _mlp


class VerticalBallPair(nn.Module):
    """Reuse a y checkpoint and its velocity parent, without sequential updates.

    Without a coupling head, only the y and vy heads may train. With one,
    both original models stay frozen and only the correction trains. Their
    horizontal/intermediate-paddle dependencies always stay frozen. The
    combined y output includes fractional y.
    """

    def __init__(self, position, coupling_width=0):
        super().__init__()
        if position.get("axis") != "y":
            raise ValueError("Vertical pair requires a y position model")
        self.position = BallPosition(**position)
        self.position.vertical.requires_grad_(True)
        self.position.vertical.horizontal.requires_grad_(False).eval()
        self.coupling = None
        if coupling_width:
            self.position.requires_grad_(False).eval()
            classes = len(self.position.vertical.classes)
            self.coupling = _mlp(classes + 2, coupling_width, 2, classes)

    def train(self, mode=True):
        super().train(mode)
        if self.coupling is None:
            self.position.vertical.train(mode)
        else:
            self.position.eval()
        return self

    def coupling_encode(self, source):
        with torch.no_grad():
            y_logits = self.position(source)
            vy_logits = self.position.vertical(source)
            return self._coupling_features(source, y_logits, vy_logits)

    def _coupling_features(self, source, y_logits, vy_logits):
        delta = self.position.classes[y_logits.argmax(-1)]
        return torch.cat(
            [source[:, 3:4] / 3.375, delta[:, None] / 6.75, vy_logits.softmax(-1)], dim=1
        )

    def forward(self, source):
        if self.coupling is None:
            return self.position(source), self.position.vertical(source)
        with torch.no_grad():
            y_logits = self.position(source)
            vy_logits = self.position.vertical(source)
            features = self._coupling_features(source, y_logits, vy_logits)
        return y_logits, self.coupling(features)

    def predict(self, source):
        y_logits, vy_logits = self(source)
        y = source[:, 1] + source[:, 8] / 8
        return torch.stack(
            [
                y + self.position.classes[y_logits.argmax(-1)],
                self.position.vertical.classes[vy_logits.argmax(-1)],
            ],
            dim=1,
        )
