"""Predict the next brick-contact memory using frozen learned brick dynamics."""

import torch
from torch import nn

from gymemu.models.ball_vertical_velocity import _mlp
from gymemu.models.brick_layout import BrickLayout


class BrickContact(nn.Module):
    """Classify next contact from current geometry and predicted brick collisions.

    The frozen layout model supplies a learned collision signal. Targets and
    native game rules are never consulted during inference.
    """

    def __init__(self, layout, width=128, depth=2, collision_geometry=False):
        super().__init__()
        self.layout = BrickLayout(**layout)
        self.layout.requires_grad_(False).eval()
        self.collision_geometry = collision_geometry
        self.network = _mlp(86 if collision_geometry else 73, width, depth, 2)

    def train(self, mode=True):
        super().train(mode)
        self.layout.eval()
        return self

    def encode(self, source):
        with torch.no_grad():
            cells, numbers, occupied = self.layout.encode(source)
            logits = self.layout.forward_encoded(cells, numbers, occupied)
            event_logits = torch.stack(
                [logits[:, 0], torch.logsumexp(logits[:, 1:], dim=1)], dim=1
            )
            parts = [numbers, event_logits.softmax(dim=1)]
            if self.collision_geometry:
                probabilities = logits.softmax(dim=1)[:, 1:]
                parts.append((cells * probabilities[..., None]).sum(dim=1))
            return torch.cat(parts, dim=1)

    def forward_encoded(self, encoded):
        return self.network(encoded)

    def forward(self, source):
        return self.forward_encoded(self.encode(source))

    def predict(self, source):
        return self(source).argmax(dim=1)
