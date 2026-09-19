"""Refine collision displacement while preserving the existing vertical pair."""

import torch
from torch import nn

from gymemu.models.ball_vertical_velocity import _mlp
from gymemu.models.vertical_ball_pair import VerticalBallPair


class VerticalDisplacementRefinement(nn.Module):
    """Learn residual y logits; use corrected displacement in frozen vy coupling.

    Spatial and paddle features, original class probabilities, and all parent
    weights are frozen. Flight predictions remain those of the original model.
    """

    def __init__(self, pair, width=64):
        super().__init__()
        self.pair = VerticalBallPair(**pair)
        if self.pair.coupling is None:
            raise ValueError("Displacement refinement requires a coupled vertical pair")
        self.pair.requires_grad_(False).eval()
        position = self.pair.position
        extra = len(position.classes) + len(position.vertical.classes)
        self.upper = _mlp(
            position.upper_network[0].in_features + extra, width, 2, len(position.classes)
        )
        self.paddle = _mlp(
            position.paddle_network[0].in_features + extra, width, 2, len(position.classes)
        )
        for head in (self.upper, self.paddle):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.pair.eval()
        return self

    def encode(self, source):
        """Return source-only frozen features and logits for each routed region."""
        position = self.pair.position
        with torch.no_grad():
            upper, paddle, flight = position.regions(source)
            features = {}
            logits = torch.empty((len(source), len(position.classes)), device=source.device)
            if upper.any():
                cells, numbers, occupied = position.upper_encode(source[upper])
                spatial = (position.cell_network(cells) * occupied[..., None]).amax(1)
                features["upper"] = torch.cat([spatial, numbers], dim=1)
                logits[upper] = position.upper_network(features["upper"])
            if paddle.any():
                features["paddle"] = position.paddle_encode(source[paddle])
                logits[paddle] = position.paddle_network(features["paddle"])
            if flight.any():
                logits[flight] = position.flight_network(position.flight_encode(source[flight]))
            vy_logits = position.vertical(source)
            paired_vy = self.pair.coupling(self.pair._coupling_features(source, logits, vy_logits))
            extra = torch.cat([logits.softmax(-1), paired_vy.softmax(-1)], dim=1)
            for name, mask in (("upper", upper), ("paddle", paddle)):
                if mask.any():
                    features[name] = torch.cat([features[name], extra[mask]], dim=1)
        return (upper, paddle, flight), features, logits, vy_logits

    def forward(self, source):
        regions, features, original, vy_logits = self.encode(source)
        y_logits = original.clone()
        for name, mask in zip(("upper", "paddle"), regions[:2]):
            if mask.any():
                y_logits[mask] = original[mask] + getattr(self, name)(features[name])
        velocity = self.pair.coupling(self.pair._coupling_features(source, y_logits, vy_logits))
        return y_logits, velocity

    def predict(self, source):
        y_logits, vy_logits = self(source)
        y = source[:, 1] + source[:, 8] / 8
        return torch.stack(
            [
                y + self.pair.position.classes[y_logits.argmax(-1)],
                self.pair.position.vertical.classes[vy_logits.argmax(-1)],
            ],
            dim=1,
        )
