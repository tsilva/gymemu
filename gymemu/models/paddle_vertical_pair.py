"""Factor paddle outcomes into learned bounce timing and outgoing velocity."""

import math
from copy import deepcopy

import torch
from torch import nn

from gymemu.models.vertical_ball_pair import VerticalBallPair


class PaddleVerticalPair(nn.Module):
    """Replace only the descending paddle region of a frozen vertical pair.

    Timing classes are no bounce, first native movement, and second movement.
    A separate head selects the negative outgoing velocity on bounce examples.
    Optional signed raster-edge distances use only the existing source state.
    Neither collision rules nor targets execute
    during inference; two learned decisions determine one consistent y/vy pair.
    """

    def __init__(self, pair, bounce_values, edge_features=False):
        super().__init__()
        if (
            not bounce_values
            or len(set(bounce_values)) != len(bounce_values)
            or any(not math.isfinite(v) or v >= 0 or v * 8 != round(v * 8) for v in bounce_values)
        ):
            raise ValueError("Expected unique negative eighth-pixel bounce velocities")
        self.pair = VerticalBallPair(**pair)
        self.pair.requires_grad_(False).eval()
        self.edge_features = edge_features
        parent = self.pair.position.paddle_network
        self.trunk = nn.Sequential(*[deepcopy(layer) for layer in list(parent)[:-1]])
        if edge_features:
            # Preserve the random stream used by the control's output heads.
            with torch.random.fork_rng(devices=[]):
                self.trunk[0] = nn.Linear(parent[0].in_features + 8, parent[0].out_features)
            self.initialize_trunk_from_pair()
        self.trunk.requires_grad_(True)
        self.timing = nn.Linear(parent[-1].in_features, 3)
        self.speed = nn.Linear(parent[-1].in_features, len(bounce_values))
        self.register_buffer("bounce_values", torch.tensor(bounce_values, dtype=torch.float32))

    def train(self, mode=True):
        super().train(mode)
        self.pair.eval()
        return self

    def encode(self, source):
        with torch.no_grad():
            features = self.pair.position.paddle_encode(source)
            if self.edge_features:
                features = torch.cat([features, self.edge_encode(source)], dim=1)
            return features

    def initialize_trunk_from_pair(self):
        """Copy the loaded parent; extra input columns initially contribute zero."""
        parent = self.pair.position.paddle_network
        with torch.no_grad():
            for layer, original in zip(self.trunk, list(parent)[:-1], strict=True):
                if isinstance(layer, nn.Linear):
                    layer.weight.zero_()
                    layer.weight[:, : original.in_features].copy_(original.weight)
                    layer.bias.copy_(original.bias)

    def edge_encode(self, source):
        """Signed edge distances, not collision decisions or executed movement.

        In RAM y coordinates the paddle spans 180..183 and the ball raster
        extends one pixel right and three pixels below its integer position.
        The second geometry uses a constant-velocity ball proposal and the
        frozen learned intermediate-paddle predictor. No reflection is applied.
        """
        direction = self.pair.position.vertical.horizontal.base.paddle_model.direction
        with torch.no_grad():
            px1 = direction.intermediate_paddle(source[:, :9])
        y = source[:, 1] + source[:, 8] / 8
        geometry = []
        for x, yy, px in (
            (source[:, 0], y, source[:, 4]),
            (source[:, 0] + source[:, 2], y + source[:, 3], px1),
        ):
            xi, yi = x.floor(), yy.floor()
            geometry.extend([xi + 1 - px, px + source[:, 5] - 1 - xi, yi + 3 - 180, 183 - yi])
        return torch.stack(geometry, dim=1) / 16

    def forward_encoded(self, features):
        hidden = self.trunk(features)
        return self.timing(hidden), self.speed(hidden)

    def decode(self, source, timing_logits, speed_logits):
        timing = timing_logits.argmax(-1)
        bounce = self.bounce_values[speed_logits.argmax(-1)]
        first = torch.where(timing == 1, bounce, source[:, 3])
        second = torch.where(timing == 0, source[:, 3], bounce)
        current_y = source[:, 1] + source[:, 8] / 8
        return torch.stack([current_y + first + second, second], dim=1)

    def predict(self, source):
        paddle = self.pair.position.regions(source)[1]
        result = source.new_empty((len(source), 2))
        if paddle.any():
            logits = self.forward_encoded(self.encode(source[paddle]))
            result[paddle] = self.decode(source[paddle], *logits)
        if (~paddle).any():
            with torch.no_grad():
                result[~paddle] = self.pair.predict(source[~paddle])
        return result
