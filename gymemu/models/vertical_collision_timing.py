"""Learn the two native-frame vertical velocities as one collision outcome."""

import math
from copy import deepcopy

import torch
from torch import nn

from gymemu.models.vertical_ball_pair import VerticalBallPair


class VerticalCollisionTiming(nn.Module):
    """Decode y from two learned movements, with vy from the second movement.

    The original pair supplies frozen features and ordinary-flight predictions.
    Collision heads classify joint native-frame velocity outcomes. No native
    collision rule or true event is used during inference.
    """

    def __init__(self, pair, active_regions=("upper", "paddle"), prior_weight=0.0):
        super().__init__()
        if not active_regions or any(r not in ("upper", "paddle") for r in active_regions):
            raise ValueError("Expected upper and/or paddle timing regions")
        if not math.isfinite(prior_weight) or prior_weight < 0:
            raise ValueError("Expected a nonnegative finite prior weight")
        self.prior_weight = prior_weight
        self.active_regions = tuple(active_regions)
        self.pair = VerticalBallPair(**pair)
        self.pair.requires_grad_(False).eval()
        position = self.pair.position
        self.register_buffer("speeds", position.vertical.classes.detach().clone())
        sums = (self.speeds[:, None] + self.speeds[None]).flatten()
        matches = sums[:, None] == position.classes[None]
        lookup = torch.where(matches.any(1), matches.long().argmax(1), -1)
        self.register_buffer("displacement_index", lookup, persistent=False)
        classes = len(self.speeds) ** 2
        self.upper = self._head(position.upper_network, classes)
        self.paddle = self._head(position.paddle_network, classes)

    @staticmethod
    def _head(parent, classes):
        layers = [deepcopy(layer) for layer in list(parent)[:-1]]
        layers.append(nn.Linear(parent[-1].in_features, classes))
        head = nn.Sequential(*layers)
        head.requires_grad_(True)
        return head

    def train(self, mode=True):
        super().train(mode)
        self.pair.eval()
        return self

    def encode(self, source):
        position = self.pair.position
        with torch.no_grad():
            upper, paddle, flight = position.regions(source)
            features = {}
            if upper.any():
                cells, numbers, occupied = position.upper_encode(source[upper])
                spatial = (position.cell_network(cells) * occupied[..., None]).amax(1)
                features["upper"] = torch.cat([spatial, numbers], dim=1)
            if paddle.any():
                features["paddle"] = position.paddle_encode(source[paddle])
        return (upper, paddle, flight), features

    def timing_log_probabilities(self, logits, current_vy):
        """Marginal probabilities of no, first, second, or both-frame changes."""
        n = len(self.speeds)
        first = self.speeds.repeat_interleave(n)
        second = self.speeds.repeat(n)
        timing = (first[None] != current_vy[:, None]).long() + 2 * (second != first).long()
        logp = logits.log_softmax(-1)
        return torch.stack(
            [logp.masked_fill(timing != code, -torch.inf).logsumexp(-1) for code in range(4)], dim=1
        )

    def decode(self, logits):
        joint = logits.argmax(-1)
        n = len(self.speeds)
        return torch.stack([self.speeds[joint // n], self.speeds[joint % n]], dim=1)

    def prior_from_logits(self, position_logits):
        """Map frozen displacement probabilities to candidate velocity pairs."""
        logp = position_logits.log_softmax(-1)
        prior = logp[:, self.displacement_index.clamp_min(0)]
        return prior.masked_fill(self.displacement_index[None] < 0, -30.0)

    def predict(self, source):
        regions, features = self.encode(source)
        result = torch.empty((len(source), 2), device=source.device, dtype=source.dtype)
        fallback = regions[2].clone()
        for name, mask in zip(("upper", "paddle"), regions[:2]):
            if name not in self.active_regions:
                fallback |= mask
            elif mask.any():
                logits = getattr(self, name)(features[name])
                if self.prior_weight:
                    with torch.no_grad():
                        original = getattr(self.pair.position, name + "_network")(features[name])
                        prior = self.prior_from_logits(original)
                    logits = logits + self.prior_weight * prior
                velocity = self.decode(logits)
                y = source[mask, 1] + source[mask, 8] / 8
                result[mask] = torch.stack([y + velocity.sum(1), velocity[:, 1]], dim=1)
        if fallback.any():
            with torch.no_grad():
                result[fallback] = self.pair.predict(source[fallback])
        return result
