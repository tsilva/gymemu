"""Atomic ball-state transition assembled from learned horizontal/vertical pairs."""

import torch
from torch import nn

from gymemu.models.horizontal_ball_pair import HorizontalBallPair
from gymemu.models.paddle_vertical_pair import PaddleVerticalPair


class BallMotion(nn.Module):
    """Predict x, combined y, vx and vy from one current 118-field state.

    Specialized branches remain separate inside one registered model. Horizontal
    heads, vertical upper heads and vertical paddle heads can train together;
    geometry dependencies, vertical flight heads and velocity correction stay
    frozen. No branch sees another branch's newly predicted state.
    """

    def __init__(self, horizontal, vertical):
        super().__init__()
        self.horizontal = HorizontalBallPair(**horizontal)
        self.vertical = PaddleVerticalPair(**vertical)
        for module in self.vertical_upper_modules():
            module.requires_grad_(True)

    def vertical_upper_modules(self):
        position = self.vertical.pair.position
        return (
            position.cell_network,
            position.upper_network,
            position.vertical.cell_network,
            position.vertical.upper_network,
        )

    def train(self, mode=True):
        super().train(mode)
        for module in self.vertical_upper_modules():
            module.train(mode)
        return self

    def forward(self, source):
        """Return trainable logits and their source-only routing masks."""
        upper, paddle, _ = self.horizontal.position.regions(source)
        position = self.vertical.pair.position
        outputs = dict(horizontal=self.horizontal(source), upper_mask=upper, paddle_mask=paddle)
        if upper.any():
            encoded = position.upper_encode(source[upper])
            outputs["upper"] = (
                position.upper_forward_encoded(*encoded),
                position.vertical.upper_forward_encoded(*encoded),
            )
        if paddle.any():
            outputs["paddle"] = self.vertical.forward_encoded(self.vertical.encode(source[paddle]))
        return outputs

    def predict(self, source):
        horizontal = self.horizontal.predict(source)
        vertical = self.vertical.predict(source)
        return torch.stack(
            [horizontal[:, 0], vertical[:, 0], horizontal[:, 1], vertical[:, 1]], dim=1
        )
