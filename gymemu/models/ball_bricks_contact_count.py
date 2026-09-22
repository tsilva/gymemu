"""Compose ball, bricks, contact and capped paddle-hit count predictors."""

import torch
from torch import nn

from gymemu.models.ball_bricks_contact import BallBricksContact
from gymemu.models.paddle_hit_count import PaddleHitCount


class BallBricksContactCount(nn.Module):
    """Append next capped count to the existing 113-output transition.

    The established state predictor stays frozen. Only the count classifier is
    trained at this integration stage. Every branch reads the same source state.
    """

    def __init__(self, state, count):
        super().__init__()
        self.state = BallBricksContact(**state)
        self.state.requires_grad_(False).eval()
        self.count = PaddleHitCount(**count)

    def train(self, mode=True):
        super().train(mode)
        self.state.eval()
        return self

    def forward(self, source):
        output = self.state(source)
        output["count"] = self.count(source)
        return output

    def predict(self, source):
        return torch.cat([self.state.predict(source), self.count.predict(source)[:, None]], dim=1)
