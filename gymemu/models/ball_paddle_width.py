"""Add learned paddle width to the ball, bricks, contact and count container."""

import torch
from torch import nn

from gymemu.models.ball_bricks_contact_count import BallBricksContactCount
from gymemu.models.paddle_width import PaddleWidth


class BallPaddleWidth(nn.Module):
    """Append width at output 114; keep the established state branch frozen."""

    def __init__(self, state, paddle_width):
        super().__init__()
        self.state = BallBricksContactCount(**state)
        self.state.requires_grad_(False).eval()
        self.paddle_width = PaddleWidth(**paddle_width)

    def train(self, mode=True):
        super().train(mode)
        self.state.eval()
        return self

    def forward(self, source):
        output = self.state(source)
        output["paddle_width"] = self.paddle_width(source)
        return output

    def predict(self, source):
        return torch.cat(
            [self.state.predict(source), self.paddle_width.predict(source)[:, None]], dim=1
        )
