"""Paddle position alongside the action-conditioned state container."""

import torch
from torch import nn

from gymemu.models.ball_paddle_charge import BallPaddleCharge
from gymemu.models.paddle_transition import PaddleTransitionMLP


class BallPaddlePosition(nn.Module):
    """Append next paddle x at output 116; every branch reads current state."""

    def __init__(self, state, position):
        super().__init__()
        if tuple(position.get("controller_fields", ())) != ("charge",):
            raise ValueError("Position predictor must use only the charge controller field")
        self.state = BallPaddleCharge(**state)
        self.state.requires_grad_(False).eval()
        self.position = PaddleTransitionMLP(**position)

    def train(self, mode=True):
        super().train(mode)
        self.state.eval()
        return self

    @staticmethod
    def position_inputs(source):
        return BallPaddleCharge.charge_inputs(source)

    def forward(self, source):
        inputs = self.position_inputs(source)
        output = self.state(source)
        output["paddle_x"] = self.position(inputs)
        return output

    def predict(self, source):
        inputs = self.position_inputs(source)
        position = source[:, 4] + self.position.delta(self.position(inputs))
        return torch.cat([self.state.predict(source), position[:, None]], dim=1)
