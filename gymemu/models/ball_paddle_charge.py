"""Action-conditioned charge update alongside the existing state container."""

import torch
from torch import nn

from gymemu.models.ball_paddle_width import BallPaddleWidth
from gymemu.models.paddle_transition import PaddleTransitionMLP


class BallPaddleCharge(nn.Module):
    """Read 118 state values plus provider action; append next charge at 115.

    Existing predictors receive only their original state contract and stay
    frozen. Charge uses current paddle x, charge and action, never a successor.
    """

    def __init__(self, state, charge):
        super().__init__()
        if tuple(charge.get("controller_fields", ())) != ("charge",):
            raise ValueError("Charge predictor must use only the charge controller field")
        self.state = BallPaddleWidth(**state)
        self.state.requires_grad_(False).eval()
        self.charge = PaddleTransitionMLP(**charge)

    def train(self, mode=True):
        super().train(mode)
        self.state.eval()
        return self

    @staticmethod
    def charge_inputs(source):
        if source.ndim != 2 or source.shape[1] != 119:
            raise ValueError("Expected 118 state values followed by one provider action")
        action = source[:, 118]
        if not torch.all(
            torch.isfinite(action) & (action == action.round()) & (action >= 0) & (action <= 2)
        ):
            raise ValueError("Expected integer provider actions 0, 1 or 2")
        inputs = source.new_zeros((len(source), 6))
        inputs[:, 0] = source[:, 4]
        inputs[:, 1] = source[:, 6]
        inputs[:, 5] = action
        return inputs

    def forward(self, source):
        inputs = self.charge_inputs(source)
        output = self.state(source[:, :118])
        output["charge"] = self.charge(inputs)
        return output

    def predict(self, source):
        inputs = self.charge_inputs(source)
        charge = source[:, 6] + self.charge.delta(self.charge(inputs))
        return torch.cat([self.state.predict(source[:, :118]), charge[:, None]], dim=1)
