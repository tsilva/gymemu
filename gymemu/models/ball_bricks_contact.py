"""Atomic ball, brick-layout and brick-contact transition."""

import torch
from torch import nn

from gymemu.models.ball_bricks import BallBricks
from gymemu.models.brick_contact import BrickContact


class BallBricksContact(nn.Module):
    """Append next contact to the four ball fields and 108 brick cells.

    All branches read the same source state. Contact retains its own frozen
    layout dependency, so loading the composition preserves parent predictions.
    """

    def __init__(self, state, contact):
        super().__init__()
        self.state = BallBricks(**state)
        self.contact = BrickContact(**contact)

    def forward(self, source):
        output = self.state(source)
        output["contact"] = self.contact(source)
        return output

    def predict(self, source):
        return torch.cat([self.state.predict(source), self.contact.predict(source)[:, None]], dim=1)
