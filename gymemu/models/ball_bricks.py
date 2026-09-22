"""Atomic ball-motion and complete brick-layout transition."""

import torch
from torch import nn

from gymemu.models.ball_motion import BallMotion
from gymemu.models.brick_layout import BrickLayout


class BallBricks(nn.Module):
    """Return x, combined y, vx, vy and 108 brick cells from one source state.

    Both branches read the original state. The brick branch learns no change or
    removal of one occupied cell; wall additions and refills remain unsupported.
    Frozen dependencies retain their existing training/evaluation boundaries.
    """

    def __init__(self, motion, bricks):
        super().__init__()
        self.motion = BallMotion(**motion)
        self.bricks = BrickLayout(**bricks)

    def forward(self, source):
        output = self.motion(source)
        output["bricks"] = self.bricks(source)
        return output

    def predict(self, source):
        return torch.cat([self.motion.predict(source), self.bricks.predict(source)], dim=1)
