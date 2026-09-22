"""Compact-state transition with an absorbing learned life-loss decision."""

import torch
from torch import nn

from gymemu.models.ball_paddle_position import BallPaddlePosition
from gymemu.models.life_termination import LifeTermination


class BallLifeTermination(nn.Module):
    """Emit 117 next-state fields and a stop flag at column 117.

    State fields are zero placeholders when stopped and must not be fed back.
    Passing the previous stop mask keeps those rows stopped without inference.
    """

    def __init__(self, state, termination):
        super().__init__()
        self.state = BallPaddlePosition(**state)
        self.state.requires_grad_(False).eval()
        self.termination = LifeTermination(**termination)

    def train(self, mode=True):
        super().train(mode)
        self.state.eval()
        return self

    def forward(self, source):
        output = self.state(source)
        output["terminated"] = self.termination(source[:, :118])
        return output

    def predict(self, source, terminated=None):
        if source.ndim != 2 or source.shape[1] != 119:
            raise ValueError("Expected 118 state values followed by one provider action")
        if terminated is None:
            terminated = torch.zeros(len(source), dtype=torch.bool, device=source.device)
        if terminated.shape != (len(source),) or terminated.dtype != torch.bool:
            raise ValueError("Expected a boolean previous-stop mask with one value per row")
        output = source.new_zeros((len(source), 118))
        output[:, 117] = 1
        active = torch.where(~terminated)[0]
        if len(active):
            current = source[active]
            self.state.position_inputs(current)  # Validate active action codes even on death.
            stop = self.termination.predict(current[:, :118])
            continuing = active[~stop]
            if len(continuing):
                output[continuing, :117] = self.state.predict(source[continuing])
                output[continuing, 117] = 0
        return output
