"""Explicit experimental MLP for retained paddle warm-up context."""

import torch
from torch import nn
from torch.nn import functional as F

from gymemu.models.state_dynamics import state_outputs
from gymemu.state_data import STATE_SIZE


class PaddleContextProbe(nn.Module):
    def __init__(
        self,
        history=1,
        past_actions=7,
        width=128,
        depth=2,
        paddle_context=32,
        hidden_size=0,
        predict_paddle_velocity=True,
    ):
        super().__init__()
        if min(history, width, depth, paddle_context) < 1 or past_actions < 0:
            raise ValueError("Invalid context probe dimensions")
        self.history, self.past_actions, self.paddle_context = history, past_actions, paddle_context
        self.predict_paddle_velocity = predict_paddle_velocity
        omitted = int(not predict_paddle_velocity)
        inputs = (
            history * (STATE_SIZE + 1 - omitted)
            + (past_actions + 1) * 4
            + paddle_context * (8 - omitted)
            + hidden_size
        )
        layers = []
        for _ in range(depth):
            layers.extend([nn.Linear(inputs, width), nn.SiLU()])
            inputs = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 116 - omitted)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[-1] = -4

    def forward(self, history, history_valid, actions, paddle_context, hidden_state=None):
        if paddle_context.shape[1:] != (self.paddle_context, 8):
            raise ValueError("Expected paddle scalars, validity, and requested-action context")
        latest = history[:, -1]
        if not self.predict_paddle_velocity:
            history = torch.cat((history[..., :5], history[..., 6:]), -1)
            paddle_context = torch.cat((paddle_context[..., :1], paddle_context[..., 2:]), -1)
        x = torch.cat(
            (
                history.flatten(1),
                history_valid.float(),
                F.one_hot(actions, 4).flatten(1).float(),
                paddle_context.flatten(1),
            ),
            1,
        )
        if hidden_state is not None:
            x = torch.cat((x, hidden_state), 1)
        raw = self.head(self.body(x))
        return state_outputs(raw, latest, self.predict_paddle_velocity), None


class HiddenStateProbe(PaddleContextProbe):
    """Five extra source-state inputs; zeros provide a parameter-matched control."""

    def __init__(self, hidden_mode="both", **kwargs):
        super().__init__(hidden_size=5, **kwargs)
        modes = {
            "none": [0, 0, 0, 0, 0],
            "controller": [1, 1, 1, 1, 0],
            "hits": [0, 0, 0, 0, 1],
            "both": [1, 1, 1, 1, 1],
        }
        if hidden_mode not in modes:
            raise ValueError("Unknown hidden-input mode")
        self.register_buffer("hidden_mask", torch.tensor(modes[hidden_mode], dtype=torch.float32))

    def forward(self, history, history_valid, actions, paddle_context, hidden_state):
        if hidden_state.shape != (len(history), 5):
            raise ValueError("Expected five reconstructed source-state variables")
        return super().forward(
            history, history_valid, actions, paddle_context, hidden_state * self.hidden_mask
        )
