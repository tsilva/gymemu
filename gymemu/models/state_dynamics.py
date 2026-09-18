"""Explicit-state predictors with a common MLP/GRU rollout interface."""

import torch
from torch import nn
from torch.nn import functional as F

from gymemu.state_data import STATE_SIZE


def state_outputs(raw, latest, predict_paddle_velocity=True):
    count = 6 if predict_paddle_velocity else 5
    motion = latest[:, :count] + raw[:, :count]
    if not predict_paddle_velocity:
        # Reserved slot preserves the existing cache/player vector layout.
        motion = F.pad(motion, (0, 1))
    return {
        "motion": motion,
        "width": (0.875 - latest[:, 6]) * 32 + raw[:, count],
        "bricks": (latest[:, 7:] * 2 - 1) * 4 + raw[:, count + 1 : count + 109],
        "terminal": raw[:, count + 109],
    }


class StateDynamics(nn.Module):
    def __init__(
        self,
        history=1,
        past_actions=0,
        width=128,
        depth=2,
        recurrent=False,
        predict_paddle_velocity=True,
    ):
        super().__init__()
        if history < 1 or past_actions < 0 or width < 1 or depth < 1:
            raise ValueError("Invalid state model dimensions")
        self.history, self.past_actions = history, past_actions
        self.recurrent = recurrent
        self.predict_paddle_velocity = predict_paddle_velocity
        state_size = STATE_SIZE - int(not predict_paddle_velocity)
        if recurrent:
            self.body = nn.GRU(state_size + 1 + 4, width, num_layers=depth, batch_first=True)
        else:
            inputs = history * (state_size + 1) + (past_actions + 1) * 4
            layers = []
            for _ in range(depth):
                layers.extend([nn.Linear(inputs, width), nn.SiLU()])
                inputs = width
            self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 116 - int(not predict_paddle_velocity))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[-1] = -4

    def forward(self, history, history_valid, actions, hidden=None):
        """Warm recurrent memory from context, then update once per generated state."""
        latest = history[:, -1]
        if not self.predict_paddle_velocity:
            history = torch.cat((history[..., :5], history[..., 6:]), -1)
        if self.recurrent:
            if hidden is not None:
                sequence = torch.cat(
                    (
                        history[:, -1],
                        history_valid[:, -1, None].float(),
                        F.one_hot(actions[:, -1], 4).float(),
                    ),
                    dim=-1,
                )[:, None]
            else:
                length = max(history.shape[1], actions.shape[1])
                hs = F.pad(
                    torch.cat((history, history_valid[..., None].float()), -1),
                    (0, 0, length - history.shape[1], 0),
                )
                aa = F.pad(actions, (length - actions.shape[1], 0), value=3)
                sequence = torch.cat((hs, F.one_hot(aa, 4).float()), dim=-1)
            features, hidden = self.body(sequence, hidden)
            features = features[:, -1]
        else:
            features = self.body(
                torch.cat(
                    (
                        history.flatten(1),
                        history_valid.float(),
                        F.one_hot(actions, 4).flatten(1).float(),
                    ),
                    -1,
                )
            )
        raw = self.head(features)
        # Residual prediction is learned; the only prior is copying the current state.
        return state_outputs(raw, latest, self.predict_paddle_velocity), hidden


class StateMLP(StateDynamics):
    def __init__(self, **kwargs):
        super().__init__(recurrent=False, **kwargs)


class StateGRU(StateDynamics):
    def __init__(self, **kwargs):
        super().__init__(recurrent=True, **kwargs)


def predicted_state(output):
    """Commit discrete state during training and playback; use STE in training."""
    brick_probability = output["bricks"].sigmoid()
    bricks = brick_probability + ((brick_probability >= 0.5).float() - brick_probability).detach()
    width_probability = output["width"].sigmoid()
    narrow = width_probability + ((width_probability >= 0.5).float() - width_probability).detach()
    return torch.cat((output["motion"], (1 - narrow * 0.25)[:, None], bricks), -1)
