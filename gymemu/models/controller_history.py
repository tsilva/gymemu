"""Isolated current controller-state inference from observed paddle history."""

import torch
from torch import nn
from torch.nn import functional as F


class ControllerHistoryMLP(nn.Module):
    """One integer target; class values come only from the training split."""

    def __init__(self, history, values, width=128):
        super().__init__()
        if history < 1 or width < 1 or not values or values != sorted(set(values)):
            raise ValueError("Invalid controller history classifier dimensions/classes")
        self.history = history
        self.register_buffer("values", torch.tensor(values, dtype=torch.long))
        self.register_buffer("scales", torch.tensor([160, 160, 16], dtype=torch.float32))
        self.network = nn.Sequential(
            nn.Linear(history * 30, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, len(values)),
        )

    def encode(self, history):
        # x, last-native-frame vx, width, observation validity, preceding action.
        valid = history[..., 3:4]
        scalars = history[..., :3] / self.scales * valid
        integers = history[..., :3].round().long()
        features = [scalars, valid, F.one_hot(history[..., 4].long(), 4).float()]
        for k, (bits, offset) in enumerate(((8, 0), (9, 160), (5, 0))):
            encoded = (
                (integers[..., k, None] + offset) >> torch.arange(bits, device=history.device)
            ) & 1
            features.append((encoded.float() * 2 - 1) * valid)
        return torch.cat(features, -1).flatten(1)

    def forward(self, history):
        return self.network(self.encode(history))

    def predict(self, history):
        return self.values[self(history).argmax(-1)]
