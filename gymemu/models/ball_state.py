"""Joint RGB and normalized ball-coordinate prediction from aligned histories."""

import torch
from torch import nn
from torch.nn import functional as F

from gymemu.models.action_history import ActionHistoryAutoencoder


class BallStateAutoencoder(ActionHistoryAutoencoder):
    def __init__(self, history, actions, shape, width=32, action_history=None):
        super().__init__(history, actions, shape, width, action_history)
        self.encoder[0] = nn.Conv2d(
            history * (shape[0] + 3) + self.action_history * (actions + 1), width, 4, 2, 1
        )
        self.coordinates = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(width * 4 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )
        self.coordinate_conditioning = nn.Linear(2, width * 4)

    def forward(self, history, action, state_history):
        batch, _, _, height, width = history.shape
        actions = F.one_hot(action.reshape(batch, self.action_history), self.actions + 1)
        context = torch.cat((actions.flatten(1), state_history.flatten(1)), dim=1)
        planes = context.to(history.dtype)[:, :, None, None].expand(-1, -1, height, width)
        x = torch.cat((history.flatten(1, 2), planes), dim=1)
        encoded = self.encoder(F.pad(x, (0, (-width) % 8, 0, (-height) % 8)))
        coordinates = self.coordinates(encoded)
        conditioned = encoded + self.coordinate_conditioning(coordinates)[:, :, None, None]
        rgb = self.decoder(conditioned)[:, :, :height, :width]
        return rgb, coordinates
