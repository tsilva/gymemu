"""Direct CNN variant conditioned on chronologically ordered executed actions."""

import torch
from torch import nn
from torch.nn import functional as F

from gymemu.models.direct import Autoencoder


class ActionHistoryAutoencoder(Autoencoder):
    def __init__(self, history, actions, shape, width=32, action_history=None):
        action_history = history if action_history is None else action_history
        if type(action_history) is not int or not 1 <= action_history <= history:
            raise ValueError("action_history must be between 1 and the RGB history length")
        super().__init__(history, actions, shape, width)
        self.action_history = action_history
        channels = history * shape[0] + action_history * (actions + 1)
        self.encoder[0] = nn.Conv2d(channels, width, 4, 2, 1)

    def forward(self, history, action):
        batch, _, _, height, width = history.shape
        encoded = F.one_hot(action.reshape(batch, self.action_history), self.actions + 1)
        planes = encoded.flatten(1).to(history.dtype)[:, :, None, None]
        planes = planes.expand(-1, -1, height, width)
        x = torch.cat([history.flatten(1, 2), planes], dim=1)
        x = F.pad(x, (0, (-width) % 8, 0, (-height) % 8))
        return self.decoder(self.encoder(x))[:, :, :height, :width]
