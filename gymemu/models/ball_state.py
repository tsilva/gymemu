"""Joint RGB and normalized ball-coordinate prediction from aligned histories."""

import torch
from torch import nn
from torch.nn import functional as F

from gymemu.models.action_history import ActionHistoryAutoencoder


class CoordinatePool(nn.AdaptiveAvgPool2d):
    """Preserve adaptive pooling bins on MPS for non-divisible spatial sizes."""

    def __init__(self):
        super().__init__((4, 4))

    def forward(self, value):
        height, width = value.shape[-2:]
        if value.device.type != "mps" or (height % 4 == 0 and width % 4 == 0):
            return super().forward(value)
        # Adaptive bins use floor(start) and ceil(end), including overlapping bins.
        # Padding or resizing here would change the existing checkpoint's predictions.
        bins = [
            value[
                ...,
                row * height // 4 : ((row + 1) * height + 3) // 4,
                col * width // 4 : ((col + 1) * width + 3) // 4,
            ].mean(dim=(-2, -1))
            for row in range(4)
            for col in range(4)
        ]
        return torch.stack(bins, dim=-1).reshape(*value.shape[:-2], 4, 4)


class BallStateAutoencoder(ActionHistoryAutoencoder):
    def __init__(self, history, actions, shape, width=32, action_history=None):
        super().__init__(history, actions, shape, width, action_history)
        self.encoder[0] = nn.Conv2d(
            history * (shape[0] + 3) + self.action_history * (actions + 1), width, 4, 2, 1
        )
        self.coordinates = nn.Sequential(
            CoordinatePool(),
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
