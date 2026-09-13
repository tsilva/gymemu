"""The original direct next-frame CNN, including its checkpoint tensor names."""

import torch
from torch import nn
from torch.nn import functional as F


class Autoencoder(nn.Module):
    """Three strided convolutions and three transposed convolutions; direct RGB output."""

    def __init__(self, history: int, actions: int, shape: tuple, width: int = 32):
        super().__init__()
        self.history, self.actions, self.shape, self.width = history, actions, tuple(shape), width
        channels = history * shape[0] + actions + 1
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, width, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(width, width * 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(width * 2, width * 4, 4, 2, 1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(width * 4, width * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(width * 2, width, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(width, shape[0], 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, history, action):
        batch, _, _, height, width = history.shape
        action = F.one_hot(action, self.actions + 1).to(history.dtype)
        planes = action[:, :, None, None].expand(-1, -1, height, width)
        x = torch.cat([history.flatten(1, 2), planes], dim=1)
        # Pad only to make stride-8 geometry exact; restore the original full canvas.
        x = F.pad(x, (0, (-width) % 8, 0, (-height) % 8))
        return self.decoder(self.encoder(x))[:, :, :height, :width]
