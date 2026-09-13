"""A frame codec and a separate action-conditioned latent dynamics model."""

import torch
from torch import nn
from torch.nn import functional as F


class FrameCodec(nn.Module):
    def __init__(self, channels, width=32, latent_channels=32):
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, width, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(width, width * 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(width * 2, latent_channels, 4, 2, 1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, width * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(width * 2, width, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(width, channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, frames):
        height, width = frames.shape[-2:]
        return self.encoder(F.pad(frames, (0, (-width) % 8, 0, (-height) % 8)))

    def decode(self, latents, shape):
        return self.decoder(latents)[..., : shape[-2], : shape[-1]]

    def forward(self, frames):
        return self.decode(self.encode(frames), frames.shape)


class LatentDynamics(nn.Module):
    def __init__(self, history, actions, latent_channels, width=64):
        super().__init__()
        self.actions = actions
        self.network = nn.Sequential(
            nn.Conv2d(history * latent_channels + actions + 1, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, latent_channels, 3, padding=1),
        )

    def forward(self, history, action):
        planes = F.one_hot(action, self.actions + 1).to(history.dtype)
        planes = planes[:, :, None, None].expand(-1, -1, *history.shape[-2:])
        return self.network(torch.cat([history.flatten(1, 2), planes], dim=1))
