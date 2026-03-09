import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialActionConditionedResidualBlock(nn.Module):
    def __init__(self, channels: int, n_actions: int, dilation: int):
        super().__init__()
        self.action_bias = nn.Linear(n_actions, channels)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        nn.init.zeros_(self.action_bias.weight)
        nn.init.zeros_(self.action_bias.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_bias = self.action_bias(action).unsqueeze(-1).unsqueeze(-1)
        hidden = F.gelu(self.conv1(x + action_bias))
        hidden = self.conv2(hidden)
        return x + hidden


class SpatialLatentWorldModel(nn.Module):
    def __init__(
        self,
        history_length: int,
        n_actions: int,
        latent_channels: int = 32,
        refine_blocks: int = 4,
    ):
        super().__init__()
        self.history_length = history_length
        self.n_actions = n_actions
        self.latent_channels = latent_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(history_length, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, latent_channels, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.dynamics_in = nn.Sequential(
            nn.Conv2d(latent_channels + n_actions, latent_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.refine_blocks = nn.ModuleList(
            [
                SpatialActionConditionedResidualBlock(
                    channels=latent_channels,
                    n_actions=n_actions,
                    dilation=2 ** (block_idx % 3),
                )
                for block_idx in range(refine_blocks)
            ]
        )
        self.latent_delta = nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.latent_delta.weight)
        nn.init.zeros_(self.latent_delta.bias)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def encode_history(self, history_frames: torch.Tensor) -> torch.Tensor:
        return self.encoder(history_frames)

    def predict_next_latent(
        self, latent_history: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        height, width = latent_history.shape[-2:]
        action_planes = action[:, :, None, None].expand(-1, -1, height, width)
        hidden = self.dynamics_in(torch.cat([latent_history, action_planes], dim=1))
        for block in self.refine_blocks:
            hidden = block(hidden, action)
        delta = self.latent_delta(hidden)
        return latent_history + delta

    def decode_latent(self, latent_state: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_state)

    def forward(self, history_frames: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        latent_history = self.encode_history(history_frames)
        next_latent = self.predict_next_latent(latent_history, action)
        residual_logits = self.decode_latent(next_latent)
        baseline = history_frames[:, -1:, :, :].clamp(1e-4, 1.0 - 1e-4)
        return torch.logit(baseline) + residual_logits
