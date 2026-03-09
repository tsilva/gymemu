import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionConditionedResidualBlock(nn.Module):
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


class FrameDynamicsModel(nn.Module):
    def __init__(self, history_length: int, n_actions: int, refine_blocks: int = 0):
        super().__init__()
        in_channels = history_length + n_actions
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 1, kernel_size=3, padding=1),
        )
        self.refine_blocks = nn.ModuleList(
            [
                ActionConditionedResidualBlock(
                    channels=24,
                    n_actions=n_actions,
                    dilation=2 ** (block_idx % 3),
                )
                for block_idx in range(refine_blocks)
            ]
        )

    def forward(self, history_frames: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        height, width = history_frames.shape[-2:]
        action_planes = action[:, :, None, None].expand(-1, -1, height, width)
        x = torch.cat([history_frames, action_planes], dim=1)
        features = self.net[0](x)
        features = self.net[1](features)
        features = self.net[2](features)
        features = self.net[3](features)
        for block in self.refine_blocks:
            features = block(features, action)
        delta = self.net[4](features)
        baseline = history_frames[:, -1:, :, :].clamp(1e-4, 1.0 - 1e-4)
        return torch.logit(baseline) + delta
