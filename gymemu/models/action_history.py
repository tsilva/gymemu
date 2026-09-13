"""Direct CNN variant conditioned on chronologically ordered executed actions."""

import torch
from torch import nn
from torch.nn import functional as F

from gymemu.models.direct import Autoencoder


class ActionHistoryAutoencoder(Autoencoder):
    def __init__(
        self, history, actions, shape, width=32, action_history=None, factor_actions=False
    ):
        action_history = history if action_history is None else action_history
        if type(action_history) is not int or not 1 <= action_history <= history:
            raise ValueError("action_history must be between 1 and the RGB history length")
        super().__init__(history, actions, shape, width)
        self.action_history = action_history
        if type(factor_actions) is not bool:
            raise ValueError("factor_actions must be boolean")
        self.factor_actions = factor_actions
        channels = history * shape[0] + action_history * (actions + 1)
        self.encoder[0] = nn.Conv2d(channels, width, 4, 2, 1)

    def forward(self, history, action):
        # Keep evaluation/playback on the original float32 arithmetic path.
        if self.factor_actions and torch.is_autocast_enabled(history.device.type):
            return self.factored_forward(history, action)
        batch, _, _, height, width = history.shape
        encoded = F.one_hot(action.reshape(batch, self.action_history), self.actions + 1)
        planes = encoded.flatten(1).to(history.dtype)[:, :, None, None]
        planes = planes.expand(-1, -1, height, width)
        x = torch.cat([history.flatten(1, 2), planes], dim=1)
        x = F.pad(x, (0, (-width) % 8, 0, (-height) % 8))
        return self.decoder(self.encoder(x))[:, :, :height, :width]

    def factored_forward(self, history, action):
        """Apply constant action planes analytically, retaining the original weights.

        The action contribution is a sum of 4x4 kernels selected by each token.
        Row/column coverage accounts for zero padding, including the extra rows
        needed by the stride-8 encoder. No full-resolution action tensor is built.
        """
        batch, _, channels, height, width = history.shape
        rgb = F.pad(history.flatten(1, 2), (0, (-width) % 8, 0, (-height) % 8))
        first = self.encoder[0]
        split = self.history * channels
        features = F.conv2d(rgb, first.weight[:, :split].contiguous(), None, 2, 1)
        encoded = F.one_hot(action.reshape(batch, self.action_history), self.actions + 1)
        encoded = encoded.flatten(1).to(history.dtype)
        weights = first.weight[:, split:].permute(0, 2, 3, 1).reshape(-1, encoded.shape[-1])
        kernel = F.linear(encoded, weights).reshape(batch, self.width, 4, 4)
        offsets = torch.arange(4, device=history.device)
        rows = torch.arange(features.shape[-2], device=history.device)[:, None] * 2 - 1 + offsets
        cols = torch.arange(features.shape[-1], device=history.device)[:, None] * 2 - 1 + offsets
        rows = ((rows >= 0) & (rows < height)).to(kernel.dtype)
        cols = ((cols >= 0) & (cols < width)).to(kernel.dtype)
        contribution = torch.einsum("bokc,hk->bohc", kernel, rows)
        contribution = torch.einsum("bohc,wc->bohw", contribution, cols)
        features = features + contribution + first.bias[None, :, None, None]
        return self.decoder(self.encoder[1:](features))[:, :, :height, :width]
