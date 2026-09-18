"""Full-RGB paddle probe using exactly the reconstruction FrameCodec encoder."""

import torch
from torch import nn
from torch.nn import functional as F

from gymemu.models.latent import FrameCodec


class PaddleStateCNN(nn.Module):
    """Shared single-frame CNN followed by a temporal/action regression head."""

    def __init__(
        self,
        history=1,
        action_history=4,
        width=32,
        latent_channels=32,
        hidden=128,
        encoder_amp=True,
        outputs=2,
    ):
        super().__init__()
        self.history, self.action_history = history, action_history
        self.encoder_amp = encoder_amp
        self.encoder = FrameCodec(3, width, latent_channels).encoder
        self.frame_head = nn.Sequential(
            nn.Flatten(), nn.Linear(latent_channels * 27 * 20, hidden), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(history * hidden + action_history * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, outputs),
        )

    def encode_history(self, frames):
        batch = frames.shape[0]
        rgb = frames.flatten(0, 1).float().div(255)
        rgb = rgb.contiguous(memory_format=torch.channels_last)
        # Match FrameCodec.encode exactly: full 210x160 RGB, bottom padding to 216.
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=frames.is_cuda and self.training and self.encoder_amp,
        ):
            latent = self.encoder(F.pad(rgb, (0, 0, 0, 6)))
        # Keep the scalar output path in float32 for subpixel numerical precision.
        return self.frame_head(latent.float()).reshape(batch, self.history, -1)

    def forward(self, frames, actions):
        visual = self.encode_history(frames).flatten(1)
        action = F.one_hot(actions, 4).flatten(1).float()
        return self.head(torch.cat([visual, action], dim=1))


class PaddlePositionCNN(PaddleStateCNN):
    """Learn visible position from full RGB, then predict state from its distribution.

    The FrameCodec encoder is unchanged. Position labels are auxiliary training
    targets only. Inference consumes RGB and actions, and gradients flow through
    the position probabilities into the encoder.
    """

    def __init__(
        self,
        history=1,
        action_history=4,
        width=32,
        latent_channels=32,
        hidden=128,
        encoder_amp=True,
    ):
        super().__init__(history, action_history, width, latent_channels, hidden, encoder_amp)
        self.position = nn.Linear(hidden, 160)
        self.head = nn.Sequential(
            nn.Linear(history * 160 + action_history * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )
        self.register_buffer("coordinates", torch.arange(160, dtype=torch.float32) / 160)

    def forward_with_positions(self, frames, actions):
        logits = self.position(self.encode_history(frames))
        probabilities = logits.softmax(-1)
        return self.predict_from_positions(probabilities, actions), logits

    def predict_from_positions(self, probabilities, actions):
        action = F.one_hot(actions, 4).flatten(1).float()
        dynamics = self.head(torch.cat([probabilities.flatten(1), action], dim=1)) / 16
        current_x = (probabilities[:, -1] * self.coordinates).sum(-1)
        prediction = torch.stack([current_x + dynamics[:, 0], dynamics[:, 1]], dim=-1)
        return prediction

    def forward(self, frames, actions):
        return self.forward_with_positions(frames, actions)[0]
