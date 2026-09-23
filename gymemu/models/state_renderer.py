"""Coordinate-conditioned neural renderer for compact native Breakout state."""

import torch
from torch import nn


class StateRenderer(nn.Module):
    """Learn per-pixel palette logits using shared spatial features.

    Inputs are ball x, integer RAM y, paddle x, width, and 108 brick cells.
    Fixed geometry places state features; RGB colors and sprite visibility are
    learned. No input image, dynamics prediction, or simulator renderer is used.
    """

    def __init__(self, palette, width=64, layers=3, hud_height=17):
        super().__init__()
        colors = torch.tensor(palette, dtype=torch.float32)
        if colors.ndim != 2 or colors.shape[1] != 3 or len(colors) < 2:
            raise ValueError("Expected at least two RGB palette entries")
        if not torch.isfinite(colors).all() or ((colors < 0) | (colors > 255)).any():
            raise ValueError("Palette values must be finite RGB bytes")
        if width < 1 or layers < 1 or not 0 <= hud_height < 210:
            raise ValueError("Invalid renderer dimensions")
        self.register_buffer("palette", colors / 255)
        self.hud_height = hud_height
        self.input_features = 20
        blocks = [nn.Linear(self.input_features, width), nn.SiLU()]
        for _ in range(layers - 1):
            blocks.extend([nn.Linear(width, width), nn.SiLU()])
        blocks.append(nn.Linear(width, len(colors)))
        self.network = nn.Sequential(*blocks)

    def encode(self, state, coordinates):
        if state.ndim != 2 or state.shape[1] != 112:
            raise ValueError("Expected N-by-112 visual states in physical units")
        if coordinates.ndim != 3 or coordinates.shape[0] != len(state):
            raise ValueError("Expected N-by-P-by-2 pixel coordinates")
        if coordinates.shape[2] != 2:
            raise ValueError("Pixel coordinates must contain x and y")
        x, y = coordinates.unbind(-1)
        row = ((y - 57) // 6).long().clamp(0, 5)
        col = ((x - 8) // 8).long().clamp(0, 17)
        occupancy = state[:, 4:].gather(1, row * 18 + col)
        return torch.stack(
            [
                x / 160,
                y / 210,
                (x - 8).clamp(-8, 8) / 8,
                (x - 152).clamp(-8, 8) / 8,
                (y - 17).clamp(-8, 8) / 8,
                (y - 32).clamp(-8, 8) / 8,
                (y - 57).clamp(-8, 8) / 8,
                (y - 93).clamp(-8, 8) / 8,
                (y - 189).clamp(-8, 8) / 8,
                (y - 193).clamp(-8, 8) / 8,
                (y - 196).clamp(-8, 8) / 8,
                (x - state[:, 0, None].floor()).clamp(-8, 8) / 8,
                (y - state[:, 1, None] - 9).clamp(-8, 8) / 8,
                (x - state[:, 2, None]).clamp(-20, 20) / 20,
                (x - state[:, 2, None] - state[:, 3, None]).clamp(-20, 20) / 20,
                state[:, 3, None].expand_as(x) / 16,
                occupancy,
                row.to(state.dtype) / 5,
                (x - 8).remainder(8) / 8,
                (y - 57).remainder(6) / 6,
            ],
            dim=-1,
        )

    def forward(self, state, coordinates):
        return self.network(self.encode(state, coordinates))

    def render(self, state):
        """Return float32 RGB NCHW with unsupported HUD pixels masked to black."""
        y, x = torch.meshgrid(
            torch.arange(210, device=state.device),
            torch.arange(160, device=state.device),
            indexing="ij",
        )
        coordinates = torch.stack([x, y], -1).reshape(1, -1, 2).expand(len(state), -1, -1)
        labels = self(state, coordinates).argmax(-1)
        rgb = self.palette[labels].reshape(-1, 210, 160, 3).permute(0, 3, 1, 2)
        rgb[:, :, : self.hud_height] = 0
        return rgb

    @staticmethod
    def from_dynamics(prediction):
        """Select visual fields from nonterminal unified-dynamics outputs."""
        if prediction.ndim != 2 or prediction.shape[1] != 118:
            raise ValueError("Expected N-by-118 unified dynamics predictions")
        if prediction[:, 117].ne(0).any():
            raise ValueError("Filter terminal predictions before rendering")
        return torch.cat(
            [
                prediction[:, 0:1],
                prediction[:, 1:2].floor(),
                prediction[:, 116:117],
                prediction[:, 114:115],
                prediction[:, 4:112],
            ],
            dim=1,
        )
