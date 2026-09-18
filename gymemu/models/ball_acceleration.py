"""Learn rare upper-field speed increases around a frozen horizontal predictor."""

import torch
from torch import nn

from gymemu.models.ball_horizontal_router import BallHorizontalRouter


class BallAcceleration(nn.Module):
    """Classify the observed upper-field outcomes: retain speed or accelerate to 2.

    The source-only domain is RAM y <= 100 and incoming magnitude < 2. Collision
    decisions are learned; event labels and native update rules are not inputs.
    """

    def __init__(self, base, width=128, depth=2, geometry=False):
        super().__init__()
        if min(width, depth) < 1:
            raise ValueError("Invalid acceleration network dimensions")
        self.base = BallHorizontalRouter(**base)
        self.base.requires_grad_(False).eval()
        self.geometry = geometry
        self.register_buffer("columns", torch.tensor([0, 1, 2, 3, 8, 9]))
        self.register_buffer("scales", torch.tensor([160, 255, 2, 3.375, 7, 1]))
        self.register_buffer("multipliers", torch.tensor([8, 1, 8, 8, 1, 1]))
        self.register_buffer("offsets", torch.tensor([0, 0, 16, 27, 0, 0]))
        self.bit_widths = (11, 8, 6, 6, 3, 1)
        inputs = 149 + (30 if geometry else 0)
        layers = []
        for _ in range(depth):
            layers.extend([nn.Linear(inputs, width), nn.ReLU()])
            inputs = width
        layers.append(nn.Linear(inputs, 2))
        self.network = nn.Sequential(*layers)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @staticmethod
    def region(source):
        if source.ndim != 2 or source.shape[1] != 118:
            raise ValueError("Expected 118 source features")
        return (source[:, 1] <= 100) & (source[:, 2].abs() < 2)

    def encode(self, source):
        self.region(source)
        numbers = source[:, self.columns]
        parts = [numbers / self.scales, source[:, 10:]]
        integers = (numbers * self.multipliers + self.offsets).round().long()
        for column, width in enumerate(self.bit_widths):
            bits = (integers[:, column, None] >> torch.arange(width, device=source.device)) & 1
            parts.append(bits.float() * 2 - 1)
        if self.geometry:
            x, x1 = source[:, 0], source[:, 0] + source[:, 2]
            y = source[:, 1] + source[:, 8] / 8
            y1 = y + source[:, 3]
            parts.append(
                torch.stack(
                    [
                        x / 160,
                        x1 / 160,
                        y / 255,
                        y1 / 255,
                        x - x.floor(),
                        x1 - x1.floor(),
                        y - y.floor(),
                        y1 - y1.floor(),
                    ],
                    dim=1,
                )
            )
            for value in (x1, y1):
                integer = (value * 8).round().long()
                bits = (integer[:, None] >> torch.arange(11, device=source.device)) & 1
                parts.append(bits.float() * 2 - 1)
        return torch.cat(parts, dim=1)

    def forward(self, source):
        """Return keep/accelerate logits for the isolated objective."""
        return self.network(self.encode(source))

    def predict(self, source):
        with torch.no_grad():
            prediction = self.base.predict(source)
        gate = self.region(source)
        if gate.any():
            selected = source[gate]
            accelerate = self(selected).argmax(-1).bool()
            speed = torch.where(accelerate, 2.0, selected[:, 2].abs())
            prediction[gate] = prediction[gate].sign() * speed
        return prediction


class BallAccelerationSpatial(BallAcceleration):
    """Share a learned interaction encoder across the 6-by-18 brick grid.

    Brick centers describe the fixed layout in source-coordinate units. Relative
    positions are features only: no overlap, collision, or speedup rules run here.
    """

    def __init__(self, base, width=64, depth=2, head_width=128):
        super().__init__(base, width=width, depth=depth)
        if head_width < 1:
            raise ValueError("Invalid acceleration head dimensions")
        row = torch.arange(6).repeat_interleave(18).float()
        column = torch.arange(18).repeat(6).float()
        self.register_buffer("cell_row", row / 5)
        self.register_buffer("cell_column", column / 17)
        self.register_buffer("cell_x", 11.5 + 8 * column)
        # The recorded RAM y is nine pixels above the native screen coordinate.
        self.register_buffer("cell_y", 50.5 + 6 * row)
        layers = []
        inputs = 13
        for _ in range(depth):
            layers.extend([nn.Linear(inputs, width), nn.ReLU()])
            inputs = width
        self.cell_network = nn.Sequential(*layers)
        self.network = nn.Sequential(
            nn.Linear(width + 6, head_width), nn.ReLU(), nn.Linear(head_width, 2)
        )

    def encode(self, source):
        self.region(source)
        numbers = source[:, self.columns] / self.scales
        x, x1 = source[:, 0], source[:, 0] + source[:, 2]
        y = source[:, 1] + source[:, 8] / 8
        y1 = y + source[:, 3]

        def broadcast(value):
            return value[:, None].expand(-1, 108)

        cells = torch.stack(
            [
                (x[:, None] - self.cell_x) / 8,
                (x1[:, None] - self.cell_x) / 8,
                (y[:, None] - self.cell_y) / 6,
                (y1[:, None] - self.cell_y) / 6,
                broadcast(source[:, 2] / 2),
                broadcast(source[:, 3] / 3.375),
                broadcast(source[:, 9]),
                self.cell_row.expand(len(source), -1),
                self.cell_column.expand(len(source), -1),
                broadcast(x - x.floor()),
                broadcast(x1 - x1.floor()),
                broadcast(y - y.floor()),
                broadcast(y1 - y1.floor()),
            ],
            dim=-1,
        )
        return cells, numbers, source[:, 10:].bool()

    def forward_encoded(self, cells, numbers, occupied):
        features = self.cell_network(cells)
        features = features * occupied[..., None]
        pooled = features.amax(dim=1)
        return self.network(torch.cat([pooled, numbers], dim=1))

    def forward(self, source):
        return self.forward_encoded(*self.encode(source))
