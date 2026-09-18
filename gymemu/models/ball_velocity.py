"""Isolated ball-velocity predictors with explicit, auditable source inputs."""

import torch
from torch import nn


class BallVelocityMLP(nn.Module):
    """Learn velocity from state; no collision or native update rules in forward."""

    def __init__(
        self,
        width=128,
        depth=2,
        encoding="scalar",
        objective="classification",
        memory=True,
        relative_offset=False,
        paddle_only=False,
        spatial_features=False,
    ):
        super().__init__()
        if min(width, depth) < 1:
            raise ValueError("Invalid hidden dimensions")
        if encoding not in ("scalar", "hybrid"):
            raise ValueError("Unknown encoding")
        if objective not in ("classification", "regression", "direction"):
            raise ValueError("Unknown objective")
        self.encoding, self.objective, self.memory = encoding, objective, memory
        self.relative_offset = relative_offset
        self.paddle_only = paddle_only
        self.spatial_features = spatial_features
        self.source_size = 9 if paddle_only else 118
        # x, RAM y, vx, vy, paddle x, width, charge, hits, y fraction, contact.
        self.register_buffer("scales", torch.tensor([160, 255, 2, 3.375, 160, 16, 3856, 12, 7, 1]))
        classes = [-1.0, 1.0] if objective == "direction" else [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]
        self.register_buffer("classes", torch.tensor(classes))
        self.bit_widths = (11, 8, 6, 6, 8, 5, 12, 4, 3, 1)
        self.register_buffer("multipliers", torch.tensor([8, 1, 8, 8, 1, 1, 1, 1, 1, 1]))
        self.register_buffer("offsets", torch.tensor([0, 0, 16, 27, 0, 0, 0, 0, 0, 0]))
        numeric_count = 9 if paddle_only else 10
        inputs = self.source_size
        if encoding == "hybrid":
            inputs += sum(self.bit_widths[:numeric_count])
        if relative_offset:
            inputs += 13 if encoding == "hybrid" else 1
        if spatial_features:
            inputs += 8
        layers = []
        for _ in range(depth):
            layers.extend([nn.Linear(inputs, width), nn.SiLU()])
            inputs = width
        outputs = {"classification": 8, "regression": 1, "direction": 2}
        layers.append(nn.Linear(inputs, outputs[objective]))
        self.network = nn.Sequential(*layers)

    def encode(self, source):
        if source.ndim != 2 or source.shape[1] != self.source_size:
            raise ValueError(f"Expected {self.source_size} source features")
        numeric_count = 9 if self.paddle_only else 10
        numbers = source[:, :numeric_count].clone()
        if not self.memory:
            # Matched architecture control keeps charge but removes collision memory.
            numbers[:, 7:10] = 0
        parts = [numbers / self.scales[:numeric_count]]
        if not self.paddle_only:
            parts.append(source[:, 10:])
        if self.encoding == "hybrid":
            integers = (
                (numbers * self.multipliers[:numeric_count] + self.offsets[:numeric_count])
                .round()
                .long()
            )
            for column, width in enumerate(self.bit_widths[:numeric_count]):
                bits = (integers[:, column, None] >> torch.arange(width, device=source.device)) & 1
                parts.append(bits.float() * 2 - 1)
        if self.relative_offset:
            # A relation between current coordinates, not a collision/update rule.
            delta = source[:, 0] - source[:, 4]
            parts.append(delta[:, None] / 160)
            if self.encoding == "hybrid":
                integer = (delta * 8 + 1280).round().long()
                bits = (integer[:, None] >> torch.arange(12, device=source.device)) & 1
                parts.append(bits.float() * 2 - 1)
        if self.spatial_features:
            # Arithmetic features of current state, not a collision/update rule.
            x, vx, px = source[:, 0], source[:, 2], source[:, 4]
            y = source[:, 1] + numbers[:, 8] / 8
            vy = source[:, 3]
            parts.append(
                torch.stack(
                    [
                        (x - px) / 16,
                        (x + vx - px) / 16,
                        (y - 172) / 16,
                        (y + vy - 172) / 16,
                        x - x.floor(),
                        x + vx - (x + vx).floor(),
                        y - y.floor(),
                        y + vy - (y + vy).floor(),
                    ],
                    dim=1,
                )
            )
        return torch.cat(parts, dim=1)

    def forward(self, source):
        return self.network(self.encode(source))

    def predict(self, source):
        output = self(source)
        if self.objective in ("classification", "direction"):
            return self.classes[output.argmax(-1)]
        return source[:, 2] + output[:, 0]
