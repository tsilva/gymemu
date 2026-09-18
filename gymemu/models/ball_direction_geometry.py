"""Direction probe with a learned intermediate-paddle predictor."""

import torch
from torch import nn

from gymemu.models.paddle_transition import PaddleTransitionMLP


class BallDirectionGeometry(nn.Module):
    """Both predictions are learned; encoding contains no collision decisions."""

    def __init__(self, paddle_values, width=128, depth=2, encoding="scalar", activation="relu"):
        super().__init__()
        if encoding not in ("scalar", "hybrid", "hybrid_absolute") or activation not in (
            "relu",
            "silu",
        ):
            raise ValueError("Unknown geometry encoding or activation")
        if min(width, depth) < 1:
            raise ValueError("Invalid hidden dimensions")
        self.encoding = encoding
        self.paddle = PaddleTransitionMLP(
            encoding="hybrid",
            objective="classification",
            controller_fields=("charge",),
            delta_values=paddle_values,
        )
        self.paddle.requires_grad_(False).eval()
        self.register_buffer("classes", torch.tensor([-1.0, 1.0]))
        inputs = {"scalar": 16, "hybrid": 62, "hybrid_absolute": 84}[encoding]
        layers = []
        for _ in range(depth):
            layers.extend(
                [nn.Linear(inputs, width), nn.ReLU() if activation == "relu" else nn.SiLU()]
            )
            inputs = width
        layers.append(nn.Linear(inputs, 2))
        self.network = nn.Sequential(*layers)

    def train(self, mode=True):
        super().train(mode)
        self.paddle.eval()
        return self

    @staticmethod
    def paddle_inputs(source):
        values = source.new_zeros((len(source), 6))
        values[:, 0] = source[:, 4]
        values[:, 1] = source[:, 6]
        # Constant placeholder action: intermediate position uses current controller state.
        return values

    def intermediate_paddle(self, source):
        with torch.no_grad():
            return source[:, 4] + self.paddle.delta(self.paddle(self.paddle_inputs(source)))

    def encode(self, source):
        if source.ndim != 2 or source.shape[1] != 9:
            raise ValueError("Expected 9 source features")
        x, vx, px = source[:, 0], source[:, 2], source[:, 4]
        y, vy = source[:, 1] + source[:, 8] / 8, source[:, 3]
        px1 = self.intermediate_paddle(source)
        # Constant-velocity proposals are features, not an executed game transition.
        x1, y1 = x + vx, y + vy
        features = [
            torch.stack(
                [
                    x / 160,
                    x1 / 160,
                    (y - 172) / 16,
                    (y1 - 172) / 16,
                    vx / 2,
                    vy / 3.375,
                    source[:, 5] / 16,
                    source[:, 7] / 12,
                    (x - px) / 16,
                    (x1 - px1) / 16,
                    (x.floor() - px) / 16,
                    (x1.floor() - px1) / 16,
                    (y.floor() - 172) / 16,
                    (y1.floor() - 172) / 16,
                    (px1 - px) / 8,
                    px / 160,
                ],
                dim=1,
            )
        ]
        if self.encoding != "scalar":
            integers = [
                ((x - px) * 8 + 1280).round().long(),
                ((x1 - px1) * 8 + 1280).round().long(),
                (y * 8).round().long(),
                (y1 * 8).round().long(),
            ]
            widths = [12, 12, 11, 11]
            if self.encoding == "hybrid_absolute":
                integers.extend([(x * 8).round().long(), (x1 * 8).round().long()])
                widths.extend([11, 11])
            for values, bits in zip(integers, widths, strict=True):
                features.append(
                    ((values[:, None] >> torch.arange(bits, device=source.device)) & 1).float() * 2
                    - 1
                )
        return torch.cat(features, dim=1)

    def forward(self, source):
        return self.network(self.encode(source))

    def predict(self, source):
        return self.classes[self(source).argmax(-1)]
