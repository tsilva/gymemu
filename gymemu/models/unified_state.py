"""One shared residual MLP for the complete compact Breakout transition."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )

    def forward(self, value):
        return value + self.layers(value)


class UnifiedStateMLP(nn.Module):
    """Share every hidden layer; split one final affine layer into class logits.

    Vocabulary is fixed from training labels. Encoding uses only current state
    and action, with scalar/binary features and no learned helper networks.
    Decoding applies predicted displacements and a single brick-removal event.
    Terminal successors are placeholders and must never be fed back.
    """

    names = (
        "dx",
        "dy",
        "vx",
        "vy",
        "bricks",
        "contact",
        "count",
        "width",
        "charge",
        "paddle_x",
        "terminated",
    )
    vocabulary_names = ("dx", "dy", "vx", "vy", "charge", "paddle_x")
    bit_widths = (11, 11, 6, 6, 8, 5, 12, 4, 3, 1)

    def __init__(self, vocabulary, width=512, blocks=6):
        super().__init__()
        if width < 1 or blocks < 1 or set(vocabulary) != set(self.vocabulary_names):
            raise ValueError("Expected positive dimensions and all six transition vocabularies")
        for name in self.vocabulary_names:
            values = vocabulary[name]
            quantum = 8 if name in ("dx", "dy", "vx", "vy") else 1
            if (
                not values
                or values != sorted(set(values))
                or any(not math.isfinite(v) or v * quantum != round(v * quantum) for v in values)
            ):
                raise ValueError(f"Invalid discrete vocabulary: {name}")
            self.register_buffer("values_" + name, torch.tensor(values, dtype=torch.float32))
        sizes = {name: len(values) for name, values in vocabulary.items()}
        sizes.update(bricks=109, contact=2, count=2, width=2, terminated=2)
        self.sizes = tuple(sizes[name] for name in self.names)
        self.input_features = 121 + sum(self.bit_widths)
        self.register_buffer("scales", torch.tensor([160, 255, 2, 3.375, 160, 16, 3856, 12, 8, 1]))
        self.trunk = nn.Sequential(
            nn.Linear(self.input_features, width),
            nn.SiLU(),
            *(ResidualBlock(width) for _ in range(blocks)),
            nn.LayerNorm(width),
        )
        self.output = nn.Linear(width, sum(self.sizes))

    def encode(self, source):
        if source.ndim != 2 or source.shape[1] != 119:
            raise ValueError("Expected 118 current state values and one provider action")
        action = source[:, 118]
        if not torch.isfinite(source).all() or not torch.all(
            (action == action.round()) & (action >= 0) & (action <= 2)
        ):
            raise ValueError("Expected finite state and integer provider action 0, 1 or 2")
        parts = [
            source[:, :10] / self.scales,
            source[:, 10:118] * 2 - 1,
            F.one_hot(action.long(), 3).to(source.dtype),
        ]
        integers = source[:, :10].clone()
        integers[:, 0] *= 8
        integers[:, 1] = source[:, 1] * 8 + source[:, 8]
        integers[:, 2] = source[:, 2] * 8 + 16
        integers[:, 3] = source[:, 3] * 8 + 27
        integers = integers.round().long()
        for column, bits in enumerate(self.bit_widths):
            value = integers[:, column, None] >> torch.arange(bits, device=source.device)
            parts.append((value & 1).to(source.dtype) * 2 - 1)
        return torch.cat(parts, dim=1)

    def forward_encoded(self, features):
        return dict(zip(self.names, self.output(self.trunk(features)).split(self.sizes, 1)))

    def forward(self, source):
        return self.forward_encoded(self.encode(source))

    def decode(self, source, logits):
        """No collision oracle: all events and movement classes are learned."""
        classes = {name: values.argmax(1) for name, values in logits.items()}
        value = {
            name: getattr(self, "values_" + name)[classes[name]] for name in self.vocabulary_names
        }
        bricks = source[:, 10:118].clone()
        # An absent cell cannot be removed; keep the established event contract.
        brick_logits = logits["bricks"].clone()
        brick_logits[:, 1:] = brick_logits[:, 1:].masked_fill(bricks <= 0, -torch.inf)
        removal = brick_logits.argmax(1)
        selected = torch.where(removal > 0)[0]
        bricks[selected, removal[selected] - 1] = 0
        state = torch.cat(
            [
                torch.stack(
                    [
                        source[:, 0] + value["dx"],
                        source[:, 1] + source[:, 8] / 8 + value["dy"],
                        value["vx"],
                        value["vy"],
                    ],
                    1,
                ),
                bricks,
                torch.stack(
                    [
                        classes["contact"],
                        (source[:, 7] + classes["count"]).clamp(0, 12),
                        12 + 4 * classes["width"],
                        source[:, 6] + value["charge"],
                        source[:, 4] + value["paddle_x"],
                    ],
                    1,
                ),
            ],
            1,
        )
        stop = classes["terminated"].bool()
        state = state.masked_fill(stop[:, None], 0)
        return torch.cat([state, stop[:, None].to(state.dtype)], 1)

    def predict(self, source, terminated=None):
        if source.ndim != 2 or source.shape[1] != 119:
            raise ValueError("Expected 118 current state values and one provider action")
        if terminated is None:
            terminated = torch.zeros(len(source), dtype=torch.bool, device=source.device)
        if terminated.shape != (len(source),) or terminated.dtype != torch.bool:
            raise ValueError("Expected one boolean previous-stop mask per row")
        result = source.new_zeros((len(source), 118))
        result[:, 117] = 1
        active = torch.where(~terminated)[0]
        if len(active):
            current = source[active]
            result[active] = self.decode(current, self(current))
        return result
