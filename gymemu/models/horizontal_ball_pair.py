"""Horizontal displacement and velocity with optional shared hidden layers."""

import copy
import math

import torch
from torch import nn

from gymemu.models.ball_position import BallPosition


class HorizontalBallPair(nn.Module):
    """Predict x and vx atomically from the same current 118-field state.

    The independent control duplicates the position hidden layers for velocity.
    The shared variant sends both losses through the same hidden layers. Frozen
    geometry dependencies never train; neither variant calls native game rules.
    """

    def __init__(self, position, velocity_values, shared=True):
        super().__init__()
        if position.get("axis") != "x":
            raise ValueError("Horizontal pair requires an x position model")
        if (
            not velocity_values
            or len(set(velocity_values)) != len(velocity_values)
            or any(not math.isfinite(v) for v in velocity_values)
        ):
            raise ValueError("Expected unique finite horizontal velocities")
        self.position = BallPosition(**position)
        self.shared = shared
        self.register_buffer("velocity_classes", torch.tensor(velocity_values, dtype=torch.float32))
        self.velocity_heads = nn.ModuleDict(
            {
                region: nn.Linear(
                    getattr(self.position, region + "_network")[-1].in_features,
                    len(velocity_values),
                )
                for region in ("upper", "paddle", "flight")
            }
        )
        self.velocity_hidden = nn.ModuleDict()
        if not shared:
            self.velocity_hidden["cell"] = copy.deepcopy(self.position.cell_network)
            for region in ("upper", "paddle", "flight"):
                self.velocity_hidden[region] = copy.deepcopy(
                    getattr(self.position, region + "_network")[:-1]
                )

    def train(self, mode=True):
        super().train(mode)
        self.position.vertical.eval()
        return self

    def forward(self, source):
        masks = self.position.regions(source)
        x_logits = source.new_empty((len(source), len(self.position.classes)))
        vx_logits = source.new_empty((len(source), len(self.velocity_classes)))
        for region, mask in zip(("upper", "paddle", "flight"), masks, strict=True):
            if not mask.any():
                continue
            selected = source[mask]
            network = getattr(self.position, region + "_network")
            if region == "upper":
                cells, numbers, occupied = self.position.upper_encode(selected)
                pooled = (self.position.cell_network(cells) * occupied[..., None]).amax(1)
                encoded = torch.cat([pooled, numbers], 1)
            else:
                encoded = getattr(self.position, region + "_encode")(selected)
            hidden = network[:-1](encoded)
            x_logits[mask] = network[-1](hidden)
            if not self.shared:
                if region == "upper":
                    pooled = (self.velocity_hidden["cell"](cells) * occupied[..., None]).amax(1)
                    encoded = torch.cat([pooled, numbers], 1)
                hidden = self.velocity_hidden[region](encoded)
            vx_logits[mask] = self.velocity_heads[region](hidden)
        return x_logits, vx_logits

    def predict(self, source):
        x_logits, vx_logits = self(source)
        return torch.stack(
            [
                source[:, 0] + self.position.classes[x_logits.argmax(-1)],
                self.velocity_classes[vx_logits.argmax(-1)],
            ],
            dim=1,
        )
