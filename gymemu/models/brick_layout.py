"""Learn a coherent brick-layout update from the audited current game state."""

import copy

import torch
from torch import nn

from gymemu.models.ball_position import BallPosition
from gymemu.models.ball_vertical_velocity import _mlp


class BrickLayout(nn.Module):
    """Choose no change or one occupied brick to remove.

    This output contract requires an audit establishing at most one removal and
    no additions per transition. It does not represent wall refills. The network
    chooses the event and cell; no native collision rules run during inference.
    """

    def __init__(self, position, width=64, depth=2, keep_width=128, keep_depth=2):
        super().__init__()
        self.position = BallPosition(**position)
        self.position.requires_grad_(False).eval()
        self.cell_network = copy.deepcopy(self.position.cell_network)
        self.cell_network.requires_grad_(True)
        cell_width = self.cell_network[-2].out_features
        self.removal_network = _mlp(2 * cell_width + 71, width, depth, 1)
        self.keep_network = _mlp(cell_width + 71, keep_width, keep_depth, 1)

    def train(self, mode=True):
        super().train(mode)
        self.position.eval()
        return self

    def encode(self, source):
        return self.position.upper_encode(source)

    def forward_encoded(self, cells, numbers, occupied):
        features = self.cell_network(cells)
        pooled = (features * occupied[..., None]).amax(dim=1)
        context = torch.cat([pooled, numbers], dim=1)
        local = torch.cat([features, context[:, None].expand(-1, 108, -1)], dim=2)
        removal = self.removal_network(local).squeeze(-1)
        removal = removal.masked_fill(~occupied, -torch.inf)
        keep = self.keep_network(context)
        return torch.cat([keep, removal], dim=1)

    def forward(self, source):
        """Return 109 logits: no change, then removal of each of the 108 cells."""
        return self.forward_encoded(*self.encode(source))

    def predict_event(self, source):
        return self(source).argmax(-1)

    def predict(self, source):
        event = self.predict_event(source)
        layout = source[:, 10:].clone()
        changed = event > 0
        rows = torch.where(changed)[0]
        layout[rows, event[changed] - 1] = 0
        return layout
