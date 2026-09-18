"""Small learned paddle transition from a known sufficient current state."""

import torch
from torch import nn
from torch.nn import functional as F


class PaddleTransitionMLP(nn.Module):
    """Generic scalar/binary encodings; no native controller update rules."""

    def __init__(
        self,
        encoding="scalar",
        objective="regression",
        width=128,
        depth=2,
        delta_min=-11,
        delta_max=11,
        controller_fields=("charge", "measure", "repeat", "held"),
        delta_values=None,
    ):
        super().__init__()
        if encoding not in ("scalar", "hybrid") or objective not in (
            "regression",
            "classification",
        ):
            raise ValueError("Unknown paddle transition encoding/objective")
        if min(width, depth) < 1 or delta_max < delta_min:
            raise ValueError("Invalid paddle transition dimensions")
        names = ("charge", "measure", "repeat", "held")
        if len(set(controller_fields)) != len(controller_fields) or any(
            field not in names for field in controller_fields
        ):
            raise ValueError("Invalid controller fields")
        self.numeric_columns = (0,) + tuple(names.index(field) + 1 for field in controller_fields)
        self.encoding, self.objective = encoding, objective
        self.delta_min, self.delta_max = delta_min, delta_max
        if delta_values is not None and (
            objective != "classification"
            or not delta_values
            or delta_values != sorted(set(delta_values))
            or any(type(value) is not int for value in delta_values)
        ):
            raise ValueError("Expected sorted unique integer displacement classes")
        self.register_buffer(
            "delta_values", None if delta_values is None else torch.tensor(delta_values).float()
        )
        self.register_buffer("scales", torch.tensor([160, 3856, 235, 60, 1], dtype=torch.float32))
        inputs = len(self.numeric_columns) + 3
        if encoding == "hybrid":
            inputs += sum((8, 12, 8, 6, 1)[k] for k in self.numeric_columns)
        layers = []
        for _ in range(depth):
            layers.extend([nn.Linear(inputs, width), nn.SiLU()])
            inputs = width
        outputs = len(delta_values) if delta_values is not None else delta_max - delta_min + 1
        layers.append(nn.Linear(inputs, 1 if objective == "regression" else outputs))
        self.network = nn.Sequential(*layers)

    def encode(self, source):
        # x, charge, measurement, repeat, held, requested action. All are current.
        values = source[:, self.numeric_columns]
        action = F.one_hot(source[:, 5].long(), 3).float()
        features = [values / self.scales[list(self.numeric_columns)] * 2 - 1, action]
        if self.encoding == "hybrid":
            integers = values.round().long()
            for j, k in enumerate(self.numeric_columns):
                width = (8, 12, 8, 6, 1)[k]
                bits = (integers[:, j, None] >> torch.arange(width, device=source.device)) & 1
                features.append(bits.float() * 2 - 1)
        return torch.cat(features, dim=1)

    def forward(self, source):
        return self.network(self.encode(source))

    def delta(self, output):
        if self.objective == "classification":
            if self.delta_values is not None:
                return self.delta_values[output.argmax(-1)]
            return output.argmax(-1).float() + self.delta_min
        return output[:, 0]
