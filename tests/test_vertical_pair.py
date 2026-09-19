"""Atomic vertical updates, freeze boundaries and reference-bounded feedback."""

import io

import numpy as np
import pytest
import torch
from test_ball_position import inputs, specification
from torch import nn
from torch.nn import functional as F

from gymemu.models import build_model
from gymemu.vertical_pair_eval import evaluate_vertical_pair


@pytest.mark.parametrize("coupling_width", [0, 16])
def test_pair_reload_atomic_prediction_and_frozen_dependencies(coupling_width):
    position = specification("y")
    position.pop("kind")
    spec = dict(kind="vertical_ball_pair", position=position, coupling_width=coupling_width)
    model = build_model(spec)
    source = inputs()
    original = source.clone()
    y_logits, vy_logits = model(source)
    expected = torch.stack(
        [
            model.position.predict(source),
            model.position.vertical.classes[vy_logits.argmax(-1)],
        ],
        dim=1,
    )
    assert torch.equal(model.predict(source), expected)
    assert torch.equal(source, original)
    frozen_model = model.position if coupling_width else model.position.vertical.horizontal
    before = {k: v.clone() for k, v in frozen_model.state_dict().items()}
    model.train()
    assert not any(m.training for m in frozen_model.modules())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    y_logits, vy_logits = model(source)
    loss = F.cross_entropy(vy_logits, torch.tensor([0, 1, 0]))
    if not coupling_width:
        loss = loss + F.cross_entropy(y_logits, torch.tensor([0, 1, 2]))
    loss.backward()
    assert all(p.grad is None for p in frozen_model.parameters())
    opt.step()
    assert all(torch.equal(v, before[k]) for k, v in frozen_model.state_dict().items())
    buffer = io.BytesIO()
    torch.save(dict(model_spec=spec, model=model.state_dict()), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    replica = build_model(checkpoint["model_spec"])
    replica.load_state_dict(checkpoint["model"])
    assert torch.equal(replica.predict(source), model.predict(source))
    assert replica.predict(source[:0]).shape == (0, 2)


class ToyVertical(nn.Module):
    def predict(self, source):
        y = source[:, 1] + source[:, 8] / 8
        vy = source[:, 3]
        return torch.stack([y + 2 * vy + (y == 2) / 8, vy + (y == 4) / 8], dim=1)


def test_rollout_optimization_matches_brute_force_and_respects_life_boundaries():
    source = np.zeros((9, 118), np.float32)
    source[:, 1] = [0, 2, 4, 6, 0, 2, 4, 6, 0]
    source[:, 3] = 1
    data = dict(
        x=source,
        target=np.c_[source[:, 1] + 2, source[:, 3]],
        episodes=np.zeros(9, int),
        steps=np.arange(9),
        starts=np.array([0] * 4 + [4] * 4 + [8]),
        events=np.zeros(9, int),
    )
    model = ToyVertical()
    result = evaluate_vertical_pair(model, data, horizons=(1, 2, 4))
    assert result["segments"] == 3
    for mode in ("reference", "y_only", "vy_only", "both"):
        for horizon in (1, 2, 4):
            predictions, targets, exact = [], [], []
            for start in range(9):
                end = start + horizon
                if end > 9 or data["starts"][start] != data["starts"][end - 1]:
                    continue
                previous = None
                correct = True
                for row in range(start, end):
                    inp = source[row : row + 1].copy()
                    if previous is not None:
                        if mode in ("y_only", "both"):
                            inp[0, 1] = np.floor(previous[0])
                            inp[0, 8] = np.rint(previous[0] % 1 * 8)
                        if mode in ("vy_only", "both"):
                            inp[0, 3] = previous[1]
                    previous = model.predict(torch.from_numpy(inp)).numpy()[0]
                    correct &= np.array_equal(previous, data["target"][row])
                predictions.append(previous)
                targets.append(data["target"][end - 1])
                exact.append(correct)
            wrong = np.array(predictions) != np.array(targets)
            observed = result["rollouts"][mode][str(horizon)]
            assert observed["windows"] == len(predictions)
            assert observed["joint_errors"] == wrong.any(1).sum()
            assert observed["y_errors"] == wrong[:, 0].sum()
            assert observed["vy_errors"] == wrong[:, 1].sum()
            assert observed["exact_entire_window"] == sum(exact)
