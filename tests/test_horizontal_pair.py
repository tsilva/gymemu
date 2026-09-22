"""Shared gradients, atomic updates, portability and recursive boundaries."""

import io

import numpy as np
import pytest
import torch
from test_ball_position import inputs, specification
from torch import nn
from torch.nn import functional as F

from gymemu.horizontal_pair_eval import evaluate_horizontal_pair
from gymemu.models import build_model


@pytest.mark.parametrize("shared", [False, True])
def test_pair_freeze_reload_and_gradient_isolation(shared):
    position = specification("x")
    position.pop("kind")
    spec = dict(
        kind="horizontal_ball_pair", position=position, velocity_values=[-1.0, 1.0], shared=shared
    )
    torch.manual_seed(5)
    model = build_model(spec)
    model.train()
    assert not any(m.training for m in model.position.vertical.modules())
    source = inputs()
    before = source.clone()
    logits, vlogits = model(source)
    assert torch.equal(logits, model.position(source))
    assert torch.equal(source, before)
    F.cross_entropy(vlogits, torch.tensor([0, 1, 0])).backward()
    assert all(p.grad is None for p in model.position.vertical.parameters())
    trunk = [p for n, p in model.position.named_parameters() if not n.startswith("vertical.")]
    if shared:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trunk)
    else:
        assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in trunk)
    b = io.BytesIO()
    torch.save(dict(model_spec=spec, model=model.state_dict()), b)
    b.seek(0)
    c = torch.load(b, weights_only=True)
    replica = build_model(c["model_spec"])
    replica.load_state_dict(c["model"])
    assert torch.equal(model.predict(source), replica.predict(source))
    assert model.predict(source[:0]).shape == (0, 2)


def test_matching_initial_predictions_and_parameter_saving():
    pos = specification("x")
    pos.pop("kind")
    models = []
    for shared in (False, True):
        torch.manual_seed(12)
        models.append(
            build_model(
                dict(
                    kind="horizontal_ball_pair",
                    position=pos,
                    velocity_values=[-1.0, 1.0],
                    shared=shared,
                )
            )
        )
    a, b = models
    for x, y in zip(a(inputs()), b(inputs()), strict=True):
        assert torch.equal(x, y)
    assert sum(p.numel() for p in b.parameters()) < sum(p.numel() for p in a.parameters())


class ToyHorizontal(nn.Module):
    def predict(self, source):
        y = source[:, 0]
        vy = source[:, 2]
        return torch.stack([y + 2 * vy + (y == 2) / 8, vy + (y == 4) / 8], dim=1)


def test_rollout_optimization_matches_brute_force_and_respects_life_boundaries():
    source = np.zeros((9, 118), np.float32)
    source[:, 0] = [0, 2, 4, 6, 0, 2, 4, 6, 0]
    source[:, 2] = 1
    data = dict(
        x=source,
        target=np.c_[source[:, 0] + 2, source[:, 2]],
        episodes=np.zeros(9, int),
        steps=np.arange(9),
        starts=np.array([0] * 4 + [4] * 4 + [8]),
        events=np.zeros(9, int),
    )
    model = ToyHorizontal()
    result = evaluate_horizontal_pair(model, data, horizons=(1, 2, 4))
    assert result["segments"] == 3
    for mode in ("reference", "x_only", "vx_only", "both"):
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
                        if mode in ("x_only", "both"):
                            inp[0, 0] = previous[0]
                        if mode in ("vx_only", "both"):
                            inp[0, 2] = previous[1]
                    previous = model.predict(torch.from_numpy(inp)).numpy()[0]
                    correct &= np.array_equal(previous, data["target"][row])
                predictions.append(previous)
                targets.append(data["target"][end - 1])
                exact.append(correct)
            wrong = np.array(predictions) != np.array(targets)
            observed = result["rollouts"][mode][str(horizon)]
            assert observed["windows"] == len(predictions)
            assert observed["joint_errors"] == wrong.any(1).sum()
            assert observed["x_errors"] == wrong[:, 0].sum()
            assert observed["vx_errors"] == wrong[:, 1].sum()
            assert observed["exact_entire_window"] == sum(exact)
