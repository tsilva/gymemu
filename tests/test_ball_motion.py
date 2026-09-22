"""Atomic four-field merge, training boundaries and recursive evaluation."""

import io

import numpy as np
import torch
from test_ball_position import inputs, specification
from torch import nn
from torch.nn import functional as F

from gymemu.ball_motion_eval import evaluate_ball_motion
from gymemu.models import build_model


def test_merged_predictions_gradients_frozen_weights_and_portable_reload():
    xspec = specification("x")
    xspec.pop("kind")
    yspec = specification("y")
    yspec.pop("kind")
    spec = dict(
        kind="ball_motion",
        horizontal=dict(position=xspec, velocity_values=[-1, 1], shared=False),
        vertical=dict(
            pair=dict(position=yspec, coupling_width=16), bounce_values=[-2, -1], edge_features=True
        ),
    )
    model = build_model(spec)
    source = inputs()
    original = source.clone()
    horizontal = model.horizontal.predict(source)
    vertical = model.vertical.predict(source)
    expected = torch.stack([horizontal[:, 0], vertical[:, 0], horizontal[:, 1], vertical[:, 1]], 1)
    assert torch.equal(model.predict(source), expected)
    assert torch.equal(source, original)
    frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    model.train()
    for dep in (
        model.horizontal.position.vertical,
        model.vertical.pair.position.vertical.horizontal,
        model.vertical.pair.coupling,
        model.vertical.pair.position.flight_network,
    ):
        assert not any(m.training for m in dep.modules())
    output = model(source)
    logits = list(output["horizontal"]) + list(output["upper"]) + list(output["paddle"])
    loss = sum(F.cross_entropy(z, torch.zeros(len(z), dtype=torch.long)) for z in logits)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    loss.backward()
    for branch in (model.horizontal, model.vertical.trunk, *model.vertical_upper_modules()):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters())
    assert all(p.grad is None for n, p in model.named_parameters() if n in frozen)
    opt.step()
    assert all(torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen)
    model.eval()
    buf = io.BytesIO()
    torch.save(dict(model_spec=spec, model=model.state_dict()), buf)
    buf.seek(0)
    cp = torch.load(buf, weights_only=True)
    restored = build_model(cp["model_spec"])
    restored.load_state_dict(cp["model"])
    assert torch.equal(model.predict(source), restored.predict(source))
    assert model.predict(source[:0]).shape == (0, 4)


class ToyBall(nn.Module):
    def predict(self, source):
        x = source[:, 0]
        y = source[:, 1] + source[:, 8] / 8
        vx = source[:, 2]
        vy = source[:, 3]
        # Errors depend on both coordinates, exposing cross-axis feedback mistakes.
        return torch.stack(
            [
                x + 2 * vx + (y == 4) / 8,
                y + 2 * vy + (x == 2) / 8,
                vx + (y == 6) / 8,
                vy + (x == 4) / 8,
            ],
            1,
        )


def test_four_field_rollouts_match_bruteforce_and_respect_boundaries():
    src = np.zeros((9, 118), np.float32)
    src[:, 0] = [0, 2, 4, 6, 0, 2, 4, 6, 0]
    src[:, 1] = src[:, 0]
    src[:, 2:4] = 1
    d = dict(
        x=src,
        target=np.c_[src[:, 0] + 2, src[:, 1] + 2, src[:, 2:4]],
        episodes=np.zeros(9, int),
        steps=np.arange(9),
        starts=np.array([0] * 4 + [4] * 4 + [8]),
        events=np.zeros(9, int),
    )
    m = ToyBall()
    modes = ("reference", "horizontal", "vertical", "both")
    result = evaluate_ball_motion(m, d, horizons=(1, 2, 4), modes=modes)
    assert result["segments"] == 3
    for mode in modes:
        for h in (1, 2, 4):
            predictions = []
            targets = []
            exact = []
            for start in range(9):
                end = start + h
                if end > 9 or d["starts"][start] != d["starts"][end - 1]:
                    continue
                previous = None
                correct = True
                for row in range(start, end):
                    inp = src[row : row + 1].copy()
                    if previous is not None:
                        if mode in ("horizontal", "both"):
                            inp[0, [0, 2]] = previous[[0, 2]]
                        if mode in ("vertical", "both"):
                            inp[0, 1] = np.floor(previous[1])
                            inp[0, 8] = np.rint(previous[1] % 1 * 8)
                            inp[0, 3] = previous[3]
                    previous = m.predict(torch.from_numpy(inp)).numpy()[0]
                    correct &= np.array_equal(previous, d["target"][row])
                predictions.append(previous)
                targets.append(d["target"][end - 1])
                exact.append(correct)
            predictions = np.array(predictions)
            targets = np.array(targets)
            wrong = predictions != targets
            observed = result["rollouts"][mode][str(h)]
            assert observed["windows"] == len(predictions)
            assert observed["joint_errors"] == wrong.any(1).sum()
            assert observed["exact_entire_window"] == sum(exact)
            for i, field in enumerate(("x", "y", "vx", "vy")):
                assert observed[field + "_errors"] == wrong[:, i].sum()
            for i, field in enumerate(("x", "y")):
                assert np.isclose(
                    observed[field + "_mae"], np.abs(predictions[:, i] - targets[:, i]).mean()
                )
