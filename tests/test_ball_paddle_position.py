"""Frozen predecessors, current-state composition and coupled paddle feedback."""

import io

import numpy as np
import pytest
import torch
from test_ball_paddle_charge import ToyCharge, charge_container_spec
from test_ball_position import inputs
from torch.nn import functional as F

from gymemu.ball_bricks_eval import evaluate_ball_bricks
from gymemu.models import build_model


def position_container_spec():
    state = charge_container_spec()
    state.pop("kind")
    return dict(
        kind="ball_paddle_position",
        state=state,
        position=dict(
            encoding="hybrid",
            objective="classification",
            controller_fields=["charge"],
            width=16,
            depth=1,
            delta_values=[-1, 0, 1],
        ),
    )


def test_position_container_preserves_state_and_only_trains_position():
    spec = position_container_spec()
    model = build_model(spec)
    source = torch.cat([inputs(), torch.arange(3)[:, None]], 1)
    original = source.clone()
    adapted = model.position_inputs(source)
    assert torch.equal(adapted[:, 0], source[:, 4])
    assert torch.equal(adapted[:, 1], source[:, 6])
    assert torch.equal(adapted[:, 5], torch.arange(3))
    assert not adapted[:, 2:5].any()
    assert torch.equal(model.position.encode(adapted)[:, 2:5], torch.eye(3))
    old = model.state.predict(source)
    expected = torch.cat(
        [old, (source[:, 4] + model.position.delta(model.position(adapted)))[:, None]], 1
    )
    assert torch.equal(model.predict(source), expected)
    assert torch.equal(source, original)
    frozen = {n: v.clone() for n, v in model.state.state_dict().items()}
    model.train()
    assert not any(m.training for m in model.state.modules())
    assert all(
        n.startswith("position.network.") for n, p in model.named_parameters() if p.requires_grad
    )
    F.cross_entropy(model(source)["paddle_x"], torch.tensor([0, 1, 2])).backward()
    assert all(p.grad is None for p in model.state.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.position.parameters())
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]).step()
    assert all(torch.equal(v, frozen[n]) for n, v in model.state.state_dict().items())
    assert torch.equal(model.predict(source)[:, :116], old)
    buf = io.BytesIO()
    torch.save(dict(model_spec=spec, model=model.state_dict()), buf)
    buf.seek(0)
    cp = torch.load(buf, weights_only=True)
    restored = build_model(cp["model_spec"])
    restored.load_state_dict(cp["model"])
    assert torch.equal(restored.predict(source), model.predict(source))
    assert restored.predict(source[:0]).shape == (0, 117)


def test_position_rejects_missing_or_invalid_actions_and_hidden_fields():
    spec = position_container_spec()
    model = build_model(spec)
    with pytest.raises(ValueError, match="provider action"):
        model.predict(inputs())
    for action in (-1, 3, 0.5, float("nan"), float("inf")):
        source = torch.cat([inputs(), torch.full((3, 1), action)], 1)
        with pytest.raises(ValueError, match="integer provider actions"):
            model.predict(source)
    spec["position"]["controller_fields"] = ["charge", "repeat"]
    with pytest.raises(ValueError, match="only the charge"):
        build_model(spec)


class ToyPosition(ToyCharge):
    def predict(self, src):
        output = super().predict(src)
        output[:, 115] += src[:, 4] / 8
        position = src[:, 4] + src[:, 6] / 8 + src[:, 118]
        return torch.cat([output, position[:, None]], 1)


def test_position_and_charge_feedback_matches_brute_force_with_life_boundaries():
    source = np.zeros((9, 119), np.float32)
    source[:, 0] = [0, 2, 4, 6, 0, 2, 4, 6, 0]
    source[:, 1] = source[:, 0]
    source[:, 2:4] = 1
    source[:, 4] = 16
    source[:, 5] = 16
    source[:, 6] = 10
    source[:, 10:118] = 1
    source[:, 118] = [0, 1, 2, 1, 2, 1, 0, 2, 1]
    target = np.c_[
        source[:, 0] + 2,
        source[:, 1] + 2,
        source[:, 2:4],
        source[:, 10:118],
        np.zeros((9, 2)),
        source[:, 5],
        source[:, 6],
        source[:, 4],
    ]
    data = dict(
        x=source,
        target=target,
        episodes=np.zeros(9, int),
        steps=np.arange(9),
        starts=np.array([0] * 4 + [4] * 4 + [8]),
        events=np.zeros(9, int),
    )
    model = ToyPosition()
    with pytest.raises(ValueError, match="requires the charge"):
        evaluate_ball_bricks(
            model, data, contact=True, hit_count=True, paddle_width=True, paddle_position=True
        )
    result = evaluate_ball_bricks(
        model,
        data,
        horizons=(1, 2, 4),
        modes=("reference", "both"),
        contact=True,
        hit_count=True,
        paddle_width=True,
        paddle_charge=True,
        paddle_position=True,
    )
    for mode in ("reference", "both"):
        for h in (1, 2, 4):
            predictions, targets, exact = [], [], []
            for start in range(9):
                end = start + h
                if end > 9 or data["starts"][start] != data["starts"][end - 1]:
                    continue
                previous, correct = None, True
                for row in range(start, end):
                    inp = source[row : row + 1].copy()
                    if previous is not None and mode == "both":
                        inp[0, [0, 2, 3]] = previous[[0, 2, 3]]
                        inp[0, 1] = np.floor(previous[1])
                        inp[0, 8] = np.rint(previous[1] % 1 * 8)
                        inp[0, 10:118] = previous[4:112]
                        inp[0, 9] = previous[112]
                        inp[0, 7] = previous[113]
                        inp[0, 5] = previous[114]
                        inp[0, 6] = previous[115]
                        inp[0, 4] = previous[116]
                    previous = model.predict(torch.from_numpy(inp)).numpy()[0]
                    correct &= np.array_equal(previous, target[row])
                predictions.append(previous)
                targets.append(target[end - 1])
                exact.append(correct)
            pred = np.array(predictions)
            truth = np.array(targets)
            wrong = pred != truth
            r = result["rollouts"][mode][str(h)]
            assert r["windows"] == len(predictions) and r["exact_entire_window"] == sum(exact)
            assert r["joint_errors"] == wrong.any(1).sum()
            assert r["charge_errors"] == wrong[:, 115].sum()
            assert r["charge_mae"] == np.abs(pred[:, 115] - truth[:, 115]).mean()
            assert r["invalid_charge_predictions"] == 0
            assert r["paddle_x_errors"] == wrong[:, 116].sum()
            assert r["paddle_x_mae"] == np.abs(pred[:, 116] - truth[:, 116]).mean()
            assert r["paddle_x_max_error"] == np.abs(pred[:, 116] - truth[:, 116]).max()
    assert np.array_equal(source[:, 118], [0, 1, 2, 1, 2, 1, 0, 2, 1])
