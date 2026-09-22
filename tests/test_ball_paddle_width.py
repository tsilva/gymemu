"""Frozen container composition and coupled paddle-width feedback."""

import io

import numpy as np
import pytest
import torch
from test_ball_bricks_contact_count import ToyCount, count_container_spec
from test_ball_position import inputs
from torch.nn import functional as F

from gymemu.ball_bricks_eval import evaluate_ball_bricks
from gymemu.models import build_model


def width_container_spec():
    state = count_container_spec()
    state.pop("kind")
    spec = dict(
        kind="ball_paddle_width", state=state, paddle_width=dict(width=16, depth=2, proposal=True)
    )
    return spec


def test_width_container_preserves_state_and_only_trains_width():
    spec = width_container_spec()
    model = build_model(spec)
    src = inputs()
    src[1, 5] = 12
    original = src.clone()
    before = model.state.predict(src)
    expected = torch.cat([before, model.paddle_width.predict(src)[:, None]], 1)
    assert torch.equal(model.predict(src), expected)
    assert torch.equal(src, original)
    frozen = {n: v.clone() for n, v in model.state.state_dict().items()}
    model.train()
    assert not any(m.training for m in model.state.modules())
    assert all(
        n.startswith("paddle_width.network.")
        for n, p in model.named_parameters()
        if p.requires_grad
    )
    loss = F.cross_entropy(model(src)["paddle_width"], torch.tensor([1, 0, 1]))
    loss.backward()
    assert all(p.grad is None for p in model.state.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.paddle_width.parameters()
    )
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]).step()
    assert all(torch.equal(v, frozen[n]) for n, v in model.state.state_dict().items())
    assert torch.equal(model.predict(src)[:, :114], before)
    assert torch.isin(model.predict(src)[:, 114], torch.tensor([12.0, 16.0])).all()
    buf = io.BytesIO()
    torch.save(dict(model_spec=spec, model=model.state_dict()), buf)
    buf.seek(0)
    cp = torch.load(buf, weights_only=True)
    restored = build_model(cp["model_spec"])
    restored.load_state_dict(cp["model"])
    assert torch.equal(restored.predict(src), model.predict(src))
    assert model.predict(src[:0]).shape == (0, 115)


class ToyWidth(ToyCount):
    def predict(self, src):
        output = super().predict(src)
        output[:, 0] += (src[:, 5] - 16) / 8
        width = torch.where(src[:, 1] >= 4, 12, src[:, 5])
        return torch.cat([output, width[:, None]], 1)


def test_width_feedback_matches_bruteforce_across_lives():
    src = np.zeros((9, 118), np.float32)
    src[:, 0] = [0, 2, 4, 6, 0, 2, 4, 6, 0]
    src[:, 1] = src[:, 0]
    src[:, 2:4] = 1
    src[:, 7] = [10, 10, 11, 12, 0, 0, 1, 2, 0]
    src[:, 5] = 16
    src[:, 10:] = 1
    target = np.c_[
        src[:, 0] + 2, src[:, 1] + 2, src[:, 2:4], src[:, 10:], np.zeros(9), src[:, 7], src[:, 5]
    ]
    data = dict(
        x=src,
        target=target,
        episodes=np.zeros(9, int),
        steps=np.arange(9),
        starts=np.array([0] * 4 + [4] * 4 + [8]),
        events=np.zeros(9, int),
    )
    model = ToyWidth()
    with pytest.raises(ValueError, match="requires the count"):
        evaluate_ball_bricks(model, data, contact=True, paddle_width=True)
    result = evaluate_ball_bricks(
        model,
        data,
        horizons=(1, 2, 4),
        modes=("reference", "both"),
        contact=True,
        hit_count=True,
        paddle_width=True,
    )
    for mode in ("reference", "both"):
        for horizon in (1, 2, 4):
            predictions, targets, exact = [], [], []
            for start in range(9):
                end = start + horizon
                if end > 9 or data["starts"][start] != data["starts"][end - 1]:
                    continue
                previous, correct = None, True
                for row in range(start, end):
                    inp = src[row : row + 1].copy()
                    if previous is not None and mode == "both":
                        inp[0, [0, 2, 3]] = previous[[0, 2, 3]]
                        inp[0, 1] = np.floor(previous[1])
                        inp[0, 8] = np.rint(previous[1] % 1 * 8)
                        inp[0, 10:] = previous[4:112]
                        inp[0, 9] = previous[112]
                        inp[0, 7] = previous[113]
                        inp[0, 5] = previous[114]
                    previous = model.predict(torch.from_numpy(inp)).numpy()[0]
                    correct &= np.array_equal(previous, target[row])
                predictions.append(previous)
                targets.append(target[end - 1])
                exact.append(correct)
            wrong = np.array(predictions) != np.array(targets)
            r = result["rollouts"][mode][str(horizon)]
            assert r["windows"] == len(predictions) and r["exact_entire_window"] == sum(exact)
            assert r["joint_errors"] == wrong.any(1).sum()
            assert r["layout_errors"] == wrong[:, 4:112].any(1).sum()
            assert r["contact_errors"] == wrong[:, 112].sum()
            assert r["count_errors"] == wrong[:, 113].sum()
            assert r["width_errors"] == wrong[:, 114].sum()
