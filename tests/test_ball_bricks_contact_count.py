"""Count composition, frozen state predictor and joint feedback semantics."""

import io

import numpy as np
import pytest
import torch
from test_ball_bricks import ToyBallBricksContact
from test_ball_position import inputs, specification
from torch.nn import functional as F

from gymemu.ball_bricks_eval import evaluate_ball_bricks
from gymemu.models import build_model


def count_container_spec():
    xp, yp = specification("x"), specification("y")
    xp.pop("kind")
    yp.pop("kind")
    layout = dict(position=yp, width=16, keep_width=16)
    spec = dict(
        kind="ball_bricks_contact_count",
        state=dict(
            state=dict(
                motion=dict(
                    horizontal=dict(position=xp, velocity_values=[-1, 1], shared=False),
                    vertical=dict(
                        pair=dict(position=yp, coupling_width=16), bounce_values=[-2, -1]
                    ),
                ),
                bricks=layout,
            ),
            contact=dict(layout=layout, width=16, collision_geometry=True),
        ),
        count=dict(vertical=yp["vertical"], width=16, depth=1),
    )
    return spec


def test_count_container_preserves_parents_and_only_trains_count_head():
    spec = count_container_spec()
    model = build_model(spec)
    src = inputs()
    src[:, 7] = torch.tensor([0, 11, 12])
    original = src.clone()
    expected = torch.cat([model.state.predict(src), model.count.predict(src)[:, None]], 1)
    assert torch.equal(model.predict(src), expected)
    assert torch.equal(src, original)
    frozen = {
        n: v.clone() for n, v in model.state_dict().items() if not n.startswith("count.network.")
    }
    model.train()
    assert not any(m.training for m in model.state.modules())
    assert not any(m.training for m in model.count.vertical.modules())
    assert all(
        n.startswith("count.network.") for n, p in model.named_parameters() if p.requires_grad
    )
    loss = F.cross_entropy(model(src)["count"], torch.tensor([0, 1, 0]))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.count.network.parameters()
    )
    assert all(p.grad is None for n, p in model.named_parameters() if n in frozen)
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]).step()
    assert all(torch.equal(model.state_dict()[n], v) for n, v in frozen.items())
    assert torch.equal(model.predict(src)[:, :113], expected[:, :113])
    with torch.no_grad():
        model.count.network[-1].weight.zero_()
        model.count.network[-1].bias.copy_(torch.tensor([-10, 10]))
    assert model.predict(src)[:, 113].tolist() == [0, 12, 12]
    buf = io.BytesIO()
    torch.save(dict(model_spec=spec, model=model.state_dict()), buf)
    buf.seek(0)
    checkpoint = torch.load(buf, weights_only=True)
    restored = build_model(checkpoint["model_spec"])
    restored.load_state_dict(checkpoint["model"])
    assert torch.equal(restored.predict(src), model.predict(src))
    assert restored.predict(src[:0]).shape == (0, 114)


class ToyCount(ToyBallBricksContact):
    def predict(self, src):
        output = super().predict(src)
        output[:, 0] += src[:, 7] / 8
        count = (src[:, 7] + (src[:, 0] >= 2)).clamp(0, 12)
        return torch.cat([output, count[:, None]], 1)


def test_count_feedback_matches_bruteforce_across_lives_and_saturation():
    src = np.zeros((9, 118), np.float32)
    src[:, 0] = [0, 2, 4, 6, 0, 2, 4, 6, 0]
    src[:, 1] = src[:, 0]
    src[:, 2:4] = 1
    src[:, 7] = [10, 10, 11, 12, 0, 0, 1, 2, 0]
    src[:, 10:] = 1
    target = np.c_[src[:, 0] + 2, src[:, 1] + 2, src[:, 2:4], src[:, 10:], np.zeros(9), src[:, 7]]
    data = dict(
        x=src,
        target=target,
        episodes=np.zeros(9, int),
        steps=np.arange(9),
        starts=np.array([0] * 4 + [4] * 4 + [8]),
        events=np.zeros(9, int),
    )
    model = ToyCount()
    with pytest.raises(ValueError, match="requires the contact"):
        evaluate_ball_bricks(model, data, hit_count=True)
    result = evaluate_ball_bricks(
        model, data, horizons=(1, 2, 4), modes=("reference", "both"), contact=True, hit_count=True
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
