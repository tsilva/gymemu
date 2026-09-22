"""Atomic layout merge, frozen dependencies and coupled feedback boundaries."""

import io

import numpy as np
import torch
from test_ball_position import inputs, specification
from torch import nn
from torch.nn import functional as F

from gymemu.ball_bricks_eval import evaluate_ball_bricks
from gymemu.models import build_model


def test_atomic_merge_layout_constraints_gradients_and_reload():
    xp = specification("x")
    xp.pop("kind")
    yp = specification("y")
    yp.pop("kind")
    motion = dict(
        horizontal=dict(position=xp, velocity_values=[-1, 1], shared=False),
        vertical=dict(
            pair=dict(position=yp, coupling_width=16), bounce_values=[-2, -1], edge_features=True
        ),
    )
    spec = dict(
        kind="ball_bricks", motion=motion, bricks=dict(position=yp, width=16, keep_width=16)
    )
    m = build_model(spec)
    src = inputs()
    original = src.clone()
    assert torch.equal(m.predict(src), torch.cat([m.motion.predict(src), m.bricks.predict(src)], 1))
    assert torch.equal(src, original)
    predicted = m.predict(src)[:, 4:]
    assert (predicted <= src[:, 10:]).all() and ((predicted != src[:, 10:]).sum(1) <= 1).all()
    empty = src.clone()
    empty[:, 10:] = 0
    assert not m.predict(empty)[:, 4:].any()
    frozen = {n: p.clone() for n, p in m.named_parameters() if not p.requires_grad}
    m.train()
    assert not any(a.training for a in m.bricks.position.modules())
    out = m(src)
    loss = F.cross_entropy(out["bricks"], torch.tensor([1, 0, 0]))
    for z in list(out["horizontal"]) + list(out["upper"]) + list(out["paddle"]):
        loss = loss + F.cross_entropy(z, torch.zeros(len(z), dtype=torch.long))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad])
    loss.backward()
    for branch in (
        m.motion,
        m.bricks.cell_network,
        m.bricks.removal_network,
        m.bricks.keep_network,
    ):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters())
    assert all(p.grad is None for n, p in m.named_parameters() if n in frozen)
    opt.step()
    assert all(torch.equal(p, frozen[n]) for n, p in m.named_parameters() if n in frozen)
    buf = io.BytesIO()
    torch.save(dict(model_spec=spec, model=m.state_dict()), buf)
    buf.seek(0)
    cp = torch.load(buf, weights_only=True)
    restored = build_model(cp["model_spec"])
    restored.load_state_dict(cp["model"])
    assert torch.equal(restored.predict(src), m.predict(src))
    assert m.predict(src[:0]).shape == (0, 112)


class ToyBallBricks(nn.Module):
    def predict(self, src):
        x = src[:, 0]
        y = src[:, 1] + src[:, 8] / 8
        ball = torch.stack(
            [x + 2 + (src[:, 10] == 0) / 8, y + 2 + (x == 2) / 8, src[:, 2], src[:, 3]], 1
        )
        bricks = src[:, 10:].clone()
        bricks[x >= 2, 0] = 0
        bricks[y >= 6, 1] = 0
        return torch.cat([ball, bricks], 1)


def test_layout_feedback_matches_bruteforce_and_enforces_life_boundaries():
    src = np.zeros((9, 118), np.float32)
    src[:, 0] = [0, 2, 4, 6, 0, 2, 4, 6, 0]
    src[:, 1] = src[:, 0]
    src[:, 2:4] = 1
    src[:, 10:] = 1
    target = np.c_[src[:, 0] + 2, src[:, 1] + 2, src[:, 2:4], src[:, 10:]]
    d = dict(
        x=src,
        target=target,
        episodes=np.zeros(9, int),
        steps=np.arange(9),
        starts=np.array([0] * 4 + [4] * 4 + [8]),
        events=np.zeros(9, int),
    )
    m = ToyBallBricks()
    modes = ("reference", "ball", "bricks", "both")
    result = evaluate_ball_bricks(m, d, horizons=(1, 2, 4), modes=modes)
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
                        if mode in ("ball", "both"):
                            inp[0, [0, 2, 3]] = previous[[0, 2, 3]]
                            inp[0, 1] = np.floor(previous[1])
                            inp[0, 8] = np.rint(previous[1] % 1 * 8)
                        if mode in ("bricks", "both"):
                            inp[0, 10:] = previous[4:]
                    previous = m.predict(torch.from_numpy(inp)).numpy()[0]
                    correct &= np.array_equal(previous, target[row])
                predictions.append(previous)
                targets.append(target[end - 1])
                exact.append(correct)
            wrong = np.array(predictions) != np.array(targets)
            r = result["rollouts"][mode][str(h)]
            assert r["windows"] == len(predictions) and r["exact_entire_window"] == sum(exact)
            assert r["joint_errors"] == wrong.any(1).sum()
            assert r["ball_errors"] == wrong[:, :4].any(1).sum()
            assert r["layout_errors"] == wrong[:, 4:].any(1).sum()
            assert r["brick_cell_errors"] == wrong[:, 4:].sum()


def test_contact_merge_preserves_parents_and_frozen_layout():
    xp = specification("x")
    xp.pop("kind")
    yp = specification("y")
    yp.pop("kind")
    layout = dict(position=yp, width=16, keep_width=16)
    spec = dict(
        kind="ball_bricks_contact",
        state=dict(
            motion=dict(
                horizontal=dict(position=xp, velocity_values=[-1, 1], shared=False),
                vertical=dict(pair=dict(position=yp, coupling_width=16), bounce_values=[-2, -1]),
            ),
            bricks=layout,
        ),
        contact=dict(layout=layout, width=16, collision_geometry=True),
    )
    m = build_model(spec)
    src = inputs()
    original = src.clone()
    expected = torch.cat([m.state.predict(src), m.contact.predict(src)[:, None]], 1)
    assert torch.equal(m.predict(src), expected)
    assert torch.equal(src, original)
    frozen = {n: v.clone() for n, v in m.contact.layout.state_dict().items()}
    m.train()
    assert not any(a.training for a in m.contact.layout.modules())
    loss = F.cross_entropy(m(src)["contact"], torch.tensor([1, 0, 0]))
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in m.contact.network.parameters()
    )
    assert all(p.grad is None for p in m.contact.layout.parameters())
    torch.optim.AdamW([p for p in m.parameters() if p.requires_grad]).step()
    assert all(torch.equal(v, frozen[n]) for n, v in m.contact.layout.state_dict().items())
    buf = io.BytesIO()
    torch.save(dict(model_spec=spec, model=m.state_dict()), buf)
    buf.seek(0)
    cp = torch.load(buf, weights_only=True)
    restored = build_model(cp["model_spec"])
    restored.load_state_dict(cp["model"])
    assert torch.equal(restored.predict(src), m.predict(src))
    assert m.predict(src[:0]).shape == (0, 113)


class ToyBallBricksContact(ToyBallBricks):
    def predict(self, src):
        out = super().predict(src)
        out[:, 0] += src[:, 9]
        out[:, 4] *= 1 - src[:, 9]
        contact = (src[:, 0] >= 2).float()
        return torch.cat([out, contact[:, None]], 1)


def test_contact_feedback_couples_to_motion_and_bricks_without_crossing_lives():
    src = np.zeros((9, 118), np.float32)
    src[:, 0] = [0, 2, 4, 6, 0, 2, 4, 6, 0]
    src[:, 1] = src[:, 0]
    src[:, 2:4] = 1
    src[:, 10:] = 1
    target = np.c_[src[:, 0] + 2, src[:, 1] + 2, src[:, 2:4], src[:, 10:], np.zeros(9)]
    d = dict(
        x=src,
        target=target,
        episodes=np.zeros(9, int),
        steps=np.arange(9),
        starts=np.array([0] * 4 + [4] * 4 + [8]),
        events=np.zeros(9, int),
    )
    model = ToyBallBricksContact()
    result = evaluate_ball_bricks(model, d, horizons=(1, 2, 4), contact=True)
    for h in (1, 2, 4):
        predictions, targets, exact = [], [], []
        for start in range(9):
            end = start + h
            if end > 9 or d["starts"][start] != d["starts"][end - 1]:
                continue
            previous, correct = None, True
            for row in range(start, end):
                inp = src[row : row + 1].copy()
                if previous is not None:
                    inp[0, [0, 2, 3]] = previous[[0, 2, 3]]
                    inp[0, 1] = np.floor(previous[1])
                    inp[0, 8] = np.rint(previous[1] % 1 * 8)
                    inp[0, 10:] = previous[4:112]
                    inp[0, 9] = previous[112]
                previous = model.predict(torch.from_numpy(inp)).numpy()[0]
                correct &= np.array_equal(previous, target[row])
            predictions.append(previous)
            targets.append(target[end - 1])
            exact.append(correct)
        wrong = np.array(predictions) != np.array(targets)
        r = result["rollouts"]["both"][str(h)]
        assert r["windows"] == len(predictions) and r["exact_entire_window"] == sum(exact)
        assert r["joint_errors"] == wrong.any(1).sum()
        assert r["layout_errors"] == wrong[:, 4:112].any(1).sum()
        assert r["brick_cell_errors"] == wrong[:, 4:112].sum()
        assert r["contact_errors"] == wrong[:, 112].sum()
