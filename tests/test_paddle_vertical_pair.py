"""Learned paddle timing and speed must decode to one consistent transition."""

import io

import pytest
import torch
from test_ball_position import inputs, specification
from torch.nn import functional as F

from gymemu.models import build_model


@pytest.mark.parametrize("edge_features", [False, True])
def test_paddle_factorization_decoding_freeze_training_and_reload(edge_features):
    position = specification("y")
    position.pop("kind")
    spec = dict(
        kind="paddle_vertical_pair",
        pair=dict(position=position, coupling_width=16),
        bounce_values=[-2, -1],
        edge_features=edge_features,
    )
    model = build_model(spec)
    source = inputs()[1:2].repeat(3, 1)
    source[:, 1] = 177
    source[:, 8] = 4
    timing = torch.eye(3) * 10
    speed = torch.tensor([[0.0, 10.0]] * 3)
    expected = torch.tensor([[181.5, 2], [175.5, -1], [178.5, -1]])
    assert torch.equal(model.decode(source, timing, speed), expected)
    changed_speed = speed.flip(1)
    assert torch.equal(model.decode(source, timing, changed_speed)[0], expected[0])
    original = {k: v.clone() for k, v in model.pair.state_dict().items()}
    model.train()
    assert all(not m.training for m in model.pair.modules())
    features = model.encode(source)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    tl, sl = model.forward_encoded(features)
    loss = F.cross_entropy(tl, torch.arange(3)) + F.cross_entropy(sl[1:], torch.ones(2).long())
    optimizer.zero_grad()
    loss.backward()
    assert all(p.grad is None for p in model.pair.parameters())
    assert model.timing.weight.grad.abs().sum() > 0
    assert model.speed.weight.grad.abs().sum() > 0
    optimizer.step()
    assert all(torch.equal(v, original[k]) for k, v in model.pair.state_dict().items())
    full = inputs()
    predicted = model.predict(full)
    assert torch.equal(predicted[[0, 2]], model.pair.predict(full[[0, 2]]))
    assert model.predict(full[:0]).shape == (0, 2)
    buffer = io.BytesIO()
    torch.save(dict(model_spec=spec, model=model.state_dict()), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    restored = build_model(checkpoint["model_spec"])
    restored.load_state_dict(checkpoint["model"])
    assert torch.equal(restored.predict(full), predicted)


def test_edge_features_preserve_control_initialization_and_use_source_geometry(monkeypatch):
    position = specification("y")
    position.pop("kind")
    spec = dict(kind="paddle_vertical_pair", pair=dict(position=position), bounce_values=[-2, -1])
    torch.manual_seed(91)
    control = build_model(spec)
    control_rng = torch.get_rng_state()
    torch.manual_seed(91)
    candidate = build_model(dict(spec, edge_features=True))
    assert torch.equal(torch.get_rng_state(), control_rng)
    source = inputs()[1:2]
    source[:, :9] = torch.tensor([135, 177, -2, 3.375, 121, 12, 851, 3, 7])
    original = source.clone()
    # A fixed learned-paddle output isolates the geometry arithmetic.
    direction = candidate.pair.position.vertical.horizontal.base.paddle_model.direction
    monkeypatch.setattr(direction, "intermediate_paddle", lambda x: x[:, 4] + 2)
    assert torch.equal(
        candidate.edge_encode(source), torch.tensor([[15, -3, 0, 6, 11, 1, 4, 2]]) / 16
    )
    assert torch.equal(source, original)
    encoded = control.encode(source)
    expanded = torch.cat([encoded, candidate.edge_encode(source)], dim=1)
    for before, after in zip(
        control.forward_encoded(encoded), candidate.forward_encoded(expanded), strict=True
    ):
        # Wider matrix multiplication can change float32 summation order.
        torch.testing.assert_close(before, after, rtol=1e-6, atol=1e-7)
    assert (candidate.trunk[0].weight[:, -8:] == 0).all()
