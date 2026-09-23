"""Bounded count updates, independent hit detection, and frozen dependencies."""

import io

import torch
from test_ball_position import inputs, specification
from torch.nn import functional as F

from gymemu.models import build_model


def model_spec():
    return dict(kind="paddle_hit_count", vertical=specification("y")["vertical"], width=16, depth=1)


def test_hit_decoder_keeps_or_increments_count_and_preserves_saturated_hits():
    model = build_model(model_spec())
    x = inputs()
    x[:, 7] = torch.tensor([2, 11, 12])
    x[2, 1] = 170
    with torch.no_grad():
        model.network[-1].weight.zero_()
        model.network[-1].bias.copy_(torch.tensor([-10, 10]))
    assert model.predict_hit(x).tolist() == [0, 1, 1]
    assert model.predict(x).tolist() == [2, 12, 12]
    with torch.no_grad():
        model.network[-1].bias.copy_(torch.tensor([10, -10]))
    assert torch.equal(model.predict(x), x[:, 7].long())
    assert model.predict(x[:0]).shape == (0,)


def test_hit_features_exclude_count_contact_and_bricks_and_freeze_parent():
    spec = model_spec()
    model = build_model(spec)
    x = inputs()
    changed = x.clone()
    changed[:, 7] = 12 - x[:, 7]
    changed[:, 9:] = 1 - x[:, 9:]
    assert torch.equal(model.encode(x), model.encode(changed))
    assert torch.equal(model.predict_hit(x), model.predict_hit(changed))
    assert model.encode(x).shape == (3, 96)
    parent = {k: v.clone() for k, v in model.vertical.state_dict().items()}
    before = model.vertical.predict(x)
    model.train()
    assert not any(m.training for m in model.vertical.modules())
    optimizer = torch.optim.AdamW(model.network.parameters())
    loss = F.cross_entropy(model(x), torch.tensor([0, 1, 0]))
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert all(torch.equal(v, parent[k]) for k, v in model.vertical.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in model.vertical.parameters())
    buffer = io.BytesIO()
    torch.save(dict(model_spec=spec, model=model.state_dict()), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    replica = build_model(checkpoint["model_spec"])
    replica.load_state_dict(checkpoint["model"])
    assert torch.equal(replica.predict(x), model.predict(x))
    assert torch.equal(replica.vertical.predict(x), before)
