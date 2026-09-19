"""Minimal stop inputs, categorical predictions, and portable checkpoints."""

import io

import pytest
import torch
from test_ball_position import inputs
from torch.nn import functional as F

from gymemu.models import build_model


def test_stop_uses_only_current_vertical_ball_state():
    model = build_model(dict(kind="life_termination"))
    source = inputs()
    changed = source.clone()
    changed[:, [0, 2, 4, 5, 6, 7, 9]] += 1
    changed[:, 10:] = 1 - changed[:, 10:]
    assert model.encode(source).shape == (3, 31)
    assert torch.equal(model(source), model(changed))
    for field in [1, 3, 8]:
        changed = source.clone()
        changed[:, field] += 1
        assert not torch.equal(model.encode(source), model.encode(changed))
    with pytest.raises(ValueError, match="118"):
        model(source[:, :117])


def test_stop_training_reload_and_network_controlled_decision():
    spec = dict(kind="life_termination", width=16, depth=1)
    model = build_model(spec)
    source = inputs()
    optimizer = torch.optim.AdamW(model.parameters())
    loss = F.cross_entropy(model(source), torch.tensor([0, 1, 0]))
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    optimizer.step()
    buffer = io.BytesIO()
    torch.save(dict(model_spec=spec, model=model.state_dict()), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    replica = build_model(checkpoint["model_spec"])
    replica.load_state_dict(checkpoint["model"])
    assert torch.equal(replica.predict(source), model.predict(source))
    assert replica.predict(source).dtype == torch.bool
    assert replica.predict(source[:0]).shape == (0,)
    for cls in [0, 1]:
        with torch.no_grad():
            replica.network[-1].weight.zero_()
            replica.network[-1].bias.zero_()
            replica.network[-1].bias[cls] = 10
        # Both outcomes must be controlled by the network, not a death rule.
        assert (replica.predict(source) == bool(cls)).all()
