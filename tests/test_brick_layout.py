"""Coherent layout decoding and preservation of the frozen ball models."""

import io

import pytest
import torch
from test_ball_position import inputs
from test_ball_position import specification as position_specification
from torch.nn import functional as F

from gymemu.models import build_model


def specification():
    position = position_specification("y")
    position.pop("kind")
    return dict(
        kind="brick_layout", position=position, width=16, depth=1, keep_width=16, keep_depth=1
    )


def test_brick_decoder_removes_one_present_cell_or_keeps_empty_layout():
    model = build_model(specification())
    x = inputs()
    x[:, 10:] = 0
    x[0, 27] = 1
    x[2, [13, 17]] = 1
    with torch.no_grad():
        model.removal_network[-1].weight.zero_()
        model.removal_network[-1].bias.fill_(10)
        model.keep_network[-1].weight.zero_()
        model.keep_network[-1].bias.fill_(-10)
    assert model.predict_event(x).tolist() == [18, 0, 4]
    predicted = model.predict(x)
    assert ((x[:, 10:] - predicted).sum(1)).tolist() == [1, 0, 1]
    assert predicted[2, 7] == 1
    assert (predicted <= x[:, 10:]).all()
    assert not model(x).isnan().any()
    with torch.no_grad():
        model.keep_network[-1].bias.fill_(20)
    assert torch.equal(model.predict(x), x[:, 10:])
    assert model.predict(x[:0]).shape == (0, 108)
    with pytest.raises(ValueError, match="118"):
        model(x[:, :117])


def test_brick_training_and_checkpoint_keep_parent_predictions_unchanged():
    spec = specification()
    model = build_model(spec)
    x = inputs()
    x[1, 10:] = 0
    changed_paddle = x.clone()
    changed_paddle[:, 4:8] += 3
    assert torch.equal(model(x), model(changed_paddle))
    parent = {k: v.clone() for k, v in model.position.state_dict().items()}
    y = model.position.predict(x)
    velocity = model.position.predict_velocities(x)
    model.train()
    assert not any(m.training for m in model.position.modules())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    loss = F.cross_entropy(model(x), torch.tensor([11, 0, 0]))
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert all(torch.equal(v, parent[k]) for k, v in model.position.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in model.position.parameters())
    assert torch.equal(model.position.predict(x), y)
    assert torch.equal(model.position.predict_velocities(x), velocity)
    checkpoint = io.BytesIO()
    torch.save(dict(model_spec=spec, model=model.state_dict()), checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, weights_only=True)
    replica = build_model(restored["model_spec"])
    replica.load_state_dict(restored["model"])
    assert torch.equal(replica.predict(x), model.predict(x))
    assert torch.equal(replica.position.predict(x), y)
    assert torch.equal(replica.position.predict_velocities(x), velocity)
