"""Frozen-pair refinement, zero initialization and portable inference."""

import io

import pytest
import torch
from test_ball_position import inputs, specification
from torch.nn import functional as F

from gymemu.models import build_model


def test_refinement_starts_identical_and_trains_only_residual_heads():
    position = specification("y")
    position.pop("kind")
    spec = dict(
        kind="vertical_displacement_refinement",
        pair=dict(position=position, coupling_width=16),
        width=16,
    )
    model = build_model(spec)
    source = inputs()
    before_source = source.clone()
    assert torch.equal(model.predict(source), model.pair.predict(source))
    original = {k: v.clone() for k, v in model.pair.state_dict().items()}
    model.train()
    assert all(not m.training for m in model.pair.modules())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    logits, _ = model(source)
    F.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
    assert all(p.grad is None for p in model.pair.parameters())
    assert model.upper[-1].weight.grad.abs().sum() > 0
    assert model.paddle[-1].weight.grad.abs().sum() > 0
    opt.step()
    assert all(torch.equal(v, original[k]) for k, v in model.pair.state_dict().items())
    assert torch.equal(source, before_source)
    assert torch.equal(model.predict(source[2:]), model.pair.predict(source[2:]))
    buffer = io.BytesIO()
    torch.save(dict(model_spec=spec, model=model.state_dict()), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    replica = build_model(saved["model_spec"])
    replica.load_state_dict(saved["model"])
    assert torch.equal(replica.predict(source), model.predict(source))
    assert replica.predict(source[:0]).shape == (0, 2)
    with pytest.raises(ValueError, match="coupled"):
        build_model(dict(spec, pair=dict(position=position)))
