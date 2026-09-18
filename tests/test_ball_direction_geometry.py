"""Contracts for a direction probe with a frozen learned paddle submodel."""

import io

import pytest
import torch

from gymemu.models import build_model


@pytest.mark.parametrize("encoding", ["scalar", "hybrid", "hybrid_absolute"])
def test_geometry_checkpoint_keeps_frozen_paddle_and_predictions(encoding):
    spec = dict(kind="ball_direction_geometry", paddle_values=[-1, 0, 1], encoding=encoding)
    model = build_model(spec)
    model.train()
    assert not model.paddle.training
    assert all(not p.requires_grad for p in model.paddle.parameters())
    x = torch.tensor([[80.5, 173, -1.5, 3.375, 72, 16, 2048, 7, 5]])
    before = {k: v.clone() for k, v in model.paddle.state_dict().items()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    loss = model(x).sum()
    loss.backward()
    opt.step()
    assert all(torch.equal(v, before[k]) for k, v in model.paddle.state_dict().items())
    encoded = model.encode(x)
    assert encoded.shape == (1, {"scalar": 16, "hybrid": 62, "hybrid_absolute": 84}[encoding])
    expected = model.predict(x)
    out = io.BytesIO()
    torch.save(dict(spec=spec, weights=model.state_dict()), out)
    out.seek(0)
    loaded = torch.load(out, weights_only=True)
    restored = build_model(loaded["spec"])
    restored.load_state_dict(loaded["weights"])
    assert torch.equal(restored.predict(x), expected)
    assert torch.isin(expected, torch.tensor([-1.0, 1.0])).all()
    with pytest.raises(ValueError, match="Expected 9"):
        restored(torch.zeros(1, 118))
