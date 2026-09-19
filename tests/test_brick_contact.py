"""Contact classification must preserve every pretrained transition component."""

import io

import pytest
import torch
from test_ball_position import inputs
from test_brick_layout import specification as layout_specification
from torch.nn import functional as F

from gymemu.models import build_model


@pytest.mark.parametrize("collision_geometry", [False, True])
def test_contact_training_preserves_layout_and_ball_predictions_on_reload(
    collision_geometry,
):
    layout = layout_specification()
    layout.pop("kind")
    spec = dict(
        kind="brick_contact",
        layout=layout,
        width=16,
        depth=1,
        collision_geometry=collision_geometry,
    )
    model = build_model(spec)
    x = inputs()
    x[1, 10:] = 0
    parent = {k: v.clone() for k, v in model.layout.state_dict().items()}
    before = model.layout.predict(x)
    ball = model.layout.position.predict(x)
    velocity = model.layout.position.predict_velocities(x)
    model.train()
    assert not any(m.training for m in model.layout.modules())
    encoded = model.encode(x)
    assert encoded.shape == (3, 86 if collision_geometry else 73)
    assert torch.isfinite(encoded).all()
    assert encoded[1, 71:73].tolist() == [1, 0]
    if collision_geometry:
        assert not encoded[1, 73:].any()
    changed = x.clone()
    changed[:, 4:8] += 3
    assert torch.equal(model(x), model(changed))
    optimizer = torch.optim.AdamW(model.network.parameters())
    loss = F.cross_entropy(model(x), torch.tensor([0, 1, 0]))
    loss.backward()
    optimizer.step()
    assert all(torch.equal(v, parent[k]) for k, v in model.layout.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in model.layout.parameters())
    buffer = io.BytesIO()
    torch.save(dict(model_spec=spec, model=model.state_dict()), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    replica = build_model(checkpoint["model_spec"])
    replica.load_state_dict(checkpoint["model"])
    assert torch.equal(replica.predict(x), model.predict(x))
    assert torch.equal(replica.layout.predict(x), before)
    assert torch.equal(replica.layout.position.predict(x), ball)
    assert torch.equal(replica.layout.position.predict_velocities(x), velocity)
    assert replica.predict(x[:0]).shape == (0,)
