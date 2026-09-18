"""Frozen direction, speed vocabulary, composition and portable checkpoint contracts."""

import io

import pytest
import torch
from torch.nn import functional as F

from gymemu.models import build_model


def test_speed_training_preserves_direction_and_checkpoint_predictions():
    spec = dict(
        kind="ball_horizontal_velocity",
        direction=dict(paddle_values=[-1, 0, 1], encoding="hybrid_absolute", width=16, depth=1),
        width=16,
        depth=1,
    )
    model = build_model(spec)
    source = torch.tensor(
        [[80.5, 173, -1.5, 3.375, 72, 16, 2048, 7, 5], [22, 180, 1, 1.5, 20, 12, 1000, 2, 0]]
    )
    before = {k: v.clone() for k, v in model.direction.state_dict().items()}
    expected_sign = model.direction.predict(source)
    model.train()
    assert not model.direction.training and not model.direction.paddle.training
    assert all(not p.requires_grad for p in model.direction.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = F.cross_entropy(model(source), torch.tensor([0, 3]))
    loss.backward()
    optimizer.step()
    assert all(torch.equal(before[k], v) for k, v in model.direction.state_dict().items())
    assert all(p.grad is None for p in model.direction.parameters())
    assert torch.equal(model.direction.predict(source), expected_sign)
    assert torch.equal(model.predict(source), expected_sign * model.predict_speed(source))
    assert torch.isin(model.predict_speed(source), torch.tensor([0.5, 1, 1.5, 2])).all()
    checkpoint = io.BytesIO()
    torch.save(dict(spec=spec, weights=model.state_dict()), checkpoint)
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    restored = build_model(saved["spec"])
    restored.load_state_dict(saved["weights"])
    assert torch.equal(restored.predict(source), model.predict(source))
    with pytest.raises(ValueError, match="Expected 9"):
        restored(torch.zeros(1, 118))


@pytest.mark.parametrize("options", [{"width": 0}, {"depth": 0}, {"activation": "unknown"}])
def test_speed_network_rejects_invalid_configuration(options):
    with pytest.raises(ValueError, match="Invalid speed"):
        build_model(
            dict(kind="ball_horizontal_velocity", direction=dict(paddle_values=[0]), **options)
        )
