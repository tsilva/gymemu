"""Coordinate decoding, portable checkpoints, and frozen velocity training boundaries."""

import io

import pytest
import torch
from torch.nn import functional as F

from gymemu.models import build_model


def specification(axis):
    return dict(
        kind="ball_position",
        axis=axis,
        values=[-1.875, 0, 1.375],
        upper_width=16,
        upper_depth=1,
        paddle_width=16,
        paddle_depth=1,
        flight_width=16,
        flight_depth=1,
        vertical=dict(
            values=[-2, 2],
            upper_width=16,
            upper_depth=1,
            paddle_width=16,
            paddle_depth=1,
            horizontal=dict(
                width=8,
                depth=1,
                head_width=16,
                base=dict(
                    global_model=dict(width=16, depth=1),
                    paddle_model=dict(
                        direction=dict(
                            paddle_values=[-1, 0, 1],
                            width=16,
                            depth=1,
                            encoding="hybrid_absolute",
                        ),
                        width=16,
                        depth=1,
                    ),
                ),
            ),
        ),
    )


def inputs():
    x = torch.zeros(3, 118)
    x[:, :10] = torch.tensor([80.5, 100, -1.5, 2, 72, 16, 2000, 5, 7, 0])
    x[:, 1] = torch.tensor([100, 170, 120])
    x[:, 10:] = 1
    return x


@pytest.mark.parametrize("axis", ["x", "y"])
def test_position_heads_decode_and_preserve_both_velocity_models(axis):
    spec = specification(axis)
    model = build_model(spec)
    x = inputs()
    for network in (model.upper_network, model.paddle_network, model.flight_network):
        with torch.no_grad():
            network[-1].weight.zero_()
            network[-1].bias.fill_(-10)
            network[-1].bias[2] = 10
    if axis == "y":
        assert model.predict(x).tolist() == [102.25, 172.25, 122.25]
        assert model.predict_y_parts(x).tolist() == [[102, 2], [172, 2], [122, 2]]
    else:
        assert model.predict(x).tolist() == [81.875] * 3
        with pytest.raises(ValueError, match="Only the y"):
            model.predict_y_parts(x)
    frozen = {k: v.clone() for k, v in model.vertical.state_dict().items()}
    velocities = model.predict_velocities(x)
    model.train()
    assert not any(m.training for m in model.vertical.modules())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    F.cross_entropy(model(x), torch.tensor([0, 1, 2])).backward()
    optimizer.step()
    assert all(torch.equal(v, frozen[k]) for k, v in model.vertical.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in model.vertical.parameters())
    assert torch.equal(velocities, model.predict_velocities(x))
    buffer = io.BytesIO()
    torch.save(dict(model_spec=spec, model=model.state_dict()), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    restored = build_model(checkpoint["model_spec"])
    restored.load_state_dict(checkpoint["model"])
    assert torch.equal(restored.predict(x), model.predict(x))
    assert torch.equal(restored.predict_velocities(x), velocities)
    assert restored.predict(x[:0]).shape == (0,)


def test_position_flight_input_contract_and_fraction_borrow():
    x = inputs()
    horizontal = build_model(specification("x"))
    vertical = build_model(specification("y"))
    changed = x.clone()
    changed[:, [1, 4, 5, 6, 7, 8, 9]] += 1
    changed[:, 10:] = 0
    assert horizontal.flight_encode(x).shape == (3, 31)
    assert torch.equal(horizontal.flight_encode(x), horizontal.flight_encode(changed))
    assert vertical.flight_encode(x).shape == (3, 1)
    assert torch.equal(vertical.flight_encode(x), vertical.flight_encode(changed))
    with torch.no_grad():
        vertical.flight_network[-1].weight.zero_()
        vertical.flight_network[-1].bias.copy_(torch.tensor([10, -10, -10]))
    x = x[2:].clone()
    x[:, 8] = 0
    assert vertical.predict_y_parts(x).tolist() == [[118, 1]]
    spec = specification("x")
    for values in ([], [0, 0], [0.1], [float("nan")], [float("inf")]):
        spec["values"] = values
        with pytest.raises(ValueError, match="eighth-pixel"):
            build_model(spec)
