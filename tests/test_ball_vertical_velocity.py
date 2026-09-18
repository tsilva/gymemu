"""Source-only routing and immutable horizontal checkpoint during vertical training."""

import io

import pytest
import torch
from torch.nn import functional as F

from gymemu.models import build_model


def specification():
    return dict(
        kind="ball_vertical_velocity",
        values=[-3.375, -2, -1.5, -1, 1, 1.5, 2, 3.375],
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
                        paddle_values=[-1, 0, 1], width=16, depth=1, encoding="hybrid_absolute"
                    ),
                    width=16,
                    depth=1,
                ),
            ),
        ),
    )


def inputs():
    x = torch.zeros(8, 118)
    x[:, :10] = torch.tensor([80.5, 55, -1.5, 2, 72, 16, 2000, 5, 3, 0])
    x[:, 10:] = 1
    x[:, 1] = torch.tensor([100, 101, 159, 160, 183, 184, 170, 31])
    x[6, 3] = -2
    return x


def test_vertical_regions_training_and_portable_horizontal_preservation():
    spec = specification()
    model = build_model(spec)
    x = inputs()
    upper, paddle, flight = model.regions(x)
    assert upper.tolist() == [True, False, False, False, False, False, False, True]
    assert paddle.tolist() == [False, False, False, True, True, False, False, False]
    assert flight.tolist() == [False, True, True, False, False, True, True, False]
    for network, category in [
        (model.upper_network, 0),
        (model.paddle_network, 1),
        (model.flight_network, 2),
    ]:
        with torch.no_grad():
            network[-1].weight.zero_()
            network[-1].bias.fill_(-10)
            network[-1].bias[category] = 10
    predicted = model.predict(x)
    assert torch.equal(predicted[upper], model.classes[0].expand(2))
    assert torch.equal(predicted[paddle], model.classes[1].expand(2))
    assert torch.equal(predicted[flight], model.classes[2].expand(4))
    frozen = {k: v.clone() for k, v in model.horizontal.state_dict().items()}
    horizontal = model.predict_horizontal(x)
    model.train()
    assert not any(m.training for m in model.horizontal.modules())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    F.cross_entropy(model(x), torch.tensor([1, 2, 3, 4, 5, 6, 7, 0])).backward()
    optimizer.step()
    assert all(torch.equal(v, frozen[k]) for k, v in model.horizontal.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in model.horizontal.parameters())
    assert torch.equal(horizontal, model.predict_horizontal(x))
    checkpoint = io.BytesIO()
    torch.save(dict(model_spec=spec, model=model.state_dict()), checkpoint)
    checkpoint.seek(0)
    cp = torch.load(checkpoint, weights_only=True)
    restored = build_model(cp["model_spec"])
    restored.load_state_dict(cp["model"])
    assert torch.equal(restored.predict(x), model.predict(x))
    assert torch.equal(restored.predict_horizontal(x), horizontal)
    assert restored.predict(x[:0]).shape == (0,)
    with pytest.raises(ValueError, match="118"):
        restored(x[:, :9])


def test_vertical_encoders_keep_charge_and_spatial_inputs_separate():
    model = build_model(specification())
    x = inputs()
    cells, numbers, occupied = model.upper_encode(x)
    assert cells.shape == (8, 108, 13) and numbers.shape == (8, 71)
    changed = x.clone()
    changed[:, 4:8] += 3
    other = model.upper_encode(changed)
    assert all(torch.equal(a, b) for a, b in zip((cells, numbers, occupied), other, strict=True))
    before = model.paddle_encode(x)
    after = model.paddle_encode(changed)
    assert before.shape == (8, 97)
    assert not torch.equal(before[:, -13:], after[:, -13:])
