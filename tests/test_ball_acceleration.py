"""Frozen full-field model, source-only acceleration domain and input contracts."""

import io

import pytest
import torch
from torch.nn import functional as F

from gymemu.models import build_model


def spec(geometry=False):
    return dict(
        kind="ball_acceleration",
        width=16,
        depth=1,
        geometry=geometry,
        base=dict(
            global_model=dict(width=16, depth=1),
            paddle_model=dict(
                direction=dict(paddle_values=[-1, 0, 1], width=16, depth=1),
                width=16,
                depth=1,
            ),
        ),
    )


def source_rows():
    x = torch.zeros(5, 118)
    x[:, :10] = torch.tensor([80.5, 55, -1.5, -2, 72, 16, 2000, 5, 3, 0])
    x[:, 10:] = 1
    x[:, 1] = torch.tensor([55, 100, 101, 55, 173])
    x[3, 2] = 2
    x[4, 3] = 2
    return x


@pytest.mark.parametrize("geometry,features", [(False, 149), (True, 179)])
def test_acceleration_input_isolation_frozen_base_and_roundtrip(geometry, features):
    model = build_model(spec(geometry))
    x = source_rows()
    assert model.region(x).tolist() == [True, True, False, False, False]
    assert model.encode(x).shape == (5, features)
    altered = x.clone()
    altered[:, 4:8] += 10
    assert torch.equal(model.encode(x), model.encode(altered))
    altered[:, 9] = 1
    assert not torch.equal(model.encode(x), model.encode(altered))
    with torch.no_grad():
        model.network[-1].weight.zero_()
        model.network[-1].bias.copy_(torch.tensor([10.0, -10.0]))
    parent = model.base.predict(x)
    retained = model.predict(x)
    assert torch.equal(retained[:2].abs(), x[:2, 2].abs())
    assert torch.equal(retained[2:], parent[2:])
    with torch.no_grad():
        model.network[-1].bias.copy_(torch.tensor([-10.0, 10.0]))
    accelerated = model.predict(x)
    assert torch.equal(accelerated[:2].abs(), torch.full((2,), 2.0))
    assert torch.equal(accelerated.sign(), parent.sign())
    assert torch.equal(accelerated[2:], parent[2:])
    frozen = {k: v.clone() for k, v in model.base.state_dict().items()}
    model.train()
    assert not any(m.training for m in model.base.modules())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    F.cross_entropy(model(x[:2]), torch.tensor([0, 1])).backward()
    optimizer.step()
    assert all(torch.equal(v, frozen[k]) for k, v in model.base.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in model.base.parameters())
    file = io.BytesIO()
    torch.save(dict(model_spec=spec(geometry), model=model.state_dict()), file)
    file.seek(0)
    cp = torch.load(file, weights_only=True)
    loaded = build_model(cp["model_spec"])
    loaded.load_state_dict(cp["model"])
    assert torch.equal(loaded.predict(x), model.predict(x))
    assert loaded.predict(x[:0]).shape == (0,)
    with pytest.raises(ValueError, match="118"):
        loaded(x[:, :9])


def test_spatial_acceleration_uses_only_present_bricks_and_portable_frozen_base():
    configuration = spec()
    configuration.pop("geometry")
    configuration.update(kind="ball_acceleration_spatial", head_width=16)
    model = build_model(configuration)
    x = source_rows()
    cells, numbers, occupied = model.encode(x)
    assert cells.shape == (5, 108, 13)
    assert numbers.shape == (5, 6)
    altered = x.clone()
    altered[:, 4:8] += 10
    assert torch.equal(model(x), model(altered))
    occupied[:] = False
    expected = model.forward_encoded(cells, numbers, occupied)
    assert torch.equal(expected, model.forward_encoded(cells + 100, numbers, occupied))
    frozen = {k: v.clone() for k, v in model.base.state_dict().items()}
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    F.cross_entropy(model(x), torch.tensor([0, 1, 0, 0, 0])).backward()
    optimizer.step()
    assert all(torch.equal(v, frozen[k]) for k, v in model.base.state_dict().items())
    assert not any(m.training for m in model.base.modules())
    buffer = io.BytesIO()
    torch.save(dict(model_spec=configuration, model=model.state_dict()), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    loaded = build_model(checkpoint["model_spec"])
    loaded.load_state_dict(checkpoint["model"])
    assert torch.equal(loaded.predict(x), model.predict(x))
    assert loaded.predict(x[:0]).shape == (0,)
