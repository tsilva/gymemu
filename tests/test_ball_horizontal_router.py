"""Source-only routing, frozen expert and portable composite checkpoint contracts."""

import io

import pytest
import torch
from torch.nn import functional as F

from gymemu.models import build_model


def router_spec():
    return dict(
        kind="ball_horizontal_router",
        global_model=dict(width=16, depth=1),
        paddle_model=dict(
            direction=dict(paddle_values=[-1, 0, 1], width=16, depth=1), width=16, depth=1
        ),
    )


def test_routing_boundaries_frozen_training_and_portable_checkpoint():
    model = build_model(router_spec())
    source = torch.zeros(8, 118)
    source[:, :9] = torch.tensor([80, 170, 1, 1, 75, 16, 2048, 2, 0])
    source[:, 1] = torch.tensor([159, 160, 183, 184, 170, 170, 160, 183])
    source[:, 3] = torch.tensor([1, 1, 1, 1, 0, -1, -1, -1])
    gate = model.paddle_region(source)
    assert gate.tolist() == [False, True, True, False, False, False, False, False]
    with torch.no_grad():
        model.global_model.network[-1].weight.zero_()
        model.global_model.network[-1].bias.fill_(-10)
        model.global_model.network[-1].bias[0] = 10
        model.paddle_model.network[-1].weight.zero_()
        model.paddle_model.network[-1].bias.fill_(-10)
        model.paddle_model.network[-1].bias[0] = 10
    prediction = model.predict(source)
    assert torch.equal(prediction[~gate], torch.full((6,), -2.0))
    assert torch.equal(prediction[gate].abs(), torch.full((2,), 0.5))
    assert torch.equal(model.predict(source.long()), prediction)
    before = {k: v.clone() for k, v in model.paddle_model.state_dict().items()}
    model.train()
    assert not any(m.training for m in model.paddle_model.modules())
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    F.cross_entropy(model.global_model(source[~gate]), torch.ones(6, dtype=torch.long)).backward()
    opt.step()
    assert all(torch.equal(before[k], v) for k, v in model.paddle_model.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in model.paddle_model.parameters())
    assert torch.equal(model.predict(source[gate]), prediction[gate])
    assert model.predict(source[:0]).shape == (0,)
    checkpoint = io.BytesIO()
    torch.save(dict(spec=router_spec(), weights=model.state_dict()), checkpoint)
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    restored = build_model(saved["spec"])
    restored.load_state_dict(saved["weights"])
    assert torch.equal(restored.predict(source), model.predict(source))
    with pytest.raises(ValueError, match="118"):
        model.predict(source[:, :9])


@pytest.mark.parametrize("options", [dict(paddle_only=True), dict(objective="direction")])
def test_router_rejects_incompatible_global_expert(options):
    spec = router_spec()
    spec["global_model"].update(options)
    with pytest.raises(ValueError, match="full-field"):
        build_model(spec)
