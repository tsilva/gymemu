"""Two-frame timing distinguishes equal outgoing speeds with different movement."""

import io

import pytest
import torch
from test_ball_position import inputs, specification
from torch.nn import functional as F

from gymemu.models import build_model


def model_spec():
    position = specification("y")
    position.pop("kind")
    return dict(kind="vertical_collision_timing", pair=dict(position=position, coupling_width=16))


def test_first_and_second_frame_bounces_decode_different_displacements():
    model = build_model(model_spec())
    source = inputs()
    for outcome, displacement in [(0, -4), (2, 0)]:
        # Speeds are [-2, +2]. Outcomes 0 and 2 both finish at -2.
        for head in (model.upper, model.paddle):
            with torch.no_grad():
                head[-1].weight.zero_()
                head[-1].bias.fill_(-100)
                head[-1].bias[outcome] = 100
        prediction = model.predict(source)
        assert torch.equal(prediction[:2, 0], source[:2, 1] + source[:2, 8] / 8 + displacement)
        assert (prediction[:2, 1] == -2).all()
        assert torch.equal(prediction[2:], model.pair.predict(source[2:]))
    model.active_regions = ("upper",)
    assert torch.equal(model.predict(source)[1:], model.pair.predict(source[1:]))
    logits = torch.eye(4) * 100
    timing = model.timing_log_probabilities(logits, torch.full((4,), 2.0)).argmax(1)
    assert timing.tolist() == [1, 3, 2, 0]


@pytest.mark.parametrize("prior_weight", [0.0, 2.0])
def test_joint_and_timing_losses_train_heads_preserve_parent_and_reload(prior_weight):
    spec = dict(model_spec(), prior_weight=prior_weight)
    model = build_model(spec)
    model.train()
    assert all(not m.training for m in model.pair.modules())
    frozen = {k: v.clone() for k, v in model.pair.state_dict().items()}
    source = inputs()
    regions, features = model.encode(source)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    loss = 0
    for name, mask in zip(("upper", "paddle"), regions[:2]):
        logits = getattr(model, name)(features[name])
        loss += F.cross_entropy(logits, torch.zeros(int(mask.sum()), dtype=torch.long))
        loss += F.nll_loss(
            model.timing_log_probabilities(logits, source[mask, 3]),
            torch.ones(int(mask.sum()), dtype=torch.long),
        )
    loss.backward()
    assert all(p.grad is None for p in model.pair.parameters())
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )
    opt.step()
    assert all(torch.equal(v, frozen[k]) for k, v in model.pair.state_dict().items())
    buffer = io.BytesIO()
    torch.save(dict(model_spec=spec, model=model.state_dict()), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = build_model(saved["model_spec"])
    restored.load_state_dict(saved["model"])
    assert torch.equal(restored.predict(source), model.predict(source))
    assert restored.predict(source[:0]).shape == (0, 2)
