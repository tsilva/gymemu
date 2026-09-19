"""Width inputs, categorical decoding, and portable learned checkpoints."""

import io

import pytest
import torch
from test_ball_position import inputs
from torch.nn import functional as F

from gymemu.models import build_model


@pytest.mark.parametrize("proposal", [False, True])
def test_width_depends_only_on_vertical_state_and_current_width(proposal):
    model = build_model(dict(kind="paddle_width", proposal=proposal))
    source = inputs()
    changed = source.clone()
    changed[:, [0, 2, 4, 6, 7, 9]] += 1
    changed[:, 10:] = 1 - changed[:, 10:]
    assert model.encode(source).shape == (3, 33 if proposal else 21)
    assert torch.equal(model(source), model(changed))
    fractional = source.clone()
    fractional[:, 8] = 0
    assert not torch.equal(model.encode(source), model.encode(fractional))
    with pytest.raises(ValueError, match="118"):
        model(source[:, :117])


@pytest.mark.parametrize("proposal", [False, True])
def test_width_training_checkpoint_and_two_value_decoder(proposal):
    spec = dict(kind="paddle_width", width=16, depth=1, proposal=proposal)
    model = build_model(spec)
    source = inputs()
    optimizer = torch.optim.AdamW(model.parameters())
    loss = F.cross_entropy(model(source), torch.tensor([0, 1, 0]))
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    optimizer.step()
    buffer = io.BytesIO()
    torch.save(dict(model_spec=spec, model=model.state_dict()), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    replica = build_model(checkpoint["model_spec"])
    replica.load_state_dict(checkpoint["model"])
    assert torch.equal(replica.predict(source), model.predict(source))
    assert replica.predict(source[:0]).shape == (0,)
    for cls, width in enumerate([12, 16]):
        with torch.no_grad():
            replica.network[-1].weight.zero_()
            replica.network[-1].bias.zero_()
            replica.network[-1].bias[cls] = 10
        assert (replica.predict(source) == width).all()
