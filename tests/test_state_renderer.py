import io

import pytest
import torch

from gymemu.models import build_model
from gymemu.models.state_renderer import StateRenderer


def test_renderer_can_learn_state_dependent_pixel_colors():
    torch.manual_seed(42)
    model = StateRenderer([[0, 0, 0], [200, 72, 72]], width=16, layers=2)
    state = torch.zeros(2, 112)
    state[:, :4] = torch.tensor([80, 150, 80, 16])
    state[1, 4] = 1
    point = torch.tensor([[[10, 59]], [[10, 59]]])
    target = torch.tensor([0, 1])
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(60):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(state, point)[:, 0], target)
        loss.backward()
        opt.step()
    assert torch.equal(model(state, point)[:, 0].argmax(-1), target)


def test_renderer_registered_checkpoint_and_full_image():
    spec = dict(kind="state_renderer", palette=[[0, 0, 0], [200, 72, 72]], width=8)
    model = build_model(spec).eval()
    state = torch.zeros(1, 112)
    state[:, :4] = torch.tensor([80, 150, 80, 16])
    saved = io.BytesIO()
    torch.save(dict(model_spec=spec, model=model.state_dict()), saved)
    saved.seek(0)
    checkpoint = torch.load(saved, weights_only=True)
    restored = build_model(checkpoint["model_spec"])
    restored.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        expected, actual = model.render(state), restored.render(state)
    assert torch.equal(expected, actual)
    assert actual.shape == (1, 3, 210, 160)
    assert actual[:, :, :17].eq(0).all()
    assert actual.isfinite().all()


def test_renderer_maps_dynamics_fields_and_refuses_terminal_placeholders():
    prediction = torch.arange(118).float()[None]
    prediction[:, 1] = 97.875
    prediction[:, 117] = 0
    state = StateRenderer.from_dynamics(prediction)
    assert state[0, :4].tolist() == [0, 97, 116, 114]
    assert torch.equal(state[:, 4:], prediction[:, 4:112])
    prediction[:, 117] = 1
    with pytest.raises(ValueError, match="terminal"):
        StateRenderer.from_dynamics(prediction)
