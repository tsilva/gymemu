import numpy as np
import torch

from gymemu.models import build_model
from gymemu.state_controller_history import controller_history_inputs


def test_history_infers_current_state_without_current_action_or_crossing_life_boundary():
    context = {
        "source_rows": np.arange(6),
        "states": np.tile([0.5, 0.0, 1.0], (6, 1)).astype(np.float32),
        "valid": np.ones(6, bool),
        "actions": np.array([0, 1, 2, 0, 1, 2]),
        "starts": np.array([0, 0, 0, 3, 3, 3]),
    }
    x, complete = controller_history_inputs(context, [2, 4], 2)
    assert x[:, :, 4].tolist() == [[0, 1], [3, 0]]
    assert complete.tolist() == [True, False]
    assert x[0, -1, :4].tolist() == [80, 0, 16, 1]
    context["actions"][2:] = [1, 0, 2, 1]
    context["states"][5] = 0
    changed, _ = controller_history_inputs(context, [2, 4], 2)
    torch.testing.assert_close(x, changed)
    padded, _ = controller_history_inputs(context, [3], 2)
    assert padded[0, 0].tolist() == [0, 0, 0, 0, 3]
    assert padded[0, 1, 4] == 3


def test_registered_history_model_roundtrip_and_invalid_observation_mask(tmp_path):
    spec = {"kind": "controller_history_mlp", "history": 2, "values": [3, 61, 125], "width": 8}
    model = build_model(spec)
    x = torch.tensor([[[0, 0, 0, 0, 3], [80, -2, 16, 1, 1]]], dtype=torch.float32)
    changed = x.clone()
    changed[0, 0, :3] = torch.tensor([100, 20, 5])
    torch.testing.assert_close(model(x), model(changed))
    assert model.encode(x).shape == (1, 60)
    assert int(model.predict(x)) in [3, 61, 125]
    path = tmp_path / "probe.pt"
    torch.save({"model_spec": spec, "model": model.state_dict()}, path)
    checkpoint = torch.load(path, weights_only=True)
    restored = build_model(checkpoint["model_spec"])
    restored.load_state_dict(checkpoint["model"])
    torch.testing.assert_close(model(x), restored(x))
