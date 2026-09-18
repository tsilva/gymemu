import pytest
import torch

from gymemu.models.paddle_transition import PaddleTransitionMLP
from gymemu.paddle_transition_training import FORMAT, load_checkpoint


def test_reduced_inputs_exclude_measurement_and_survive_checkpoint(tmp_path):
    spec = {
        "kind": "paddle_transition_mlp",
        "encoding": "hybrid",
        "objective": "classification",
        "controller_fields": ["charge", "repeat", "held"],
    }
    model = PaddleTransitionMLP(**{k: v for k, v in spec.items() if k != "kind"})
    source = torch.tensor([[50.0, 1703.0, 132.0, 60.0, 1.0, 2.0]])
    changed = source.clone()
    changed[:, 2] = float("nan")
    torch.testing.assert_close(model(source), model(changed))
    assert model.encode(source).shape == (1, 34)
    changed = source.clone()
    changed[:, 1] += 1
    assert not torch.equal(model.encode(source), model.encode(changed))
    path = tmp_path / "model.pt"
    torch.save({"format": FORMAT, "model_spec": spec, "model": model.state_dict()}, path)
    restored, _ = load_checkpoint(path)
    torch.testing.assert_close(model(source), restored(source))


def test_old_checkpoint_and_empty_controller_subset(tmp_path):
    spec = {"kind": "paddle_transition_mlp", "encoding": "hybrid"}
    old = PaddleTransitionMLP(encoding="hybrid")
    path = tmp_path / "old.pt"
    torch.save({"format": FORMAT, "model_spec": spec, "model": old.state_dict()}, path)
    restored, _ = load_checkpoint(path)
    source = torch.tensor([[50.0, 1703.0, 132.0, 60.0, 1.0, 2.0]])
    torch.testing.assert_close(old(source), restored(source))
    small = PaddleTransitionMLP(encoding="hybrid", controller_fields=[])
    changed = source.clone()
    changed[:, 1:5] = float("nan")
    torch.testing.assert_close(small(source), small(changed))
    assert small.encode(source).shape == (1, 12)


@pytest.mark.parametrize("fields", [["charge", "charge"], ["unknown"]])
def test_reject_invalid_controller_fields(fields):
    with pytest.raises(ValueError, match="controller fields"):
        PaddleTransitionMLP(controller_fields=fields)
