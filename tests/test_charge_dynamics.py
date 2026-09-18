import copy

import pytest
import torch

from gymemu.commands.dynamics import compose_state_config
from gymemu.models import build_model
from gymemu.paddle_transition_training import evaluate, load_checkpoint, train
from gymemu.state_approach import StateApproach


@pytest.mark.parametrize("kind", ["state_mlp", "state_gru", "state_paddle_context_probe"])
def test_velocity_is_not_an_input_or_output(kind):
    spec = dict(
        kind=kind, history=2, past_actions=1, width=8, depth=1, predict_paddle_velocity=False
    )
    if kind == "state_paddle_context_probe":
        spec["paddle_context"] = 2
    model = build_model(spec)
    with torch.no_grad():
        model.head.weight.normal_()
    history = torch.rand(2, 2, 115)
    valid = torch.ones(2, 2, dtype=torch.bool)
    actions = torch.tensor([[0, 1], [2, 0]])
    context = torch.rand(2, 2, 8)
    extras = {"paddle_context": context} if kind == "state_paddle_context_probe" else {}
    first, _ = model(history, valid, actions, **extras)
    history[..., 5] = float("nan")
    context[..., 1] = float("nan")
    second, _ = model(history, valid, actions, **extras)
    for key in first:
        torch.testing.assert_close(first[key], second[key])
    assert (second["motion"][:, 5] == 0).all()
    assert model.head.out_features == 115


def test_velocity_targets_do_not_affect_loss():
    config = compose_state_config()
    config["model"].update(width=8, depth=1)
    approach = StateApproach(config["model"], config["loss"])
    batch = {
        "history": torch.rand(2, 1, 115),
        "history_valid": torch.ones(2, 1, dtype=torch.bool),
        "action_history": torch.zeros(2, 1, dtype=torch.long),
        "actions": torch.zeros(2, 1, dtype=torch.long),
        "target": torch.rand(2, 1, 115),
        "valid": torch.ones(2, 1, dtype=torch.bool),
        "terminal": torch.zeros(2, 1, dtype=torch.bool),
    }
    first, _ = approach.loss(batch)
    changed = copy.deepcopy(batch)
    changed["target"][..., 5] = float("nan")
    second, _ = approach.loss(changed)
    torch.testing.assert_close(first, second)


def test_charge_discrete_classes_metrics_and_checkpoint(tmp_path):
    data = {
        "x": torch.tensor(
            [[50.0, 1900.0, 0.0, 60.0, 0.0, 1.0], [51.0, 2000.0, 0.0, 60.0, 1.0, 2.0]]
        ),
        "y": torch.tensor([-120.0, 120.0]),
        "episodes": torch.tensor([1, 2]).numpy(),
        "steps": torch.tensor([1, 2]).numpy(),
        "early": torch.tensor([True, False]).numpy(),
        "target": "paddle_charge",
        "identity": "synthetic",
        "cache_identity": "synthetic",
        "hidden_provenance": {},
    }
    validation = dict(data, episodes=torch.tensor([3, 4]).numpy())
    result = train(
        data,
        validation,
        tmp_path / "charge",
        epochs=1,
        width=8,
        depth=1,
        objective="classification",
        encoding="hybrid",
        controller_fields=["charge"],
    )
    assert "mae_native" in result["validation"]
    assert "mae_pixels" not in result["validation"]
    model, checkpoint = load_checkpoint(tmp_path / "charge/best.pt")
    assert checkpoint["config"]["target"] == "paddle_charge"
    assert checkpoint["model_spec"]["delta_values"] == [-120, 120]
    assert model.delta(torch.tensor([[0.0, 1.0]])).item() == 120
    assert evaluate(model, validation)["target"] == "paddle_charge"
