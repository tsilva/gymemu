import copy

import pytest
import torch
from test_scheduled_sampling import Spy, configured, model_for

from gymemu.models.action_history import ActionHistoryAutoencoder


@pytest.mark.parametrize("shape", [(3, 32, 32), (3, 21, 17), (3, 210, 160)])
def test_factored_actions_preserve_outputs_and_parameter_gradients(shape):
    torch.set_num_threads(2)
    reference = ActionHistoryAutoencoder(2, 3, shape, width=4)
    fast = ActionHistoryAutoencoder(2, 3, shape, width=4, factor_actions=True)
    fast.load_state_dict(reference.state_dict())
    history = torch.rand(2, 2, *shape)
    actions = torch.tensor([[3, 0], [1, 2]])  # START and every native action.
    target = torch.rand(2, *shape)
    a, b = reference(history, actions), fast.factored_forward(history, actions)
    assert torch.equal(a, fast(history, actions))  # Float32 inference keeps the reference path.
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
    (a - target).square().mean().backward()
    (b - target).square().mean().backward()
    for pa, pb in zip(reference.parameters(), fast.parameters(), strict=True):
        torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize("probability", [0.0, 0.13, 0.8, 1.0])
def test_selective_feedback_matches_dense_with_identical_masks(probability):
    dense = model_for(configured())
    dense.predictor = Spy()
    fast = copy.deepcopy(dense)
    # Different replacement times per row, with zero padding and valid STARTs.
    recorded = torch.rand(19, 4, 3, 21, 17)
    actions = torch.zeros(19, 3, 2, dtype=torch.long)
    actions[0, :2] = -1
    actions[1, 0] = -1
    actions[1, 1] = 2
    recorded[0] = 0
    recorded[1, :3] = 0
    masks = torch.rand(2, 19) < probability
    expected = dense.mixed_history(recorded, actions, masks=masks)
    actual = fast.selective_history(recorded, actions, masks=masks)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not actual.requires_grad
    assert not actual[0].any()
    for _, _, requires_grad in fast.predictor.calls:
        assert not requires_grad
    # Unselected steps do no inference, including p=0.
    assert len(fast.predictor.calls) == int(masks.sum(dim=0).max())


def test_fast_recipe_is_explicit_and_preserves_training_budget():
    from gymemu.config import compose_config

    base = compose_config(["recipe=breakout_scheduled"])
    fast = compose_config(["recipe=breakout_scheduled_fast"])
    assert fast.history == base.history
    assert fast.trainer == base.trainer
    assert fast.approach.options.schedule == base.approach.options.schedule
    assert fast.approach.options.rollout_steps == base.approach.options.rollout_steps
    assert fast.model.factor_actions is True
    assert fast.approach.options.feedback_dtype == "autocast"


def test_reordered_cnn_feedback_preserves_final_loss_and_gradients():
    dense = model_for(configured())
    fast = copy.deepcopy(dense)
    fast.selective_threshold = 1.0
    dense.begin_epoch(10)
    fast.begin_epoch(10)
    history = torch.rand(19, 4, 3, 21, 17)
    actions = torch.randint(0, 3, (19, 3, 2))
    target = torch.rand(19, 3, 21, 17)
    torch.manual_seed(42)
    expected = dense.loss(history, actions, target)
    torch.manual_seed(42)
    actual = fast.loss(history, actions, target)
    torch.testing.assert_close(actual, expected)
    expected.backward()
    actual.backward()
    for a, b in zip(dense.parameters(), fast.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-8, rtol=1e-4)


def test_dense_probability_updates_reuse_one_compiled_graph():
    from gymemu.engine import batch_loss

    model = model_for(configured())
    model.feedback_dtype = "autocast"
    model.selective_threshold = 0.2
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(batch_loss, backend=backend, fullgraph=True)
    history = torch.zeros(2, 4, 3, 21, 17, dtype=torch.uint8)
    actions = torch.zeros(2, 3, 2, dtype=torch.long)
    target = torch.zeros(2, 3, 21, 17, dtype=torch.uint8)
    for epoch in (4, 5, 8):
        model.begin_epoch(epoch)
        loss, _ = compiled(model, history, actions, target, True, True)
        assert torch.isfinite(loss)
    assert len(graphs) == 1


@pytest.mark.parametrize(
    "override",
    [
        "+approach.options.feedback_dtype=int8",
        "+approach.options.selective_threshold=1.1",
    ],
)
def test_invalid_execution_options_rejected(override):
    with pytest.raises(ValueError):
        model_for(configured(override))
