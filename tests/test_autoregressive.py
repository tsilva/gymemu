import copy

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from gymemu.approaches import build_approach
from gymemu.batches import CachedBatchLoader
from gymemu.cache import build_cache
from gymemu.checkpoints import load_model
from gymemu.config import compose_config
from gymemu.data import Frames, Windows, read_episodes
from gymemu.engine import train
from gymemu.scenes import load_scene
from play import Player


def configured(*extra):
    return compose_config(
        [
            "recipe=breakout_autoregressive_ball_region",
            "experiment=smoke",
            "approach.options.sprite_height=2",
            "approach.options.sprite_width=1",
            *extra,
        ]
    )


def model_for(cfg):
    return build_approach(
        OmegaConf.to_container(cfg.approach, resolve=True), cfg.history, 2, (3, 21, 17)
    )


def test_future_targets_actions_and_episode_boundaries(snapshot, tmp_path):
    frames = Frames(snapshot, compact=True)
    episodes = read_episodes(snapshot, "train")
    data = Windows(frames, episodes, 2, [0, 2], action_history=2, future_steps=8)
    history, actions, targets = data[1]
    assert not history[0].any()
    assert torch.equal(history[1], frames.get(10))
    assert torch.equal(targets[0], frames.get(20))
    assert torch.equal(targets[1], frames.get(30))
    assert not targets[2:].any()
    assert actions[:2].tolist() == [[2, 1], [1, 0]]
    assert actions[2:].eq(-1).all()
    assert data[2][1][1:].eq(-1).all()  # Last target retained; next episode excluded.
    assert not data[3][0].any()  # Next episode starts with empty RGB history.
    assert data[3][1][0].eq(2).all()  # Bootstrap has no game action.
    cache = build_cache(snapshot, tmp_path / "cache", workers=1)
    cached = Windows(
        Frames(snapshot, compact=True, cache=cache),
        episodes,
        2,
        [0, 2],
        action_history=2,
        future_steps=8,
    )
    order = [4, 0, 2, 3, 1]
    for standard, fast in zip(
        DataLoader(data, batch_size=3, sampler=order),
        CachedBatchLoader(cached, 3, sampler=order),
        strict=True,
    ):
        assert all(torch.equal(a, b) for a, b in zip(standard, fast, strict=True))


@pytest.mark.parametrize("detach, expected_gradient", [(False, 0.875), (True, 0.8125)])
def test_rollout_feedback_gradients_and_mask_normalization(detach, expected_gradient):
    model = model_for(configured("approach.options.ball_region_weight=0"))
    model.detach_feedback = detach

    class LinearSpy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.outputs = []
            self.inputs = []

        def forward(self, history, actions):
            self.inputs.append(history.clone())
            output = history[:, -1] * self.weight
            output.retain_grad()
            self.outputs.append(output)
            return output

    spy = LinearSpy()
    model.predictor = spy
    model.begin_epoch(2)  # Two steps, despite configured maximum of eight.
    history = torch.ones(2, 2, 3, 21, 17)
    actions = torch.tensor([[[0, 0], [0, 0]], [[0, 0], [-1, -1]]])
    targets = torch.zeros(2, 2, 3, 21, 17)
    targets[1, 1] = 100  # Invalid padding must not contribute.
    loss = model.loss(history, actions, targets)
    # Example means: (0.25 + 0.0625)/2 and 0.25; batch mean = 0.203125.
    assert loss.item() == pytest.approx(0.203125)
    assert torch.equal(spy.inputs[1][:, -1], spy.outputs[0])
    loss.backward()
    assert spy.weight.grad.item() == pytest.approx(expected_gradient)
    assert spy.outputs[1].grad[1].eq(0).all()
    assert model.epoch_metrics()["rgb_mse"] == pytest.approx(0.5625 / 3)
    assert [model.begin_epoch(i)["rollout_steps"] for i in range(1, 7)] == [1, 2, 4, 8, 8, 8]


def test_autocast_feedback_preserves_recurrence_gradients_and_float32_path():
    cfg = configured("+approach.options.feedback_dtype=autocast")
    model = model_for(cfg)
    model.begin_epoch(2)
    history = torch.rand(2, 2, 3, 21, 17)
    actions = torch.zeros(2, 2, 2, dtype=torch.long)
    target = torch.rand(2, 2, 3, 21, 17)
    seen = []
    outputs = []

    def observe(module, inputs, output):
        seen.append(inputs[0].dtype)
        output.retain_grad()
        outputs.append(output)

    handle = model.predictor.register_forward_hook(observe)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model.loss(history, actions, target)
    loss.backward()
    assert seen == [torch.bfloat16, torch.bfloat16]
    assert all(output.grad is not None and output.grad.abs().sum() > 0 for output in outputs)
    handle.remove()
    # Playback/evaluation and old recipes retain their original float32 arithmetic.
    reference = model_for(configured())
    reference.load_state_dict(model.state_dict())
    assert torch.equal(model(history, actions[:, 0]), reference(history, actions[:, 0]))


def test_invalid_autoregressive_feedback_dtype():
    with pytest.raises(ValueError, match="feedback_dtype"):
        model_for(configured("+approach.options.feedback_dtype=int8"))


def test_detached_feedback_cuts_entire_window_but_supervises_every_prediction():
    model = model_for(configured("+approach.options.detach_feedback=true"))
    model.begin_epoch(3)
    history = torch.rand(2, 2, 3, 21, 17, requires_grad=True)
    actions = torch.zeros(2, 4, 2, dtype=torch.long)
    target = torch.rand(2, 4, 3, 21, 17)
    inputs, outputs = [], []

    def observe(module, args, output):
        inputs.append(args[0])
        output.retain_grad()
        outputs.append(output)

    handle = model.predictor.register_forward_hook(observe)
    model.loss(history, actions, target).backward()
    handle.remove()
    assert len(outputs) == 4
    assert all(not value.requires_grad for value in inputs)
    assert history.grad is None
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in outputs)
    for index in range(1, 4):
        assert torch.equal(inputs[index][:, -1], outputs[index - 1])


@pytest.mark.parametrize("value", ["1", "null", "yes"])
def test_invalid_detach_feedback(value):
    with pytest.raises(ValueError, match="detach_feedback"):
        model_for(configured(f"+approach.options.detach_feedback={value}"))


def test_detached_recipe_preserves_matched_experiment_settings():
    original = compose_config(["recipe=breakout_autoregressive_fast"])
    detached = compose_config(["recipe=breakout_detached"])
    approach = OmegaConf.to_container(detached.approach, resolve=True)
    assert approach["options"].pop("detach_feedback") is True
    assert approach == OmegaConf.to_container(original.approach, resolve=True)
    assert detached.trainer == original.trainer
    assert detached.optimizer == original.optimizer
    assert detached.game == original.game


def test_detached_fast_recipe_keeps_objective_curriculum_and_inference():
    base = compose_config(["recipe=breakout_detached"])
    fast = compose_config(["recipe=breakout_detached_fast"])
    assert fast.approach.options == base.approach.options
    assert fast.approach.stages == base.approach.stages
    assert fast.optimizer == base.optimizer
    assert fast.game == base.game and fast.history == base.history
    assert fast.trainer.diagnostics == base.trainer.diagnostics
    models = [
        build_approach(OmegaConf.to_container(cfg.approach, resolve=True), 8, 3, (3, 21, 17))
        for cfg in (base, fast)
    ]
    models[1].load_state_dict(models[0].state_dict())
    history, actions = torch.rand(2, 8, 3, 21, 17), torch.zeros(2, 8, dtype=torch.long)
    assert torch.equal(models[0](history, actions), models[1](history, actions))


def test_compiled_detached_loss_matches_eager_through_curriculum_and_short_batch():
    eager = model_for(configured("+approach.options.detach_feedback=true"))
    compiled_model = copy.deepcopy(eager)
    compiled_loss = torch.compile(compiled_model.loss, backend="aot_eager", fullgraph=True)
    for epoch, size in [(1, 2), (2, 2), (3, 2), (3, 1)]:
        eager.begin_epoch(epoch)
        compiled_model.begin_epoch(epoch)
        history = torch.rand(size, 2, 3, 21, 17)
        actions = torch.zeros(size, 4, 2, dtype=torch.long)
        actions[0, -1] = -1
        targets = torch.rand(size, 4, 3, 21, 17)
        eager.zero_grad(set_to_none=True)
        compiled_model.zero_grad(set_to_none=True)
        expected = eager.loss(history, actions, targets)
        actual = compiled_loss(history, actions, targets)
        torch.testing.assert_close(actual, expected)
        expected.backward()
        actual.backward()
        for a, b in zip(eager.parameters(), compiled_model.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad)


def test_fast_recipe_preserves_objective_and_exposes_execution_changes():
    base = compose_config(["recipe=breakout_autoregressive_ball_region"])
    fast = compose_config(["recipe=breakout_autoregressive_fast"])
    assert base.trainer.batch_size == 8 and not base.trainer.compile
    assert fast.trainer.compile
    assert not fast.trainer.compile_layout_optimization
    assert not fast.approach.models.predictor.get("factor_actions", False)
    assert fast.approach.options.feedback_dtype == "autocast"
    for key, value in base.approach.options.items():
        assert fast.approach.options[key] == value
    assert fast.approach.stages == base.approach.stages
    assert fast.game == base.game and fast.history == base.history
    assert fast.trainer.diagnostics == base.trainer.diagnostics


@pytest.mark.parametrize("cached,detach", [(False, False), (True, False), (True, True)])
def test_train_checkpoint_and_play(snapshot, tmp_path, cached, detach):
    cfg = configured(
        f"recipe={'breakout_detached_fast' if detach else 'breakout_autoregressive_ball_region'}",
        "game=custom",
        "trainer.epochs=2",
        "approach.options.rollout_steps=3",
        "approach.options.rollout_schedule=[3]",
        f"++approach.options.detach_feedback={str(detach).lower()}",
    )
    cfg.game.dataset = str(snapshot)
    cfg.wandb.mode = "disabled"
    cfg.r2.enabled = False
    cfg.output = str(tmp_path / "run")
    if cached:
        cfg.trainer.loader = "cached"
        cfg.trainer.frame_cache = str(build_cache(snapshot, tmp_path / "cache", workers=1))
    output = train(cfg)
    replay = compose_config(recipe=output / "recipe.yaml")
    assert replay.approach.options.detach_feedback is detach
    model, metadata = load_model(output / "best.pt", torch.device("cpu"))
    assert model.detach_feedback is detach
    player = Player(
        model, metadata, torch.device("cpu"), load_scene(output / "start-scene.npz", metadata)
    )
    player.advance(2)
    player.advance(0)
    assert player.steps == 2 and player.pixels().shape == (21, 17, 3)
    player.reset()
    assert player.steps == 0
