import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from gymemu.approaches import build_approach
from gymemu.batches import CachedBatchLoader
from gymemu.cache import build_cache
from gymemu.checkpoints import load_model
from gymemu.config import compose_config
from gymemu.data import Episode, Frames, Windows
from gymemu.engine import batch_loss, train
from gymemu.scenes import load_scene
from play import Player


def configured(*overrides):
    return compose_config(["recipe=breakout_scheduled", "experiment=smoke", *overrides])


def model_for(cfg):
    return build_approach(
        OmegaConf.to_container(cfg.approach, resolve=True), cfg.history, 2, (3, 21, 17)
    )


def long_episodes():
    return [
        Episode(
            1, np.array([10, 20, 30, 10, 20, 30, 10, 20, 30]), np.array([2, 0, 2, 2, 0, 0, 2, 0])
        ),
        Episode(3, np.array([70, 10]), np.array([0])),
    ]


@pytest.mark.parametrize("action_history", [1, 2, 4])
def test_extended_prefix_preserves_every_target_and_aligned_action(snapshot, action_history):
    frames = Frames(snapshot, compact=True)
    normal = Windows(frames, long_episodes(), 4, [0, 2], action_history=action_history)
    data = Windows(
        frames, long_episodes(), 4, [0, 2], action_history=action_history, rollout_steps=3
    )
    for index in range(len(data)):
        history, actions, target = data[index]
        assert history.shape == (7, 3, 21, 17)
        assert actions.shape == (4, action_history)
        assert torch.equal(history[-4:], normal[index][0])
        assert torch.equal(target, normal[index][2])
        assert torch.equal(actions[-1], torch.as_tensor(normal[index][1]).reshape(-1))
    # Non-contiguous image IDs, including the anchor's complete earlier RGB history.
    assert torch.equal(
        data[7][0], torch.stack([frames.get(i) for i in [10, 20, 30, 10, 20, 30, 10]])
    )
    for index in (0, 9):
        assert not data[index][0].any()
        assert (data[index][1][:-1] == -1).all()
        assert (data[index][1][-1] == 2).all()  # Valid bootstrap, unlike negative positions.
    assert (data[1][1][0:2] == -1).all()
    assert (data[1][1][2] == 2).all()


def test_cached_prefixes_match_standard_loader_with_shuffling_and_partial_batch(snapshot, tmp_path):
    cache = build_cache(snapshot, tmp_path / "cache", workers=1)
    data = Windows(
        Frames(snapshot, compact=True, cache=cache),
        long_episodes(),
        4,
        [0, 2],
        action_history=4,
        rollout_steps=3,
    )
    order = [10, 9, 1, 7, 0, 4, 8]
    for a, b in zip(
        DataLoader(data, batch_size=3, sampler=order),
        CachedBatchLoader(data, 3, sampler=order),
        strict=True,
    ):
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_schedule_and_reference_architecture_control():
    cfg = compose_config(["recipe=breakout_scheduled"])
    reference = compose_config(["recipe=breakout_actions"])
    for key in ("model", "trainer", "optimizer", "game", "seed", "history"):
        assert (
            OmegaConf.to_container(cfg, resolve=True)[key]
            == OmegaConf.to_container(reference, resolve=True)[key]
        )
    model = model_for(cfg)
    values = [model.begin_epoch(e)["prediction_probability"] for e in range(1, 11)]
    assert values == pytest.approx([0, 0, 0.8 / 6, 1.6 / 6, 0.4, 3.2 / 6, 4 / 6, 0.8, 0.8, 0.8])
    assert "prediction_probability" not in model.state_dict()
    base = model_for(reference)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in base.parameters())


@pytest.mark.parametrize(
    "override",
    [
        "approach.options.rollout_steps=0",
        "approach.options.rollout_steps=3",
        "approach.options.schedule.warmup_epochs=-1",
        "approach.options.schedule.ramp_epochs=0",
        "approach.options.schedule.max_probability=1.1",
    ],
)
def test_invalid_curriculum_rejected(override):
    with pytest.raises(ValueError):
        model_for(configured(override))


class Spy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.05))
        self.calls = []

    def forward(self, history, actions):
        self.calls.append(
            (history.detach().clone(), actions.detach().clone(), torch.is_grad_enabled())
        )
        return history[:, -1] + self.weight


def test_feedback_is_sequential_detached_and_padding_stays_empty(snapshot):
    cfg = configured(
        "approach.options.schedule.warmup_epochs=0",
        "approach.options.schedule.ramp_epochs=1",
        "approach.options.schedule.max_probability=1",
    )
    model = model_for(cfg)
    model.predictor = spy = Spy()
    model.begin_epoch(1)
    data = Windows(Frames(snapshot), long_episodes(), 2, [0, 2], action_history=2, rollout_steps=2)
    history, actions, target = next(iter(DataLoader(data, batch_size=4)))
    loss = model.loss(history, actions, target)
    assert [call[2] for call in spy.calls] == [False, False, True]
    assert not spy.calls[-1][0][0].any()  # Bootstrap target has no real prior context.
    assert not spy.calls[-1][0][1, 0].any()  # One nonexistent frame remains zero.
    assert torch.allclose(spy.calls[-1][0][1, 1], torch.full_like(target[0], 0.05))
    assert torch.allclose(spy.calls[-1][0][2, 1], torch.full_like(target[0], 0.10))
    assert torch.equal(spy.calls[-1][1], actions[:, -1])
    expected_prediction = spy.calls[-1][0][:, -1] + spy.weight.detach()
    assert torch.allclose(loss, (expected_prediction - target).square().mean())
    loss.backward()
    # Only the supervised final forward contributes a derivative, not prefix generation.
    assert torch.allclose(spy.weight.grad, 2 * (expected_prediction - target).mean())
    before = spy.calls[0][0].clone()
    spy.calls.clear()
    model.loss(history, actions, target + 0.4)
    assert torch.equal(before, spy.calls[0][0])  # Changing the label cannot leak into inputs.


def test_coin_flip_is_per_example_per_frame_and_reproducible():
    model = model_for(
        configured(
            "approach.options.schedule.warmup_epochs=0",
            "approach.options.schedule.ramp_epochs=1",
            "approach.options.schedule.max_probability=0.5",
        )
    )
    model.predictor = Spy()
    model.begin_epoch(1)
    history = torch.zeros(128, 4, 3, 21, 17)
    actions = torch.zeros(128, 3, 2, dtype=torch.long)
    torch.manual_seed(13)
    mixed = model.mixed_history(history, actions)
    torch.manual_seed(13)
    assert torch.equal(mixed, model.mixed_history(history, actions))
    replaced = mixed[:, :, 0, 0, 0] > 0
    assert 0.35 < replaced.float().mean() < 0.65
    assert (replaced[:, 0] != replaced[:, 1]).any()
    assert torch.equal(mixed[:, :, :1, :1, :1].expand_as(mixed), mixed)


def test_zero_probability_and_evaluation_match_reference_without_rng_use(snapshot):
    model = model_for(configured())
    base = model_for(compose_config(["recipe=breakout_actions", "experiment=smoke"]))
    base.load_state_dict(model.state_dict())
    data = Windows(Frames(snapshot), long_episodes(), 2, [0, 2], action_history=2, rollout_steps=2)
    h, a, t = next(iter(DataLoader(data, batch_size=4)))
    state = torch.get_rng_state()
    loss = model.loss(h, a, t)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(loss, base.loss(h[:, -2:], a[:, -1], t))
    model.begin_epoch(10)
    model.eval()
    state = torch.get_rng_state()
    actual = model.evaluate(h[:, -2:], a[:, -1], t)
    expected = base.evaluate(h[:, -2:], a[:, -1], t)
    assert all(torch.equal(x, y) for x, y in zip(actual, expected, strict=True))
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("recipe", ["breakout_scheduled", "breakout_scheduled_fast"])
def test_training_replay_checkpoint_and_player(snapshot, tmp_path, cached, recipe):
    cfg = configured(
        f"recipe={recipe}",
        "game=custom",
        "trainer.epochs=3",
        "approach.options.schedule.warmup_epochs=1",
        "approach.options.schedule.ramp_epochs=2",
    )
    if recipe.endswith("_fast"):
        cfg.approach.options.selective_threshold = 1.0
    cfg.game.dataset = str(snapshot)
    cfg.output = str(tmp_path / "run")
    if cached:
        cfg.trainer.loader = "cached"
        cfg.trainer.frame_cache = str(build_cache(snapshot, tmp_path / "cache", workers=1))
    output = train(cfg)
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [r["curriculum"]["prediction_probability"] for r in records] == [0, 0.4, 0.8]
    model, metadata = load_model(output / "best.pt", torch.device("cpu"))
    player = Player(
        model, metadata, torch.device("cpu"), load_scene(output / "start-scene.npz", metadata)
    )
    player.advance(2)
    assert player.steps == 1 and player.pixels().shape == (21, 17, 3)
    replay = compose_config([f"output={tmp_path / 'replay'}"], recipe=output / "recipe.yaml")
    assert replay.approach.options.rollout_steps == cfg.history
    repeated = train(replay)
    original, _ = load_model(output / "last.pt", torch.device("cpu"))
    again, _ = load_model(repeated / "last.pt", torch.device("cpu"))
    assert all(torch.equal(v, again.state_dict()[k]) for k, v in original.state_dict().items())


def test_compiled_loss_accepts_probability_updates_without_changing_contract(snapshot):
    model = model_for(
        configured(
            "approach.options.schedule.warmup_epochs=0", "approach.options.schedule.ramp_epochs=2"
        )
    )
    data = Windows(
        Frames(snapshot, compact=True),
        long_episodes(),
        2,
        [0, 2],
        action_history=2,
        rollout_steps=2,
    )
    history, actions, target = next(iter(DataLoader(data, batch_size=4)))
    compiled = torch.compile(batch_loss, backend="eager", fullgraph=True)
    for epoch in (1, 2):
        model.begin_epoch(epoch)
        loss, _ = compiled(model, history, actions, target, True, True)
        loss.backward()
        assert torch.isfinite(loss) and model.predictor.encoder[0].weight.grad.abs().sum() > 0
