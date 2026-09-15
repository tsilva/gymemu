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


def test_rollout_feedback_gradients_and_mask_normalization():
    model = model_for(configured("approach.options.ball_region_weight=0"))

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
    assert spy.weight.grad.item() == pytest.approx(0.875)  # Detached feedback gives 0.8125.
    assert spy.outputs[1].grad[1].eq(0).all()
    assert model.epoch_metrics()["rgb_mse"] == pytest.approx(0.5625 / 3)
    assert [model.begin_epoch(i)["rollout_steps"] for i in range(1, 7)] == [1, 2, 4, 8, 8, 8]


@pytest.mark.parametrize("cached", [False, True])
def test_train_checkpoint_and_play(snapshot, tmp_path, cached):
    cfg = configured(
        "game=custom",
        "trainer.epochs=2",
        "approach.options.rollout_steps=3",
        "approach.options.rollout_schedule=[3]",
    )
    cfg.game.dataset = str(snapshot)
    cfg.wandb.mode = "disabled"
    cfg.r2.enabled = False
    cfg.output = str(tmp_path / "run")
    if cached:
        cfg.trainer.loader = "cached"
        cfg.trainer.frame_cache = str(build_cache(snapshot, tmp_path / "cache", workers=1))
    output = train(cfg)
    model, metadata = load_model(output / "best.pt", torch.device("cpu"))
    player = Player(
        model, metadata, torch.device("cpu"), load_scene(output / "start-scene.npz", metadata)
    )
    player.advance(2)
    player.advance(0)
    assert player.steps == 2 and player.pixels().shape == (21, 17, 3)
    player.reset()
    assert player.steps == 0
