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
from gymemu.engine import train
from gymemu.models.action_history import ActionHistoryAutoencoder
from gymemu.models.direct import Autoencoder
from gymemu.player import handle_key
from gymemu.scenes import load_scene, write_scene
from play import Player


def episodes():
    return [
        Episode(
            1, np.array([10, 20, 30, 10, 20, 30, 10, 20, 30]), np.array([2, 0, 2, 2, 0, 0, 2, 0])
        ),
        Episode(3, np.array([70, 10]), np.array([0])),
    ]


def test_actions_align_with_history_current_action_and_episode_boundaries(snapshot):
    frames = Frames(snapshot, compact=True)
    data = Windows(frames, episodes(), 4, [0, 2], action_history=4)
    for index, expected in [
        (0, [2, 2, 2, 2]),
        (1, [2, 2, 2, 1]),
        (2, [2, 2, 1, 0]),
        (4, [1, 0, 1, 1]),
        (5, [0, 1, 1, 0]),
        (9, [2, 2, 2, 2]),
    ]:
        history, actions, target = data[index]
        assert actions.tolist() == expected
        if index in (0, 9):
            assert not history.any()
    history, actions, target = data[5]
    assert torch.equal(history, torch.stack([frames.get(i) for i in [20, 30, 10, 20]]))
    assert torch.equal(target, frames.get(30))
    single = Windows(frames, episodes(), 4, [0, 2])
    for i in range(len(data)):
        assert single[i][1] == data[i][1][-1].item()
        assert torch.equal(single[i][0], data[i][0])
        assert torch.equal(single[i][2], data[i][2])


def test_cached_and_standard_batches_have_identical_action_sequences(snapshot, tmp_path):
    cache = build_cache(snapshot, tmp_path / "cache", workers=1)
    data = Windows(
        Frames(snapshot, compact=True, cache=cache), episodes(), 4, [0, 2], action_history=4
    )
    order = [9, 5, 0, 2, 10, 7, 1]
    normal = DataLoader(data, batch_size=3, sampler=order)
    cached = CachedBatchLoader(data, 3, workers=2, sampler=order)
    for left, right in zip(normal, cached, strict=True):
        assert left[1].ndim == right[1].ndim == 2
        assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))


def test_previous_actions_influence_prediction_and_receive_gradients():
    torch.set_num_threads(2)
    model = ActionHistoryAutoencoder(4, 2, (3, 21, 17), width=4)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.01)
        model.encoder[0].weight[:, 12].fill_(0.05)  # Oldest action slot, first category.
    history = torch.ones(1, 4, 3, 21, 17)
    left = model(history, torch.tensor([[0, 1, 0, 1]]))
    right = model(history, torch.tensor([[1, 1, 0, 1]]))
    assert left.shape == (1, 3, 21, 17)
    assert not torch.equal(left, right)  # Same RGB and current action, different prior action.
    left.square().mean().backward()
    assert model.encoder[0].weight.grad[:, 12].abs().sum() > 0


def test_current_action_only_control_matches_reference_with_same_weights():
    baseline = Autoencoder(4, 2, (3, 21, 17), width=4)
    control = ActionHistoryAutoencoder(4, 2, (3, 21, 17), width=4, action_history=1)
    control.load_state_dict(baseline.state_dict())
    history, action = torch.rand(2, 4, 3, 21, 17), torch.tensor([0, 2])
    assert torch.equal(baseline(history, action), control(history, action))


class Spy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, history, actions):
        self.calls.append((history.clone(), actions.clone()))
        return torch.full((1, 3, 21, 17), len(self.calls) / 10)


def metadata():
    return {
        "history": 4,
        "action_history": 4,
        "shape": [3, 21, 17],
        "action_values": [0, 2],
        "dataset": {"dataset": "fixture", "revision": None},
    }


def test_recorded_actions_match_training_then_follow_actual_player_actions(snapshot, tmp_path):
    frames = Frames(snapshot, compact=True)
    config = metadata()
    game = {
        "start_scene": None,
        "train_split": "train",
        "eval_split": "heldout",
        "start": {"split": "train", "episode_id": 1, "frame_position": 6},
    }
    path = tmp_path / "scene.npz"
    write_scene(path, game, config, frames, {"train": episodes()})
    with np.load(path, allow_pickle=False) as archive:
        assert archive["actions"].tolist() == [2, 0, 0]  # Excludes next executed action.
    spy = Spy()
    player = Player(spy, config, torch.device("cpu"), load_scene(path, config))
    expected_history, expected_actions, _ = Windows(
        frames, episodes(), 4, [0, 2], action_history=4
    )[7]
    assert not spy.calls
    player.advance(2)
    assert torch.equal(spy.calls[0][0][0], expected_history.float() / 255)
    assert torch.equal(spy.calls[0][1][0], expected_actions)
    first_prediction = player.frame.clone()
    player.advance(0)
    assert spy.calls[-1][1].tolist() == [[0, 0, 1, 0]]
    assert torch.equal(spy.calls[-1][0][0, -1], first_prediction)
    before = list(player.past_actions)
    handle_key(player, "left", {"left": 2}, repeat=True)
    assert len(spy.calls) == 2 and list(player.past_actions) == before
    player.reset()
    assert player.steps == 0 and len(spy.calls) == 2
    player.advance(2)
    assert torch.equal(spy.calls[-1][0], spy.calls[0][0])
    assert torch.equal(spy.calls[-1][1], spy.calls[0][1])


def test_empty_bootstrap_has_no_executed_action_and_reset_clears_context():
    spy = Spy()
    player = Player(spy, metadata(), torch.device("cpu"))
    player.advance(2)
    assert spy.calls[-1][1].tolist() == [[2, 2, 2, 2]]
    assert not spy.calls[-1][0].any() and not player.past_actions and player.steps == 0
    player.advance(0)
    assert spy.calls[-1][1].tolist() == [[2, 2, 2, 0]] and player.steps == 1
    player.advance(2)
    assert spy.calls[-1][1].tolist() == [[2, 2, 0, 1]]
    player.reset()
    assert not player.past_actions and player.frame is None


def test_scene_rejects_missing_unknown_or_future_actions(tmp_path):
    path = tmp_path / "scene.npz"
    frames = np.zeros((3, 3, 21, 17), dtype=np.uint8)
    np.savez_compressed(path, frames=frames)
    with pytest.raises(ValueError, match="lacks recorded action history"):
        load_scene(path, metadata())
    assert len(load_scene(path, {**metadata(), "action_history": 1})) == 3
    for actions in ([0], [0, 2, 0], [0, 42]):
        np.savez_compressed(path, frames=frames, actions=np.array(actions))
        with pytest.raises(ValueError, match="Scene"):
            load_scene(path, metadata())
    with pytest.raises(ValueError, match="lacks aligned recorded action history"):
        Player(Spy(), metadata(), torch.device("cpu"), list(torch.zeros(3, 3, 21, 17)))


@pytest.mark.parametrize("loader", ["standard", "cached"])
def test_recipe_training_checkpoint_and_playback(snapshot, tmp_path, loader):
    cfg = compose_config(["recipe=breakout_actions", "game=custom", "experiment=smoke"])
    cfg.game.dataset = str(snapshot)
    cfg.wandb.mode = "disabled"
    cfg.r2.enabled = False
    cfg.game.start.frame_position = 1
    cfg.output = str(tmp_path / "run")
    if loader == "cached":
        cfg.trainer.frame_cache = str(build_cache(snapshot, tmp_path / "cache", workers=1))
        cfg.trainer.loader = loader
    output = train(cfg)
    model, config = load_model(output / "best.pt", torch.device("cpu"))
    assert config["action_history"] == cfg.history == 2
    assert config["approach"]["kind"] == "direct_actions"
    player = Player(
        model, config, torch.device("cpu"), load_scene(output / "start-scene.npz", config)
    )
    player.advance(2)
    assert player.steps == 1 and player.pixels().shape == (21, 17, 3)
    replay = compose_config(recipe=output / "recipe.yaml")
    assert replay.model.action_history == 2
    summary = json.loads((output / "summary.json").read_text())
    assert summary["action_history"] == 2
    wrong = tmp_path / "wrong.pt"
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    checkpoint["config"]["action_history"] = 1
    torch.save(checkpoint, wrong)
    with pytest.raises(ValueError, match="action-history contract"):
        load_model(wrong, torch.device("cpu"))


def test_recipe_is_a_controlled_change_to_the_successful_recipe():
    before = OmegaConf.to_container(compose_config(["recipe=breakout_cnn"]), resolve=True)
    after = OmegaConf.to_container(compose_config(["recipe=breakout_actions"]), resolve=True)
    for key in ("game", "trainer", "optimizer", "history", "seed"):
        assert before[key] == after[key]
    assert before["approach"]["stages"] == after["approach"]["stages"]
    assert after["model"]["action_history"] == 8 and after["model"]["width"] == 32
    model = build_approach(after["approach"], 8, 3, (3, 210, 160))
    assert model.predictor.encoder[0].in_channels == 24 + 8 * 4
    assert sum(p.numel() for p in model.parameters()) == 358211
    for length in (0, 9):
        with pytest.raises(ValueError, match="action_history"):
            ActionHistoryAutoencoder(8, 3, (3, 210, 160), action_history=length)
