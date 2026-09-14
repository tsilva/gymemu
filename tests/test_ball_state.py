import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from conftest import write
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from gymemu.approaches import build_approach
from gymemu.batches import CachedBatchLoader
from gymemu.cache import build_cache
from gymemu.checkpoints import load_model
from gymemu.config import compose_config
from gymemu.data import Frames, Windows, read_episodes, recorded_state
from gymemu.engine import batch_loss, train
from gymemu.scenes import load_scene, save_start_state
from play import Player

FIELDS = ("ball_x_normalized", "ball_y_normalized")


def record(x, y):
    labels = ["dict", [[name, ["scalar", value]] for name, value in zip(FIELDS, (x, y))]]
    return json.dumps({"arrays": [], "structure": json.dumps(["dict", [["labels", labels]]])})


@pytest.fixture
def labeled_snapshot(snapshot):
    for split in ("train", "heldout"):
        rows = pq.read_table(snapshot / "transitions" / split / "00000.parquet").to_pylist()
        for row in rows:
            # Shared image ID 10 gets a different state in episode 3. Labels belong to
            # trajectory positions, not to an image-ID lookup or frame-row offset.
            x = row["episode_id"] / 10
            y = (row["step"] + 1) / 10
            row["record_json"] = record(x, y)
        write(snapshot, "transitions", split, rows)
    return snapshot


def dataset(root, cache=None):
    return Windows(
        Frames(root, compact=True, cache=cache),
        read_episodes(root, "train", state_fields=FIELDS),
        2,
        [0, 2],
        action_history=2,
        state_fields=FIELDS,
    )


def test_successor_labels_align_with_images_padding_and_episode_boundaries(labeled_snapshot):
    data = dataset(labeled_snapshot)
    for index, state_history, state_target in [
        (0, [[0, 0, 0], [0, 0, 0]], [0, 0, 0]),
        (1, [[0, 0, 0], [0, 0, 0]], [0.1, 0.1, 1]),
        (2, [[0, 0, 0], [0.1, 0.1, 1]], [0.1, 0.2, 1]),
        (3, [[0, 0, 0], [0, 0, 0]], [0, 0, 0]),
        (4, [[0, 0, 0], [0, 0, 0]], [0.3, 0.1, 1]),
    ]:
        rgb, action, target, history, state = data[index]
        torch.testing.assert_close(history, torch.tensor(state_history, dtype=torch.float32))
        torch.testing.assert_close(state, torch.tensor(state_target, dtype=torch.float32))
    rgb, _, target, history, state = data[2]
    assert torch.equal(rgb, torch.stack([data.frames.get(10), data.frames.get(20)]))
    assert torch.equal(target, data.frames.get(30))
    assert not data[3][0].any()  # No context from episode 1 leaks into episode 3.
    plain = Windows(
        data.frames, read_episodes(labeled_snapshot, "train"), 2, [0, 2], action_history=2
    )
    for i in range(len(data)):
        assert all(torch.equal(a, b) for a, b in zip(data[i][:3], plain[i], strict=True))


@pytest.mark.parametrize("x,y", [(None, 0.1), (True, 0.1), (float("nan"), 0.1), (0.1, 1.1)])
def test_reject_invalid_labels(x, y):
    with pytest.raises(ValueError, match="normalized state labels"):
        recorded_state(record(x, y), FIELDS)


def test_missing_labels_fail_and_baseline_remains_usable(snapshot):
    assert read_episodes(snapshot, "train")
    with pytest.raises(Exception, match="record_json"):
        read_episodes(snapshot, "train", state_fields=FIELDS)
    with pytest.raises(ValueError, match="normalized state labels"):
        recorded_state(json.dumps({"structure": '["dict", []]'}), FIELDS)


def test_cached_batches_preserve_float_coordinates(labeled_snapshot, tmp_path):
    cache = build_cache(labeled_snapshot, tmp_path / "cache", workers=1)
    data = dataset(labeled_snapshot, cache)
    order = [4, 0, 2, 3, 1]
    standard = DataLoader(data, batch_size=2, sampler=order)
    cached = CachedBatchLoader(data, 2, workers=2, sampler=order)
    for left, right in zip(standard, cached, strict=True):
        assert len(left) == 5 and left[3].dtype == torch.float32
        assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))


def approach():
    cfg = compose_config(["recipe=breakout_ball", "experiment=smoke"])
    model = build_approach(OmegaConf.to_container(cfg.approach, resolve=True), 2, 2, (3, 21, 17))
    model.prepare_stage("next_frame_and_ball")
    return model


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires Apple MPS")
@pytest.mark.parametrize("height,width", [(27, 20), (20, 27), (3, 3), (28, 20)])
def test_coordinate_pool_mps_matches_cpu_values_and_gradients(height, width):
    torch.manual_seed(19)
    pool = approach().predictor.coordinates[0]
    source = torch.randn(2, 16, height, width, requires_grad=True)
    apple = source.detach().to("mps").requires_grad_()
    expected = torch.nn.functional.adaptive_avg_pool2d(source, (4, 4))
    actual = pool(apple)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=1e-5)
    weights = torch.randn_like(expected)
    (expected * weights).sum().backward()
    (actual * weights.to("mps")).sum().backward()
    torch.testing.assert_close(apple.grad.cpu(), source.grad, atol=1e-6, rtol=1e-5)


def test_joint_loss_conditions_rgb_and_shares_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(13)
    model = approach()
    history = torch.rand(2, 2, 3, 21, 17)
    actions = torch.tensor([[0, 1], [1, 0]])
    states = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.2, 1.0]]] * 2, requires_grad=True)
    rgb, xy = model(history, actions, states)
    changed = states.detach().clone()
    changed[:, -1, 0] = 0.9
    other_rgb, other_xy = model(history, actions, changed)
    assert rgb.shape == (2, 3, 21, 17) and xy.shape == (2, 2)
    assert ((xy >= 0) & (xy <= 1)).all()
    assert not torch.equal(rgb, other_rgb) and not torch.equal(xy, other_xy)
    xy.square().mean().backward(retain_graph=True)
    assert model.predictor.encoder[0].weight.grad.abs().sum() > 0
    assert states.grad[:, -1, :2].abs().sum() > 0
    model.zero_grad()
    rgb.square().mean().backward()
    assert model.predictor.coordinates[-2].weight.grad.abs().sum() > 0
    targets = torch.tensor([[0.4, 0.6, 1.0], [0.0, 0.0, 0.0]])
    expected = 0.01 * ((xy[0].float() - targets[0, :2]).square().mean() / 2)
    torch.testing.assert_close(model.joint_loss(rgb, xy, rgb, targets), expected)
    absent = torch.zeros_like(targets)
    assert model.joint_loss(rgb, xy, rgb, absent) == 0


@pytest.mark.parametrize("loader", ["standard", "cached"])
def test_train_reload_and_autoregressive_playback(labeled_snapshot, tmp_path, loader):
    cfg = compose_config(["recipe=breakout_ball", "game=custom", "experiment=smoke"])
    cfg.wandb.mode = "disabled"
    cfg.r2.enabled = False
    cfg.r2.enabled = False
    cfg.game.dataset = str(labeled_snapshot)
    cfg.game.start.frame_position = 1
    cfg.game.start.split = "heldout"
    cfg.output = str(tmp_path / "run")
    if loader == "cached":
        cfg.trainer.frame_cache = str(build_cache(labeled_snapshot, tmp_path / "cache", workers=1))
        cfg.trainer.loader = loader
    output = train(cfg)
    model, config = load_model(output / "best.pt", torch.device("cpu"))
    assert config["state_fields"] == list(FIELDS)
    replay = compose_config(recipe=output / "recipe.yaml")
    assert replay.approach.options.coordinate_loss_weight == 0.01
    scene = load_scene(output / "start-scene.npz", config)
    torch.testing.assert_close(scene.states, torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.1, 1.0]]))
    # A trained model sees exactly the same inputs in held-out windows and playback.
    data = Windows(
        Frames(labeled_snapshot, compact=True),
        read_episodes(labeled_snapshot, "heldout", state_fields=FIELDS),
        2,
        [0, 2],
        action_history=2,
        state_fields=FIELDS,
    )
    history, actions, target, states, state_target = next(iter(DataLoader(data, batch_size=3)))
    loss, mse = batch_loss(model, history, actions, target, True, False, states, state_target)
    expected_rgb, expected_state = model.predict_step(
        history[2:3].float() / 255, actions[2:3], states[2:3]
    )
    assert torch.isfinite(loss) and mse >= 0 and loss >= mse
    player = Player(model, config, torch.device("cpu"), scene)
    player.advance(2)
    torch.testing.assert_close(player.frame, expected_rgb[0])
    torch.testing.assert_close(player.state_history[-1], expected_state[0])
    previous = player.state_history[-1].clone()
    player.advance(0)
    torch.testing.assert_close(player.state_history[-2], previous)
    assert player.steps == 2 and player.pixels().shape == (21, 17, 3)
    player.reset()
    torch.testing.assert_close(torch.stack(list(player.state_history)), scene.states)
    player.advance(2)
    torch.testing.assert_close(player.frame, expected_rgb[0])
    empty = Player(model, config, torch.device("cpu"))
    empty.advance(2)
    assert empty.steps == 0 and len(empty.state_history) == 1 and not empty.past_actions
    empty.advance(0)
    assert empty.steps == 1
    empty.reset()
    assert not empty.state_history and empty.frame is None
    named = save_start_state(
        output / "start-scene.npz", "ball", "Test scene", config, tmp_path / "states"
    )
    torch.testing.assert_close(load_scene(named, config).states, scene.states)
    with pytest.raises(ValueError, match="State history"):
        Player(model, config, torch.device("cpu"), list(scene))
    missing = tmp_path / "missing.npz"
    np.savez_compressed(missing, frames=np.zeros((2, 3, 21, 17), dtype=np.uint8), actions=[0])
    with pytest.raises(ValueError, match="state_fields"):
        load_scene(missing, config)
    saved = torch.load(output / "best.pt", weights_only=True)
    saved["config"]["state_fields"] = []
    torch.save(saved, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="state-fields contract"):
        load_model(tmp_path / "bad.pt", torch.device("cpu"))


def test_recipe_keeps_action_control_data_and_budget():
    before = OmegaConf.to_container(compose_config(["recipe=breakout_actions"]), resolve=True)
    after = OmegaConf.to_container(compose_config(["recipe=breakout_ball"]), resolve=True)
    for key in ("game", "trainer", "optimizer", "history", "seed"):
        assert before[key] == after[key]
    assert after["model"]["kind"] == "ball_state_cnn"
    assert after["approach"]["stages"][0]["objective"] == "next_frame_and_ball"
