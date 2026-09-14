import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from conftest import write
from omegaconf import OmegaConf
from PIL import Image

from gymemu.approaches import build_approach
from gymemu.ball_regions import ball_region_mask
from gymemu.checkpoints import load_model
from gymemu.config import compose_config
from gymemu.engine import train
from gymemu.scenes import load_scene
from play import Player

SCENES = Path(__file__).resolve().parents[1] / "start_states/breakout"


def mask(target, padding=4):
    return ball_region_mask(target, sprite_height=4, sprite_width=2, padding=padding)


def approach(weight=0.3):
    cfg = compose_config(["recipe=breakout_ball_region", "experiment=smoke"])
    spec = OmegaConf.to_container(cfg.approach, resolve=True)
    spec["options"]["ball_region_weight"] = weight
    return build_approach(spec, 2, 2, (3, 21, 17))


@pytest.mark.parametrize("name,x,y", [
    ("ball-up", 53, 148), ("near-bricks", 27, 109), ("paddle-approach", 85, 179),
    ("half-cleared", 28, 140), ("almost-cleared", 124, 117), ("above-bricks", 25, 35),
])
def test_saved_frames_detect_exact_documented_ball_regions(name, x, y):
    with np.load(SCENES / f"{name}.npz", allow_pickle=False) as data:
        frames = torch.from_numpy(data["frames"].copy()).float() / 255
    regions, available = mask(frames)
    expected = torch.zeros_like(regions[-1])
    expected[:, y - 4 : y + 8, x - 4 : x + 6] = True
    assert available[-1] and torch.equal(regions[-1], expected)


def test_absence_ambiguity_touching_components_and_edge_clipping():
    target = torch.zeros(5, 3, 21, 17)
    target[1, :, 0:4, 0:2] = 0.5  # Gray ball at image edge, padding clipped.
    target[2, :, 4:8, 4:6] = 0.7
    target[2, :, 12:16, 12:14] = 0.2  # Two candidates, neither selected.
    target[3, :, 4:8, 4:6] = 0.7
    target[3, :, 4, 6] = 0.7  # Same-color contact joins the component.
    target[4, :, 4:8, 4:6] = 0.7
    target[4, :, 3, 3] = 0.7  # Diagonal contact does not join four-connected sprites.
    regions, valid = mask(target)
    assert valid.tolist() == [False, True, False, False, True]
    assert regions[1].sum() == 48 and regions[1, :, :8, :6].all()
    assert not regions[[0, 2, 3]].any()
    tiny, present = mask(torch.zeros(2, 3, 2, 1))
    assert tiny.shape == (2, 1, 2, 1) and not tiny.any() and not present.any()


def test_loss_and_gradients_weight_only_target_region_and_ignore_missing_detections():
    model = approach()
    target = torch.zeros(2, 3, 21, 17)
    target[0, :, 8:12, 7:9] = 0.5
    prediction = (target + 0.1).requires_grad_()
    loss = model.joint_loss(prediction, target)
    torch.testing.assert_close(loss, torch.tensor(0.01 * (1 + 0.3 / 2)))
    loss.backward()
    # Region averaging gives each masked RGB value 1 + lambda * H*W/area weight.
    ratio = prediction.grad[0, 0, 8, 7] / prediction.grad[0, 0, 0, 0]
    torch.testing.assert_close(ratio, torch.tensor(1 + 0.3 * 21 * 17 / 120))
    torch.testing.assert_close(prediction.grad[1, 0, 8, 7], prediction.grad[0, 0, 0, 0])
    metrics = model.epoch_metrics()
    assert metrics["ball_detection_coverage"] == 0.5
    assert metrics["ball_region_mse"] == pytest.approx(0.005)
    model.reset_epoch_metrics()
    losses = [model.joint_loss(prediction[i:i+1], target[i:i+1]) for i in range(2)]
    torch.testing.assert_close(torch.stack(losses).mean(), loss)
    assert model.epoch_metrics() == pytest.approx(metrics)
    assert all("_region_totals" not in key for key in model.state_dict())
    control = approach(0)
    torch.testing.assert_close(control.joint_loss(prediction, target), torch.tensor(0.01))


def test_recipe_is_a_loss_only_ablation_and_predicts_once_from_recorded_history():
    base = OmegaConf.to_container(compose_config(["recipe=breakout_actions"]), resolve=True)
    region = OmegaConf.to_container(compose_config(["recipe=breakout_ball_region"]), resolve=True)
    for key in ("game", "model", "trainer", "optimizer", "history", "seed"):
        assert base[key] == region[key]
    model = approach()
    assert model.training_rollout_steps == 0 and model.state_fields == ()
    history = torch.rand(2, 2, 3, 21, 17)
    actions = torch.tensor([[0, 1], [1, 0]])
    calls = []
    hook = model.predictor.register_forward_pre_hook(lambda _, args: calls.append(args))
    model.loss(history, actions, torch.zeros(2, 3, 21, 17)).backward()
    hook.remove()
    assert len(calls) == 1 and calls[0][0] is history and calls[0][1] is actions


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf"), True])
def test_reject_invalid_loss_weight(weight):
    with pytest.raises(ValueError, match="ball_region_weight"):
        approach(weight)


def test_compiled_loss_preserves_values_gradients_and_metrics():
    torch.set_num_threads(2)
    target = torch.zeros(2, 3, 21, 17)
    target[0, :, 8:12, 7:9] = 0.5
    plain, compiled = approach(), approach()
    x = torch.rand_like(target, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected = plain.joint_loss(x, target)
    actual = torch.compile(compiled.joint_loss, backend="aot_eager", fullgraph=True)(y, target)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(y.grad, x.grad)
    assert compiled.epoch_metrics() == pytest.approx(plain.epoch_metrics())


def test_train_reload_metrics_and_generated_playback(snapshot, tmp_path):
    rows = pq.read_table(snapshot / "frames/assets/00000.parquet").to_pylist()
    for index, row in enumerate(rows):
        frame = np.zeros((21, 17, 3), dtype=np.uint8)
        frame[5:9, index + 3:index + 5] = [200, 72, 72]
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="PNG")
        row["image"] = {"bytes": buffer.getvalue(), "path": None}
    write(snapshot, "frames", "assets", rows)
    cfg = compose_config(["recipe=breakout_ball_region", "experiment=smoke"])
    cfg.game.dataset = str(snapshot)
    cfg.game.revision = None
    cfg.game.start.episode_id = 2
    cfg.game.start.frame_position = 1
    cfg.wandb.mode = "disabled"
    cfg.r2.enabled = False
    cfg.output = str(tmp_path / "run")
    output = train(cfg)
    model, config = load_model(output / "best.pt", torch.device("cpu"))
    assert config["approach"]["options"]["ball_region_weight"] == 0.3
    saved = compose_config(recipe=output / "recipe.yaml")
    assert saved.approach.options.sprite_height == 4
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    for phase in ("train", "validation"):
        metrics = records[0][phase]
        assert metrics["ball_detection_coverage"] == 1
        assert metrics["ball_region_mse"] > 0
        assert metrics["loss"] == pytest.approx(
            metrics["rgb_mse"] + 0.3 * metrics["ball_region_mse"]
        )
    assert records[0]["validation"]["mse"] == pytest.approx(
        records[0]["validation"]["rgb_mse"]
    )
    scene = load_scene(output / "start-scene.npz", config)
    player = Player(model, config, torch.device("cpu"), scene)
    player.advance(2)
    first = player.frame.clone()
    player.advance(0)
    torch.testing.assert_close(player.history[-2], first)
    assert player.steps == 2 and player.pixels().shape == (21, 17, 3)
    player.reset()
    player.advance(2)
    torch.testing.assert_close(player.frame, first)
