"""Research identity at the resolved launch boundary."""

import json

from gymemu.config import compose_config
from gymemu.engine import train
from gymemu.research import resolve_goal


def test_recipe_override_does_not_change_goal_identity():
    base = {
        "id": "next-frame",
        "environment": "Breakout-Atari2600-v0",
        "dataset": {"id": "example/trajectories", "revision": "a" * 40},
        "evaluation": {"split": "heldout", "targets": [1, 2], "metric": "rgb-mse-fp32"},
        "probes": {"horizon": 2, "starts": [{"episode": 5, "offset": 1, "frame_ids": [2, 3]}]},
    }
    original = resolve_goal(base)
    recipe_change = resolve_goal(base)
    assert original["revision"] == recipe_change["revision"]
    changed = resolve_goal(base, {"evaluation.metric": "other"})
    assert changed["revision"] == original["revision"]
    assert changed["variant"] != original["variant"]
    assert changed["diff"] == [
        {"path": "evaluation.metric", "before": "rgb-mse-fp32", "after": "other"}
    ]
    assert changed["comparability"] != original["comparability"]
    assert resolve_goal(base, {"evaluation": {"metric": "other"}})["variant"] == changed["variant"]


def test_goal_revision_and_probe_identity_follow_contract():
    goal = {
        "id": "next-frame",
        "environment": "Breakout-Atari2600-v0",
        "dataset": {"id": "example/trajectories", "revision": "a" * 40},
        "evaluation": {"split": "heldout", "targets": [1, 2], "metric": "rgb-mse-fp32"},
        "probes": {"horizon": 2, "starts": [{"episode": 5, "offset": 1, "frame_ids": [2, 3]}]},
    }
    initial = resolve_goal(goal)
    goal["probes"]["starts"][0]["frame_ids"] = [2, 4]
    assert resolve_goal(goal)["revision"] != initial["revision"]
    goal["probes"]["starts"][0]["frame_ids"] = [2, 3]
    goal["evaluation"]["targets"] = [1, 3]
    assert resolve_goal(goal)["comparability"] != initial["comparability"]


def test_offline_training_records_stable_local_run_identity(snapshot, tmp_path):
    cfg = compose_config(["game=custom", "experiment=smoke"])
    cfg.game.dataset = str(snapshot)
    cfg.wandb.mode = "disabled"
    cfg.r2.enabled = False
    cfg.output = str(tmp_path / "run")
    output = train(cfg)
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["id"] and manifest["status"] == "complete"
    assert manifest["environment"] == cfg.game.env_id
    assert manifest["recipe"]["sha256"]
    assert manifest["stages"] == [stage.name for stage in cfg.approach.stages]
    assert manifest["publication"] == "offline"


def test_preassigned_run_id_survives_training(snapshot, tmp_path):
    cfg = compose_config(["game=custom", "experiment=smoke"])
    cfg.game.dataset = str(snapshot)
    cfg.wandb.mode = "disabled"
    cfg.r2.enabled = False
    cfg.run_id = "a" * 32
    cfg.output = str(tmp_path / "run")
    output = train(cfg)
    assert json.loads((output / "run.json").read_text())["id"] == "a" * 32
