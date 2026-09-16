"""Interrupted training must produce the same updates as uninterrupted training."""

import json

import pytest
import torch

from gymemu.config import compose_config
from gymemu.engine import train
from gymemu.training_state import load_training, resume_config


def config_for(snapshot, output, approach):
    overrides = ["game=custom", f"approach={approach}", "experiment=smoke"]
    if approach == "autoregressive_ball_region":
        overrides += ["+game.ball_sprite.height=4", "+game.ball_sprite.width=2"]
    cfg = compose_config(overrides)
    cfg.game.dataset = str(snapshot)
    cfg.output = str(output)
    cfg.wandb.mode = "disabled"
    cfg.r2.enabled = False
    cfg.trainer.train_batches = None
    cfg.trainer.epochs = 2
    cfg.trainer.diagnostics.enabled = True
    cfg.trainer.diagnostics.log_every = 2
    cfg.trainer.diagnostics.probe_every = 2
    if approach == "scheduled_actions":
        cfg.approach.options.schedule.warmup_epochs = 0
        cfg.approach.options.schedule.ramp_epochs = 1
    return cfg


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize(
    "approach", ["direct", "latent", "scheduled_actions", "autoregressive_ball_region"]
)
@pytest.mark.parametrize(
    "boundary", ["mid_epoch", "before_eval", "after_epoch", "stage_end", "later_stage"]
)
def test_resume_mid_epoch_matches_uninterrupted(
    snapshot, tmp_path, monkeypatch, approach, boundary
):
    import gymemu.engine as engine

    full = train(config_for(snapshot, tmp_path / "full", approach))
    original = engine.save_training
    interrupted = tmp_path / "interrupted"

    def save_and_crash(path, state):
        original(path, state)
        stage = int(boundary == "later_stage" and approach == "latent")
        epoch = 2 if boundary in ("stage_end", "later_stage") else 1
        batches = 3 if boundary == "before_eval" else 1
        matches = (
            state["epoch_complete"]
            if boundary in ("after_epoch", "stage_end")
            else not state["epoch_complete"] and state["progress"]["batches"] == batches
        )
        if state["stage_index"] == stage and state["epoch"] == epoch and matches:
            raise RuntimeError("power cut after atomic save")

    monkeypatch.setattr(engine, "save_training", save_and_crash)
    # Force every update to save without depending on wall-clock timing.
    monkeypatch.setattr(engine, "checkpoint_due", lambda *args: True)
    with pytest.raises(RuntimeError, match="power cut"):
        train(config_for(snapshot, interrupted, approach))
    monkeypatch.setattr(engine, "save_training", original)
    resumed = train(resume_config(interrupted / "resume.pt", [f"output={tmp_path / 'resumed'}"]))
    expected, actual = load_training(full / "resume.pt"), load_training(resumed / "resume.pt")
    for key in ["state_dict", "optimizer", "rng", "total_updates", "total_train_samples", "best"]:
        assert_same(expected[key], actual[key])
    a = torch.load(full / "best.pt", weights_only=True)
    b = torch.load(resumed / "best.pt", weights_only=True)
    assert_same(a["state_dict"], b["state_dict"])
    assert (
        json.loads((full / "summary.json").read_text())["best_mse"]
        == json.loads((resumed / "summary.json").read_text())["best_mse"]
    )

    records = {
        (d["stage"], d["epoch"]): d
        for d in map(json.loads, (full / "metrics.jsonl").read_text().splitlines())
    }
    if (resumed / "metrics.jsonl").exists():
        for line in (resumed / "metrics.jsonl").read_text().splitlines():
            record = json.loads(line)
            expected_record = records[(record["stage"], record["epoch"])]
            for split in ("train", "validation"):
                for key, value in record[split].items():
                    if key != "seconds":
                        assert value == expected_record[split][key]


@pytest.mark.parametrize("loader", ["cached", "standard"])
def test_stop_signal_checkpoint_and_cli_resume(snapshot, tmp_path, monkeypatch, loader):
    import os
    import signal
    import subprocess
    import sys
    from pathlib import Path

    from gymemu.cache import build_cache

    cfg = config_for(snapshot, tmp_path / "signal", "direct")
    cfg.trainer.loader = loader
    cfg.trainer.workers = 2
    if loader == "cached":
        cfg.trainer.frame_cache = str(build_cache(snapshot, tmp_path / "cache", workers=1))
    full_cfg = config_for(snapshot, tmp_path / "full", "direct")
    full_cfg.trainer = cfg.trainer.copy()
    full = train(full_cfg)
    original = torch.optim.Adam.step

    def step_then_stop(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        os.kill(os.getpid(), signal.SIGTERM)
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", step_then_stop)
    output = train(cfg)
    saved = load_training(output / "resume.pt")
    assert saved["progress"]["batches"] == 1
    assert saved["optimizer"]["state"]
    assert not (output / "summary.json").exists()
    resumed = tmp_path / "resumed"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "train.py"),
            "--resume",
            str(output / "resume.pt"),
            f"output={resumed}",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected, actual = load_training(full / "resume.pt"), load_training(resumed / "resume.pt")
    for key in ["state_dict", "optimizer", "rng", "total_updates", "total_train_samples"]:
        assert_same(expected[key], actual[key])


def test_reject_weights_only_and_changed_recipe(snapshot, tmp_path):
    output = train(config_for(snapshot, tmp_path / "run", "direct"))
    with pytest.raises(ValueError, match="Not a resumable"):
        resume_config(output / "latest.pt")
    for override in ["trainer.batch_size=3", "optimizer.betas=[0.5,0.9]", "seed=100"]:
        with pytest.raises(ValueError, match="cannot change"):
            resume_config(output / "resume.pt", [override])


def test_rng_roundtrip_and_failed_write_preserves_checkpoint(tmp_path, monkeypatch):
    import random

    import numpy as np

    from gymemu.training_state import restore_rng, rng_state, save_training

    state = rng_state(torch.device("cpu"))
    expected = random.random(), np.random.random(), torch.rand(3)
    restore_rng(state)
    actual = random.random(), np.random.random(), torch.rand(3)
    assert_same(expected, actual)
    path = tmp_path / "resume.pt"
    save_training(path, {"value": torch.ones(2)})
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        save_training(path, {"value": torch.zeros(2)})
    assert path.read_bytes() == before
