"""Tracking preserves local metrics and isolates stage losses and run lifetimes."""

import json
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from gymemu.config import compose_config
from gymemu.engine import train
from gymemu.tracking import project_name, validate_tracking


def tracking_config(snapshot, output, approach="direct"):
    cfg = compose_config(["game=custom", f"approach={approach}", "experiment=smoke"])
    cfg.game.dataset = str(snapshot)
    cfg.game.env_id = "Breakout-Atari2600-v0"
    cfg.output = str(output)
    cfg.wandb.mode = "offline"
    cfg.r2.enabled = False
    cfg.trainer.diagnostics.enabled = False
    return cfg


class FakeRun:
    def __init__(self):
        self.config = {}
        self.summary = {}
        self.records = []
        self.definitions = []
        self.exits = []

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))

    def log(self, metrics):
        self.records.append(metrics)

    def finish(self, *, exit_code):
        self.exits.append(exit_code)


@pytest.fixture
def fake_wandb(monkeypatch):
    calls = []

    def init(**kwargs):
        run = FakeRun()
        calls.append((kwargs, run))
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    return calls


def test_project_uses_canonical_environment_not_config_alias():
    for game in ("breakout", "breakout_large"):
        cfg = compose_config([f"game={game}"])
        assert cfg.wandb.mode == "online"
        config = OmegaConf.to_container(cfg, resolve=True)
        assert project_name(config) == "gymemu-Breakout-Atari2600-v0"
    assert project_name({"game": {"env_id": "ALE/Pong-v5"}}) == "gymemu-ALE-Pong-v5"
    assert project_name({"game": {"env_id": "CartPole-v1"}}) == "gymemu-CartPole-v1"


@pytest.mark.parametrize("approach", ["direct", "latent", "scheduled_actions", "reconstruction"])
def test_epoch_metrics_match_local_records(snapshot, tmp_path, fake_wandb, approach):
    cfg = tracking_config(snapshot, tmp_path / "run", approach)
    cfg.trainer.epochs = 2
    output = train(cfg)
    assert len(fake_wandb) == 1
    options, run = fake_wandb[0]
    assert options["project"] == "gymemu-Breakout-Atari2600-v0"
    assert options["dir"] == str(output)
    assert options["mode"] == "offline"
    assert options["config"]["expected_dataset"]["local_content_sha256"]
    local = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert len(local) == len(run.records)
    for index, (record, logged) in enumerate(zip(local, run.records, strict=True), 1):
        stage = record["stage"]
        assert logged["train/step"] == logged["eval/step"] == index * 2
        assert logged["train/samples"] == index * 4
        assert logged["train/epoch"] == record["epoch"]
        assert logged["train/lr"] == cfg.trainer.learning_rate
        assert logged[f"train/{stage}/loss/epoch"] == record["train"]["loss"]
        assert logged[f"eval/{stage}/loss"] == record["validation"]["loss"]
        for split, namespace in (("train", "train"), ("validation", "eval")):
            for key, value in record[split].items():
                if key not in ("loss", "mse"):
                    suffix = "/epoch" if namespace == "train" else ""
                    assert logged[f"{namespace}/{stage}/{key}{suffix}"] == value
            assert logged[f"{namespace}/rate"] > 0
        for key, value in record["curriculum"].items():
            assert logged[f"train/{stage}/curriculum/{key}"] == value
        final_stage = stage == cfg.approach.stages[-1].name
        score_keys = [key for key in logged if key.endswith("/mse")]
        assert score_keys == (["eval/mse"] if final_stage else [])
        if final_stage:
            assert logged["eval/mse"] == record["validation"]["mse"]
        assert not any(key.startswith(("stages/", "evaluation/")) for key in logged)
        assert not {"optimizer_steps", "train_samples_seen", "stage", "objective"} & logged.keys()
    definitions = {args[0]: kwargs for args, kwargs in run.definitions}
    assert definitions["eval/*"]["step_metric"] == "eval/step"
    assert definitions["train/*"]["step_metric"] == "train/step"
    assert not {"optimizer_steps", "stages/*"} & definitions.keys()
    assert run.config["metrics_schema_version"] == 3
    summary = json.loads((output / "summary.json").read_text())
    assert run.summary == summary
    assert run.config["evaluation"] == summary["evaluation"]
    assert run.exits == [0]


def test_disabled_tracking_needs_neither_sdk_nor_environment_id(snapshot, tmp_path, monkeypatch):
    cfg = tracking_config(snapshot, tmp_path / "disabled")
    cfg.wandb.mode = "disabled"
    cfg.r2.enabled = False
    cfg.game.env_id = None
    monkeypatch.setitem(sys.modules, "wandb", None)
    output = train(cfg)
    assert (output / "summary.json").is_file()
    assert not (output / "wandb").exists()
    # Standalone recipes saved before tracking was introduced remain usable.
    del cfg.wandb
    cfg.output = str(tmp_path / "old-recipe")
    train(cfg)


def test_tracking_validation_precedes_dataset_access(snapshot, tmp_path, fake_wandb):
    cfg = tracking_config(snapshot, tmp_path / "invalid")
    cfg.game.env_id = None
    with pytest.raises(ValueError, match="canonical environment ID"):
        train(cfg)
    assert not fake_wandb
    assert not (tmp_path / "invalid").exists()
    with pytest.raises(ValueError, match="wandb.mode"):
        validate_tracking({"wandb": {"mode": "typo"}})


@pytest.mark.parametrize("error", [RuntimeError("training failed"), KeyboardInterrupt()])
def test_failed_or_interrupted_training_finishes_run(
    snapshot, tmp_path, monkeypatch, fake_wandb, error
):
    import gymemu.engine as engine

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(engine, "run_epoch", fail)
    with pytest.raises(type(error)):
        train(tracking_config(snapshot, tmp_path / "failed"))
    assert fake_wandb[0][1].exits == [1]
    assert not (tmp_path / "failed/summary.json").exists()


def test_real_offline_sdk_handles_sequential_direct_and_latent_runs(
    snapshot, tmp_path, monkeypatch
):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    for approach in ("direct", "latent"):
        output = train(tracking_config(snapshot, tmp_path / approach, approach))
        assert len(list((output / "wandb").glob("offline-run-*/run-*.wandb"))) == 1
        assert (output / "summary.json").is_file()


def test_real_offline_diagnostics_have_explicit_axes_and_media(snapshot, tmp_path):
    cfg = tracking_config(snapshot, tmp_path / "diagnostics", "latent")
    cfg.trainer.diagnostics.enabled = True
    cfg.trainer.diagnostics.horizon = 4
    output = train(cfg)
    records = [json.loads(line) for line in (output / "diagnostics.jsonl").read_text().splitlines()]
    assert {record["train/stage"] for record in records} == {"representation", "dynamics"}
    assert any("probe/mse/h1" in record for record in records)
    assert list((output / "wandb").glob("offline-run-*/files/media/images/probe/*"))


def test_graceful_stop_is_recorded_as_interrupted(snapshot, tmp_path, monkeypatch, fake_wandb):
    import os
    import signal

    import torch

    original = torch.optim.Adam.step

    def stop_after_update(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        os.kill(os.getpid(), signal.SIGTERM)
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", stop_after_update)
    output = train(tracking_config(snapshot, tmp_path / "stopped"))
    assert (output / "resume.pt").exists()
    assert fake_wandb[0][1].summary["status"] == "interrupted"
    assert fake_wandb[0][1].exits == [0]
