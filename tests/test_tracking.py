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


@pytest.mark.parametrize("approach", ["direct", "latent", "scheduled_actions"])
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
        prefix = f"stages/{record['stage']}"
        assert logged["optimizer_steps"] == index * 2
        assert logged["train_samples_seen"] == index * 4
        assert logged[f"{prefix}/epoch"] == record["epoch"]
        assert logged[f"{prefix}/learning_rate"] == cfg.trainer.learning_rate
        for split in ("train", "validation"):
            for key, value in record[split].items():
                assert logged[f"{prefix}/{split}/{key}"] == value
            assert logged[f"{prefix}/{split}/samples_per_second"] > 0
        for key, value in record["curriculum"].items():
            assert logged[f"{prefix}/curriculum/{key}"] == value
        if record["stage"] == "representation":
            assert "evaluation/next_frame_rgb_mse" not in logged
        else:
            assert logged["evaluation/next_frame_rgb_mse"] == record["validation"]["mse"]
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
