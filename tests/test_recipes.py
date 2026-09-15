import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from gymemu.checkpoints import load_model
from gymemu.config import compose_config
from gymemu.engine import train
from gymemu.optimizers import build_optimizer
from gymemu.recipes import recipe_arguments, write_recipe

ROOT = Path(__file__).resolve().parents[1]


def test_successful_breakout_recipe_and_overrides():
    cfg = compose_config(["recipe=breakout_cnn"])
    assert cfg.game.revision == "676ff6388f4218d3c3a3ce9f2f33e075fa7314a3"
    assert cfg.approach.kind == "direct" and cfg.approach.models.predictor.width == 32
    assert cfg.history == 8 and cfg.seed == 47
    assert cfg.trainer.batch_size == 64 and cfg.trainer.precision == "bf16"
    assert cfg.trainer.loader == "cached" and cfg.trainer.workers == 2
    assert cfg.approach.stages[0].epochs == 10 and cfg.optimizer.kind == "adam"
    cfg = compose_config(["recipe=breakout_cnn", "experiment=smoke", "trainer.epochs=2"])
    assert cfg.trainer.device == "cpu" and not cfg.trainer.compile
    assert cfg.trainer.loader == "standard" and cfg.trainer.frame_cache is None
    assert cfg.approach.stages[0].epochs == 2
    cfg = compose_config(["recipe=latent", "game=custom", "experiment=smoke"])
    assert cfg.game.name == "custom" and cfg.approach.kind == "latent"
    assert cfg.approach.models.codec.kind == "frame_codec"


@pytest.mark.parametrize("kind", ["direct", "latent"])
def test_run_recipe_replays_weights_and_tracks_source(snapshot, tmp_path, kind, monkeypatch):
    monkeypatch.setenv("GYMEMU_IMAGE_SOURCE_SHA", "a" * 40)
    monkeypatch.setenv("GYMEMU_IMAGE_REF", "ghcr.io/tsilva/gymemu/train@sha256:" + "b" * 64)
    cfg = compose_config([f"recipe={kind}", "game=custom", "experiment=smoke"])
    cfg.game.dataset = str(snapshot)
    cfg.wandb.mode = "disabled"
    cfg.r2.enabled = False
    cfg.output = str(tmp_path / "first")
    first = train(cfg)
    replay = compose_config([f"output={tmp_path / 'second'}"], recipe=first / "recipe.yaml")
    second = train(replay)
    original, _ = load_model(first / "best.pt", torch.device("cpu"))
    repeated, metadata = load_model(second / "best.pt", torch.device("cpu"))
    for name, value in original.state_dict().items():
        assert torch.equal(value, repeated.state_dict()[name]), name
    receipt = json.loads((first / "reproduction.json").read_text())
    assert receipt["dataset"]["local_content_sha256"]
    assert receipt["container"]["source_commit"] == "a" * 40
    assert receipt["container"]["image_ref"].endswith("b" * 64)
    assert (
        receipt["recipe_sha256"] == hashlib.sha256((first / "recipe.yaml").read_bytes()).hexdigest()
    )
    assert (
        receipt["source_archive_sha256"]
        == hashlib.sha256((first / "source.tar.gz").read_bytes()).hexdigest()
    )
    assert receipt["packages"]["torch"] and receipt["python"] and receipt["device"] == "cpu"
    with tarfile.open(first / "source.tar.gz") as archive:
        assert "uv.lock" in archive.getnames() and "gymemu/engine.py" in archive.getnames()
        assert "containers/train/Dockerfile" in archive.getnames()
        assert "gymemu/web_assets/index.html" in archive.getnames()
        assert "gymemu/web_assets/vendor/gridstack/gridstack-all.js" in archive.getnames()
        for member in archive.getmembers():
            assert (
                hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                == receipt["source_files"][member.name]
            )
    assert metadata["recipe_sha256"] == receipt["recipe_sha256"]
    tuned = compose_config(["trainer.epochs=3", "model.width=6"], recipe=first / "recipe.yaml")
    assert all(stage.epochs == 3 for stage in tuned.approach.stages)
    assert tuned.model.width == 6
    assert OmegaConf.to_container(tuned.approach.models, resolve=True) != OmegaConf.to_container(
        cfg.approach.models, resolve=True
    )
    with pytest.raises(FileExistsError):
        write_recipe(first / "recipe.yaml", OmegaConf.to_container(cfg, resolve=True))


def test_recipe_freezes_environment_values_and_supports_external_cli(
    snapshot, tmp_path, monkeypatch
):
    monkeypatch.setenv("GYMEMU_RECIPE_TEST_SEED", "73")
    cfg = compose_config(["recipe=latent", "game=custom", "experiment=smoke"])
    cfg.seed = "${oc.decode:${oc.env:GYMEMU_RECIPE_TEST_SEED}}"
    cfg.name = "GYMEMU_RECIPE_TEST_SEED"
    cfg.trainer.threads = "${oc.decode:${oc.env:${name}}}"
    cfg.game.dataset = str(snapshot)
    cfg.wandb.mode = "disabled"
    cfg.r2.enabled = False
    cfg.output = str(tmp_path / "old-output")
    filename = tmp_path / "external configs" / "saved.yaml"
    filename.parent.mkdir()
    write_recipe(filename, OmegaConf.to_container(cfg, resolve=True), cfg)
    monkeypatch.setenv("GYMEMU_RECIPE_TEST_SEED", "999")
    cfg2 = compose_config(recipe=filename)
    assert cfg2.seed == 73 and cfg2.output is None
    assert cfg2.trainer.threads == 73
    cfg2.trainer.threads = 2
    # Keep the subprocess smoke bounded to two CPU threads.
    saved = OmegaConf.load(filename)
    saved.trainer.threads = 2
    OmegaConf.save(saved, filename)
    args = [
        sys.executable,
        str(ROOT / "train.py"),
        "--recipe",
        str(filename),
        "--multirun",
        "seed=1,2",
        f'hydra.sweep.dir="{tmp_path / "replays"}"',
    ]
    result = subprocess.run(
        args,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, "HYDRA_FULL_ERROR": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for i in (0, 1):
        assert (tmp_path / "replays" / str(i) / "summary.json").exists()
    with pytest.raises(ValueError, match="cannot be combined"):
        recipe_arguments(["--recipe", str(filename), "--config-name=config"])


def test_hub_revision_pinned_in_run_recipe(snapshot, tmp_path, monkeypatch):
    import gymemu.engine as engine

    monkeypatch.setattr(
        engine,
        "resolve_dataset",
        lambda *_: (snapshot, {"dataset": "example/game", "revision": "a" * 40}),
    )
    cfg = compose_config(["game=custom", "experiment=smoke"])
    cfg.game.dataset, cfg.game.revision = "example/game", "main"
    cfg.wandb.mode = "disabled"
    cfg.r2.enabled = False
    cfg.output = str(tmp_path / "run")
    output = train(cfg)
    assert compose_config(recipe=output / "recipe.yaml").game.revision == "a" * 40
    assert OmegaConf.load(output / "resolved.yaml").game.revision == "a" * 40


def test_optimizer_yaml_preserves_adam_and_selects_adamw():
    a, b = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(3))
    cfg = compose_config()
    selected = build_optimizer([a], OmegaConf.to_container(cfg.optimizer), 0.001)
    reference = torch.optim.Adam([b], lr=0.001)
    for _ in range(3):
        a.grad, b.grad = a.detach().clone(), b.detach().clone()
        selected.step()
        reference.step()
    assert torch.equal(a, b)
    cfg = compose_config(["optimizer=adamw", "optimizer.weight_decay=0.2"])
    selected = build_optimizer([a], OmegaConf.to_container(cfg.optimizer), 0.003)
    assert isinstance(selected, torch.optim.AdamW)
    assert selected.param_groups[0]["weight_decay"] == 0.2
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer([a], {"kind": "os.system"}, 0.001)


def test_replay_rejects_changed_dataset(snapshot, tmp_path):
    import pyarrow.parquet as pq

    cfg = compose_config(["game=custom", "experiment=smoke"])
    cfg.game.dataset = str(snapshot)
    cfg.wandb.mode = "disabled"
    cfg.r2.enabled = False
    cfg.output = str(tmp_path / "first")
    output = train(cfg)
    shard = next((snapshot / "frames/assets").glob("*.parquet"))
    pq.write_table(pq.read_table(shard), shard, compression="NONE")
    cfg = compose_config([f"output={tmp_path / 'changed'}"], recipe=output / "recipe.yaml")
    with pytest.raises(ValueError, match="Dataset differs from saved recipe"):
        train(cfg)
    assert not (tmp_path / "changed/best.pt").exists()
