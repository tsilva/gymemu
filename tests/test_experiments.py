import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from compare import collect_runs
from gymemu.approaches import build_approach
from gymemu.checkpoints import load_model
from gymemu.config import compose_config
from gymemu.engine import train
from play import Player, default_bindings, load_scene

REPO = Path(__file__).resolve().parents[1]


def config_for(snapshot, output, approach="direct"):
    cfg = compose_config(["game=custom", f"approach={approach}", "experiment=smoke"])
    cfg.game.dataset = str(snapshot)
    cfg.output = str(output)
    return cfg


def test_hierarchical_configs_and_independent_stage_overrides():
    cfg = compose_config(
        [
            "approach=latent",
            "experiment=smoke",
            "model.latent_channels=7",
            "approach.stages.0.epochs=2",
            "approach.models.dynamics.width=5",
        ]
    )
    result = OmegaConf.to_container(cfg, resolve=True)
    assert result["approach"]["models"]["codec"] == {
        "kind": "frame_codec",
        "width": 4,
        "latent_channels": 7,
    }
    assert [s["epochs"] for s in result["approach"]["stages"]] == [2, 1]
    assert result["approach"]["models"]["dynamics"]["width"] == 5
    with pytest.raises(Exception):
        compose_config(["trainer.batch_sze=4"])


@pytest.mark.parametrize("kind", ["direct", "latent"])
def test_approach_train_checkpoint_and_generic_game_playback(snapshot, tmp_path, kind):
    output = tmp_path / kind
    cfg = config_for(snapshot, output, kind)
    cfg.trainer.eval_batches = None
    train(cfg)
    model, metadata = load_model(output / "best.pt", torch.device("cpu"))
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["evaluation"]["samples"] == 3
    assert metadata["format_version"] == 2 and metadata["approach"]["kind"] == kind
    assert metadata["action_values"] == [0, 2]
    assert default_bindings(metadata) == ["1=0", "2=2"]
    player = Player(
        model, metadata, torch.device("cpu"), load_scene(output / "start-scene.npz", metadata)
    )
    player.advance(2)
    assert player.steps == 1 and player.pixels().shape == (21, 17, 3)
    assert (output / "resolved.yaml").is_file()
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    expected = ["prediction"] if kind == "direct" else ["representation", "dynamics"]
    assert [r["stage"] for r in records] == expected
    assert summary["optimizer_steps"] == 2 * len(expected)
    for stage in expected:
        assert (output / "stages" / stage / "best.pt").is_file()
    if kind == "latent":
        representation, _ = load_model(
            output / "stages/representation/best.pt", torch.device("cpu")
        )
        for key, value in representation.codec.state_dict().items():
            assert torch.equal(value, model.codec.state_dict()[key]), key
    before = (output / "best.pt").read_bytes()
    with pytest.raises(ValueError, match="Output directory"):
        train(cfg)
    assert (output / "best.pt").read_bytes() == before


def test_latent_stage_freezing_and_loss_gradients():
    torch.set_num_threads(2)
    cfg = compose_config(["approach=latent", "experiment=smoke", "model.latent_channels=3"])
    spec = OmegaConf.to_container(cfg.approach, resolve=True)
    model = build_approach(spec, 2, 2, [3, 21, 17])
    history, action, target = (
        torch.rand(2, 2, 3, 21, 17),
        torch.tensor([0, 2]),
        torch.rand(2, 3, 21, 17),
    )
    for objective, trained, frozen in [
        ("reconstruction", model.codec, model.dynamics),
        ("latent_prediction", model.dynamics, model.codec),
    ]:
        optimizer = torch.optim.Adam(model.prepare_stage(objective), lr=0.01)
        before_frozen = {k: v.clone() for k, v in frozen.state_dict().items()}
        before_trained = {k: v.clone() for k, v in trained.state_dict().items()}
        optimizer.zero_grad()
        model.loss(history, action, target).backward()
        optimizer.step()
        assert any(not torch.equal(v, trained.state_dict()[k]) for k, v in before_trained.items())
        assert all(torch.equal(v, frozen.state_dict()[k]) for k, v in before_frozen.items())
        assert not frozen.training
    assert model(history, action).shape == target.shape


def test_comparison_separates_evaluation_contracts(snapshot, tmp_path):
    for name, approach, batches in [
        ("direct", "direct", None),
        ("latent", "latent", None),
        ("partial", "direct", 1),
    ]:
        cfg = config_for(snapshot, tmp_path / name, approach)
        cfg.trainer.eval_batches = batches
        train(cfg)
    groups = collect_runs([tmp_path])
    assert sorted(len(rows) for rows in groups.values()) == [1, 2]
    pair = next(rows for rows in groups.values() if len(rows) == 2)
    assert {row["approach"] for row in pair} == {"direct", "latent"}
    assert pair[0]["best_mse"] <= pair[1]["best_mse"]


def test_hydra_multirun_from_another_directory(snapshot, tmp_path):
    destination = tmp_path / "sweep with spaces"
    args = [
        sys.executable,
        str(REPO / "train.py"),
        "--multirun",
        "game=custom",
        f'game.dataset="{snapshot}"',
        "experiment=smoke",
        "approach=direct,latent",
        f'hydra.sweep.dir="{destination}"',
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
    for index in [0, 1]:
        assert (destination / str(index) / "summary.json").is_file()
        assert (destination / str(index) / ".hydra/config.yaml").is_file()
    assert len(collect_runs([destination])) == 1


def test_unknown_checkpoint_approach_is_rejected(tmp_path):
    checkpoint = tmp_path / "bad.pt"
    torch.save(
        {
            "state_dict": {},
            "config": {
                "format_version": 2,
                "history": 2,
                "shape": [3, 21, 17],
                "action_values": [0],
                "approach": {"kind": "os.system", "models": {}},
            },
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="Unknown approach"):
        load_model(checkpoint, torch.device("cpu"))


def test_all_training_stages_exclude_heldout_targets(snapshot, tmp_path, monkeypatch):
    import gymemu.engine as engine

    seen = set()
    original = engine.build_approach

    def build(*args):
        model = original(*args)
        loss = model.loss

        def checked_loss(history, action, target):
            values = set(target[:, 0, 0, 0].mul(255).round().int().tolist())
            if torch.is_grad_enabled():
                assert values <= {10, 20, 30, 70}
                seen.add(model.objective)
            else:
                assert values <= {40, 50, 60}
            return loss(history, action, target)

        model.loss = checked_loss
        return model

    monkeypatch.setattr(engine, "build_approach", build)
    train(config_for(snapshot, tmp_path / "run", "latent"))
    assert seen == {"reconstruction", "latent_prediction"}


def test_overlap_outside_smoke_limit_is_rejected(snapshot, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    for kind in ["episodes", "transitions"]:
        path = snapshot / kind / "heldout/00000.parquet"
        rows = pq.read_table(path).to_pylist()
        # Overlap with the second training episode, outside the one-episode smoke limit.
        for row in rows:
            row["episode_id"] = 3
        pq.write_table(pa.Table.from_pylist(rows), path)
    cfg = config_for(snapshot, tmp_path / "run")
    cfg.trainer.limit_episodes = 1
    with pytest.raises(ValueError, match="episode IDs overlap"):
        train(cfg)


def test_new_direct_approach_preserves_baseline_pixels():
    from gymemu.models.direct import Autoencoder

    cfg = compose_config(["experiment=smoke"])
    model = build_approach(OmegaConf.to_container(cfg.approach, resolve=True), 2, 2, [3, 21, 17])
    baseline = Autoencoder(2, 2, (3, 21, 17), width=4)
    baseline.load_state_dict(model.predictor.state_dict())
    history = torch.rand(2, 2, 3, 21, 17)
    action = torch.tensor([0, 2])
    assert torch.equal(model(history, action), baseline(history, action))
