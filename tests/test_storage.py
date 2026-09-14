"""R2 manifests must only reference verified, replayable run artifacts."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from gymemu.checkpoints import load_model
from gymemu.config import compose_config
from gymemu.engine import train
from gymemu.storage import CheckpointStore, r2_client, validate_storage
from play import Player, load_scene
from upload_checkpoints import main as upload_main


class FakeS3:
    def __init__(self):
        self.meta = SimpleNamespace(endpoint_url="https://account.r2.cloudflarestorage.com")
        self.objects = {}
        self.metadata = {}
        self.uploads = []
        self.buckets = []
        self.fail_upload = False
        self.fail_manifest = False
        self.corrupt_head = False

    def head_bucket(self, *, Bucket):
        self.buckets.append(Bucket)

    def upload_fileobj(self, stream, bucket, key, *, ExtraArgs, Config):
        if self.fail_upload:
            raise OSError("simulated offline storage")
        self.objects[key] = stream.read()
        self.metadata[key] = ExtraArgs["Metadata"]
        self.uploads.append(key)

    def head_object(self, *, Bucket, Key):
        return {
            "ContentLength": len(self.objects[Key]) + int(self.corrupt_head),
            "Metadata": self.metadata[Key],
        }

    def put_object(self, *, Bucket, Key, Body, ContentType):
        if self.fail_manifest and Key.endswith("/manifest.json"):
            raise OSError("simulated manifest upload failure")
        self.objects[Key] = Body


@pytest.fixture
def fake_s3(monkeypatch):
    client = FakeS3()
    monkeypatch.setattr("gymemu.storage.r2_client", lambda: client)
    return client


def config_for(snapshot, output, approach="direct"):
    cfg = compose_config(["game=custom", f"approach={approach}", "experiment=smoke"])
    cfg.game.dataset = str(snapshot)
    cfg.game.env_id = "Fixture-v0"
    cfg.wandb.mode = "disabled"
    cfg.output = str(output)
    return cfg


@pytest.mark.parametrize("approach", ["direct", "latent"])
def test_uploaded_checkpoint_and_scene_restore_playback(snapshot, tmp_path, fake_s3, approach):
    cfg = config_for(snapshot, tmp_path / "run", approach)
    output = train(cfg)
    receipt = json.loads((output / "r2.json").read_text())
    manifest = json.loads(fake_s3.objects[f"{receipt['prefix']}/manifest.json"])
    assert manifest["status"] == receipt["status"] == "complete"
    assert manifest["bucket"] == "gymemu-models"
    assert all(bucket == "gymemu-models" for bucket in fake_s3.buckets)
    assert receipt["prefix"].startswith("runs/Fixture-v0/")
    assert {
        "best.pt",
        "last.pt",
        "latest.pt",
        "start-scene.npz",
        "recipe.yaml",
        "source.tar.gz",
        "reproduction.json",
        "summary.json",
        "metrics.jsonl",
    } <= manifest["objects"].keys()
    assert "r2.json" not in manifest["objects"]
    downloaded = tmp_path / "downloaded"
    for relative, record in manifest["objects"].items():
        payload = fake_s3.objects[record["key"]]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        assert len(payload) == record["size_bytes"]
        path = downloaded / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    model, metadata = load_model(downloaded / "best.pt", torch.device("cpu"))
    player = Player(
        model, metadata, torch.device("cpu"), load_scene(downloaded / "start-scene.npz", metadata)
    )
    player.advance(2)
    assert player.steps == 1 and player.pixels().shape == (21, 17, 3)
    summary = json.loads((downloaded / "summary.json").read_text())
    assert summary["r2_manifest_uri"].endswith(f"{receipt['prefix']}/manifest.json")
    if approach == "latent":
        assert "stages/representation/best.pt" in manifest["objects"]
        assert "stages/dynamics/best.pt" in manifest["objects"]
    count = len(fake_s3.uploads)
    upload_main([str(output)])
    assert len(fake_s3.uploads) == count
    assert json.loads((output / "r2.json").read_text())["run_id"] == receipt["run_id"]


def test_failed_upload_keeps_previous_manifest_until_retry(snapshot, tmp_path, fake_s3):
    output = tmp_path / "run"
    output.mkdir()
    cfg = config_for(snapshot, output)
    store = CheckpointStore(OmegaConf.to_container(cfg, resolve=True), output)
    (output / "latest.pt").write_bytes(b"first checkpoint")
    store.sync()
    key = f"{store.state['prefix']}/manifest.json"
    previous = fake_s3.objects[key]
    fake_s3.fail_upload = True
    (output / "latest.pt").write_bytes(b"second checkpoint")
    store.sync()
    assert store.state["status"] == "pending"
    assert fake_s3.objects[key] == previous
    with pytest.raises(RuntimeError, match="local files are intact"):
        store.sync(final=True)
    fake_s3.fail_upload = False
    retried = CheckpointStore(OmegaConf.to_container(cfg, resolve=True), output)
    retried.sync(final=True)
    manifest = json.loads(fake_s3.objects[key])
    assert manifest["status"] == "complete"
    assert fake_s3.objects[manifest["objects"]["latest.pt"]["key"]] == b"second checkpoint"
    assert any(value == previous for k, value in fake_s3.objects.items() if "/manifests/" in k)


@pytest.mark.parametrize("failure", ["corrupt_head", "fail_manifest"])
def test_upload_verification_and_manifest_failure_remain_retryable(
    snapshot, tmp_path, fake_s3, failure
):
    cfg = config_for(snapshot, tmp_path)
    store = CheckpointStore(OmegaConf.to_container(cfg, resolve=True), tmp_path)
    (tmp_path / "best.pt").write_bytes(b"weights")
    setattr(fake_s3, failure, True)
    with pytest.raises(RuntimeError, match="R2 upload incomplete"):
        store.sync(final=True)
    assert json.loads((tmp_path / "r2.json").read_text())["status"] == "pending"
    assert f"{store.state['prefix']}/manifest.json" not in fake_s3.objects
    setattr(fake_s3, failure, False)
    store.sync(final=True)
    assert store.state["status"] == "complete"


def test_disabled_and_old_recipes_need_no_sdk_or_credentials(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)
    for config in ({}, {"r2": {"enabled": False}}):
        store = CheckpointStore(config, tmp_path)
        store.sync(final=True)
        assert store.summary() == {}
        assert not (tmp_path / "r2.json").exists()


def test_missing_credentials_fail_before_dataset_access(snapshot, tmp_path, monkeypatch):
    for suffix in ("ENDPOINT_URL", "ACCESS_KEY_ID", "SECRET_ACCESS_KEY"):
        monkeypatch.delenv(f"GYMEMU_MODELS_R2_{suffix}", raising=False)
    with pytest.raises(ValueError, match="GYMEMU_MODELS_R2_ACCESS_KEY_ID"):
        train(config_for(snapshot, tmp_path / "run"))
    assert not (tmp_path / "run").exists()


def test_storage_validation_rejects_missing_identity_and_bad_prefix():
    config = {
        "game": {"env_id": None},
        "r2": {"enabled": True, "bucket": "gymemu-models", "prefix": "runs"},
    }
    with pytest.raises(ValueError, match="canonical environment ID"):
        validate_storage(config)
    config["game"]["env_id"] = "Fixture-v0"
    config["r2"]["prefix"] = "../gradlab"
    with pytest.raises(ValueError, match="r2.prefix"):
        validate_storage(config)


def test_runtime_credentials_are_explicit_and_use_r2_sdk_settings(monkeypatch):
    import boto3

    options = {}
    monkeypatch.setattr(boto3, "client", lambda service, **kwargs: options.update(kwargs))
    monkeypatch.setenv("GYMEMU_MODELS_R2_ENDPOINT_URL", "https://account.r2.cloudflarestorage.com")
    monkeypatch.setenv("GYMEMU_MODELS_R2_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("GYMEMU_MODELS_R2_SECRET_ACCESS_KEY", "test-secret-key")
    r2_client()
    assert options["aws_access_key_id"] == "test-access-key"
    assert options["aws_secret_access_key"] == "test-secret-key"
    assert options["region_name"] == "auto"
    assert options["config"].request_checksum_calculation == "when_required"
    config = OmegaConf.to_yaml(compose_config())
    assert "test-secret-key" not in config and "test-access-key" not in config
