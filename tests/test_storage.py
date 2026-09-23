"""R2 manifests must only reference verified, replayable run artifacts."""

import hashlib
import json
import sys
from contextlib import contextmanager
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
        stream.close()  # boto3's upload_fileobj consumes and closes its input.
        self.metadata[key] = ExtraArgs["Metadata"]
        self.uploads.append(key)

    def head_object(self, *, Bucket, Key):
        return {
            "ContentLength": len(self.objects[Key]) + int(self.corrupt_head),
            "Metadata": self.metadata[Key],
        }

    def put_object(self, *, Bucket, Key, Body, ContentType, IfMatch=None, IfNoneMatch=None):
        if self.fail_manifest and Key.endswith("/manifest.json"):
            raise OSError("simulated manifest upload failure")
        current = self.objects.get(Key)
        if IfNoneMatch == "*" and current is not None:
            raise ValueError("precondition failed: object exists")
        if IfMatch is not None and (
            current is None or '"' + hashlib.md5(current).hexdigest() + '"' != IfMatch
        ):
            raise ValueError("precondition failed: object changed")
        self.objects[Key] = Body
        return {"ETag": '"' + hashlib.md5(Body).hexdigest() + '"'}


@pytest.fixture
def fake_s3(monkeypatch):
    client = FakeS3()
    monkeypatch.setattr("gymemu.storage.r2_client", lambda **kwargs: client)
    from gymemu.tracking import Tracker

    @contextmanager
    def fake_tracking(*args, **kwargs):
        yield Tracker()

    monkeypatch.setattr("gymemu.engine.track_run", fake_tracking)
    return client


def config_for(snapshot, output, approach="direct"):
    cfg = compose_config(["game=custom", f"approach={approach}", "experiment=smoke"])
    cfg.game.dataset = str(snapshot)
    cfg.game.env_id = "Fixture-v0"
    cfg.wandb.mode = "online"
    cfg.r2.public_base_url = "https://gymemu-assets.example.test"
    cfg.output = str(output)
    return cfg


@pytest.mark.parametrize("approach", ["direct", "latent"])
def test_uploaded_checkpoint_and_scene_restore_playback(snapshot, tmp_path, fake_s3, approach):
    cfg = config_for(snapshot, tmp_path / "run", approach)
    output = train(cfg)
    receipt = json.loads((output / "r2.json").read_text())
    manifest = json.loads(fake_s3.objects[f"{receipt['prefix']}/manifest.json"])
    assert manifest["status"] == receipt["status"] == "complete"
    index = json.loads(fake_s3.objects[f"runs/catalog/runs/{receipt['run_id']}.json"])
    assert index["id"] == receipt["run_id"]
    assert index["status"] == "complete"
    assert index["manifest_key"] == f"{receipt['prefix']}/manifest.json"
    assert index["best_mse"] == json.loads((output / "summary.json").read_text())["best_mse"]
    assert manifest["bucket"] == "gymemu"
    assert set(fake_s3.buckets) == {"gymemu", "gymemu-public"}
    assert "best.pt" in manifest["public_objects"]
    assert "start-scene.npz" in manifest["public_objects"]
    assert "resume.pt" not in manifest["public_objects"]
    assert "run.json" not in manifest["public_objects"]
    for name, public in manifest["public_objects"].items():
        assert public["url"].startswith("https://gymemu-assets.example.test/inference/")
        assert fake_s3.objects[public["url"].split("/", 3)[-1]] == (output / name).read_bytes()
    assert receipt["prefix"].startswith("runs/Fixture-v0/")
    assert {
        "resume.pt",
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


def test_finished_training_waits_for_required_publication(snapshot, tmp_path, fake_s3):
    cfg = config_for(snapshot, tmp_path / "run")
    fake_s3.fail_manifest = True
    with pytest.raises(RuntimeError, match="local files are intact"):
        train(cfg)
    run = json.loads((tmp_path / "run/run.json").read_text())
    assert run["status"] == "publication_pending"
    run_id = run["id"]
    fake_s3.fail_manifest = False
    upload_main([str(tmp_path / "run")])
    assert json.loads((tmp_path / "run/r2.json").read_text())["run_id"] == run_id
    assert json.loads((tmp_path / "run/run.json").read_text())["status"] == "complete"


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


def test_mutable_manifest_rejects_concurrent_change(snapshot, tmp_path, fake_s3):
    cfg = config_for(snapshot, tmp_path)
    store = CheckpointStore(OmegaConf.to_container(cfg, resolve=True), tmp_path)
    (tmp_path / "best.pt").write_bytes(b"weights")
    store.sync(final=True)
    pointer = f"{store.state['prefix']}/manifest.json"
    fake_s3.objects[pointer] = b'{"other": "writer"}'
    with pytest.raises(RuntimeError, match="R2 upload incomplete"):
        store.sync(final=True)
    assert fake_s3.objects[pointer] == b'{"other": "writer"}'


def test_public_copy_uses_same_file_when_checkpoint_is_replaced(snapshot, tmp_path, fake_s3):
    cfg = config_for(snapshot, tmp_path)
    store = CheckpointStore(OmegaConf.to_container(cfg, resolve=True), tmp_path)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"first version")
    original_upload = fake_s3.upload_fileobj

    def replace_after_private_upload(stream, bucket, key, *, ExtraArgs, Config):
        original_upload(stream, bucket, key, ExtraArgs=ExtraArgs, Config=Config)
        if bucket == "gymemu":
            newer = tmp_path / "best.tmp"
            newer.write_bytes(b"second version")
            newer.replace(checkpoint)

    fake_s3.upload_fileobj = replace_after_private_upload
    store.sync(final=True)
    manifest = json.loads(fake_s3.objects[f"{store.state['prefix']}/manifest.json"])
    private = manifest["objects"]["best.pt"]
    public = manifest["public_objects"]["best.pt"]
    assert private["sha256"] == public["sha256"]
    assert fake_s3.objects[private["key"]] == b"first version"
    assert fake_s3.objects[public["url"].split("/", 3)[-1]] == b"first version"
    assert checkpoint.read_bytes() == b"second version"


def test_disabled_and_old_recipes_need_no_sdk_or_credentials(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)
    for config in ({}, {"r2": {"enabled": False}}):
        store = CheckpointStore(config, tmp_path)
        store.sync(final=True)
        assert store.summary() == {}
        assert not (tmp_path / "r2.json").exists()


def test_disabled_wandb_keeps_run_offline_even_if_r2_is_configured(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)
    store = CheckpointStore(
        {
            "game": {"env_id": "Fixture-v0"},
            "wandb": {"mode": "disabled"},
            "r2": {"enabled": True, "bucket": "gymemu", "prefix": "runs"},
        },
        tmp_path,
    )
    store.sync(final=True)
    assert not store.enabled
    assert not (tmp_path / "r2.json").exists()


def test_offline_run_syncs_under_its_original_identity(snapshot, tmp_path, fake_s3, monkeypatch):
    from gymemu.commands.sync import main as sync_main

    cfg = config_for(snapshot, tmp_path / "offline")
    cfg.wandb.mode = "disabled"
    output = train(cfg)
    run_id = json.loads((output / "run.json").read_text())["id"]
    assert not (output / "r2.json").exists()
    assert not fake_s3.objects

    class FakeRun:
        url = "https://wandb.ai/test/gymemu/runs/abc"
        summary = {}

        def log(self, value):
            pass

        def finish(self, exit_code=0):
            pass

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: FakeRun()))
    sync_main([str(output)])
    receipt = json.loads((output / "r2.json").read_text())
    assert receipt["run_id"] == run_id
    assert f"runs/catalog/runs/{run_id}.json" in fake_s3.objects
    run_path = output / "run.json"
    run = json.loads(run_path.read_text())
    run["status"] = "publication_pending"
    run["publication"] = "offline"
    run_path.write_text(json.dumps(run))
    sync_main([str(output)])
    retried = json.loads(run_path.read_text())
    assert retried["id"] == run_id
    assert retried["status"] == "complete"
    assert retried["publication"] == "online"


def test_missing_credentials_fail_before_dataset_access(snapshot, tmp_path, monkeypatch):
    monkeypatch.setenv("GYMEMU_R2_CONFIG", str(tmp_path / "missing-r2.toml"))
    for suffix in ("ENDPOINT_URL", "ACCESS_KEY_ID", "SECRET_ACCESS_KEY"):
        monkeypatch.delenv(f"GYMEMU_MODELS_R2_{suffix}", raising=False)
    with pytest.raises(ValueError, match="GYMEMU_MODELS_R2_ACCESS_KEY_ID"):
        train(config_for(snapshot, tmp_path / "run"))
    assert not (tmp_path / "run").exists()


def test_storage_validation_rejects_missing_identity_and_bad_prefix():
    config = {
        "game": {"env_id": None},
        "r2": {"enabled": True, "bucket": "gymemu", "prefix": "runs"},
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
