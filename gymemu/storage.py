"""Publish local run files to R2 as immutable objects and a committed manifest."""

import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from gymemu.credentials import CREDENTIAL_PREFIX, PUBLIC_CREDENTIAL_PREFIX, r2_credentials

ARTIFACTS = (
    "run.json",
    "goal.yaml",
    "probe.json",
    "diagnostics.jsonl",
    "resume.pt",
    "best.pt",
    "last.pt",
    "latest.pt",
    "start-scene.npz",
    "config.json",
    "resolved.yaml",
    "recipe.yaml",
    "source.tar.gz",
    "reproduction.json",
    "metrics.jsonl",
    "summary.json",
)


def inference_artifact(name):
    return name == "start-scene.npz" or (
        name.endswith(".pt") and name != "resume.pt" and "/resume" not in name
    )


def validate_storage(config):
    settings = config.get("r2", {})
    enabled = settings.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("r2.enabled must be true or false")
    if not enabled:
        return
    bucket = settings.get("bucket", "")
    if not isinstance(bucket, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", bucket):
        raise ValueError("r2.bucket must be a valid bucket name")
    prefix = settings.get("prefix", "")
    if not isinstance(prefix, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*", prefix
    ):
        raise ValueError(
            "r2.prefix must use slash-separated letters, digits, hyphens or underscores"
        )
    env_id = config["game"].get("env_id")
    if not isinstance(env_id, str) or not env_id.strip():
        raise ValueError("R2 uploads require game.env_id with the canonical environment ID")


def r2_client(*, public=False):
    """Read credentials at runtime, never from saved Hydra/W&B configuration."""
    prefix = PUBLIC_CREDENTIAL_PREFIX if public else CREDENTIAL_PREFIX
    values = r2_credentials(public=public)
    missing = [f"{prefix}_{name}" for name, value in values.items() if not value]
    if missing:
        raise ValueError("R2 uploads require environment variables: " + ", ".join(missing))
    endpoint = urlparse(values["ENDPOINT_URL"])
    if (
        endpoint.scheme != "https"
        or not endpoint.hostname
        or not endpoint.hostname.endswith(".r2.cloudflarestorage.com")
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
        or endpoint.path not in ("", "/")
    ):
        raise ValueError(f"{prefix}_ENDPOINT_URL must be an HTTPS R2 account endpoint")
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=values["ENDPOINT_URL"],
        region_name="auto",
        aws_access_key_id=values["ACCESS_KEY_ID"],
        aws_secret_access_key=values["SECRET_ACCESS_KEY"],
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=5,
            read_timeout=30,
            retries={"mode": "standard", "total_max_attempts": 3},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(_json(value))
    temporary.replace(path)


class CheckpointStore:
    def __init__(self, config, output):
        validate_storage(config)
        self.output = Path(output)
        self.enabled = config.get("r2", {}).get("enabled", False) and (
            config.get("wandb", {}).get("mode") == "online"
        )
        self.client = None
        self.state = None
        if not self.enabled:
            return
        settings = config["r2"]
        self.bucket = settings["bucket"]
        self.public_bucket = settings.get("public_bucket", "")
        self.public_base_url = settings.get("public_base_url") or os.environ.get(
            "GYMEMU_PUBLIC_R2_BASE_URL"
        )
        public_url = urlparse(self.public_base_url or "")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", self.public_bucket):
            raise ValueError("r2.public_bucket must name the separate public bucket")
        if (
            public_url.scheme != "https"
            or not public_url.hostname
            or public_url.username
            or public_url.password
            or public_url.query
            or public_url.fragment
        ):
            raise ValueError("GYMEMU_PUBLIC_R2_BASE_URL must be an HTTPS bucket URL")
        self.public_base_url = self.public_base_url.rstrip("/")
        self.catalog_prefix = settings["prefix"]
        self.client = r2_client()
        self.public_client = r2_client(public=True)
        endpoint = self.client.meta.endpoint_url
        if self.public_client.meta.endpoint_url != endpoint:
            raise ValueError("Private and public R2 credentials must use the same account endpoint")
        # Fail before dataset loading or optimization when access is misconfigured.
        self.client.head_bucket(Bucket=self.bucket)
        self.public_client.head_bucket(Bucket=self.public_bucket)
        receipt = self.output / "r2.json"
        if receipt.exists():
            self.state = json.loads(receipt.read_text())
            if self.state["bucket"] != self.bucket or self.state["endpoint_url"] != endpoint:
                raise ValueError("R2 retry destination differs from the saved upload receipt")
            if (
                self.state.get("public_bucket") != self.public_bucket
                or self.state.get("public_base_url") != self.public_base_url
            ):
                raise ValueError(
                    "Public R2 retry destination differs from the saved upload receipt"
                )
        else:
            run_file = self.output / "run.json"
            run_id = (
                json.loads(run_file.read_text())["id"]
                if run_file.is_file()
                else (config.get("run_id") or uuid4().hex)
            )
            environment = re.sub(r"[^A-Za-z0-9_.-]", "-", config["game"]["env_id"].strip())
            self.state = {
                "version": 1,
                "bucket": self.bucket,
                "endpoint_url": endpoint,
                "prefix": f"{settings['prefix']}/{environment}/{run_id}",
                "run_id": run_id,
                "env_id": config["game"]["env_id"],
                "objects": {},
                "public_bucket": self.public_bucket,
                "public_base_url": self.public_base_url,
                "public_objects": {},
                "status": "pending",
            }

    @property
    def uri(self):
        if not self.enabled:
            return None
        return f"s3://{self.bucket}/{self.state['prefix']}/manifest.json"

    def _files(self):
        files = [self.output / name for name in ARTIFACTS]
        files.extend(sorted(self.output.glob("stages/*/*.pt")))
        files.extend(sorted(self.output.glob("diagnostics/*.png"))[:12])
        for path in files:
            if path.is_file():
                if path.is_symlink() or not path.resolve().is_relative_to(self.output.resolve()):
                    raise ValueError("R2 artifacts must be regular files inside the run directory")
                yield path

    def _upload(self, path):
        # Atomic checkpoint replacement leaves an already-open file descriptor stable.
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            size = os.fstat(stream.fileno()).st_size
            stream.seek(0)
            key = f"{self.state['prefix']}/objects/{digest}"
            record = {"key": key, "sha256": digest, "size_bytes": size}
            relative = path.relative_to(self.output).as_posix()
            # boto3 closes its input; a duplicate descriptor keeps the same inode
            # available even if an atomic checkpoint save replaces the pathname.
            public_stream = (
                os.fdopen(os.dup(stream.fileno()), "rb") if inference_artifact(relative) else None
            )
            try:
                if (
                    self.state["objects"].get(relative) != record
                    and record not in self.state["objects"].values()
                ):
                    # Identical root and stage checkpoint aliases share one immutable object.
                    from boto3.s3.transfer import TransferConfig

                    self.client.upload_fileobj(
                        stream,
                        self.bucket,
                        key,
                        ExtraArgs={"Metadata": {"sha256": digest}},
                        Config=TransferConfig(max_concurrency=2, use_threads=False),
                    )
                    head = self.client.head_object(Bucket=self.bucket, Key=key)
                    if (
                        head["ContentLength"] != size
                        or head.get("Metadata", {}).get("sha256") != digest
                    ):
                        raise RuntimeError(
                            "R2 upload size or checksum metadata differs from local file"
                        )
                self.state["objects"][relative] = record
                if public_stream:
                    self._upload_public(public_stream, relative, digest, size)
            finally:
                if public_stream:
                    public_stream.close()

    def _upload_public(self, stream, relative, digest, size):
        key = f"inference/{digest}"
        record = {
            "url": f"{self.public_base_url}/{key}",
            "sha256": digest,
            "size_bytes": size,
        }
        if self.state["public_objects"].get(relative) == record:
            return
        from boto3.s3.transfer import TransferConfig

        stream.seek(0)
        self.public_client.upload_fileobj(
            stream,
            self.public_bucket,
            key,
            ExtraArgs={"Metadata": {"sha256": digest}},
            Config=TransferConfig(max_concurrency=2, use_threads=False),
        )
        head = self.public_client.head_object(Bucket=self.public_bucket, Key=key)
        if head["ContentLength"] != size or head.get("Metadata", {}).get("sha256") != digest:
            raise RuntimeError("Public R2 upload size or checksum differs from local file")
        self.state["public_objects"][relative] = record

    def sync(self, *, final=False):
        if not self.enabled:
            return
        # Save the run identity before I/O so retries always reuse this remote prefix.
        _write_json(self.output / "r2.json", self.state)
        try:
            for path in self._files():
                self._upload(path)
            manifest = {
                key: value
                for key, value in self.state.items()
                if key not in ("status", "error_type", "manifest_etag", "catalog_etag")
            }
            manifest["status"] = "complete" if final else "running"
            payload = _json(manifest)
            digest = hashlib.sha256(payload).hexdigest()
            # Retain prior manifests so earlier periodic checkpoints remain discoverable.
            self.client.put_object(
                Bucket=self.bucket,
                Key=f"{self.state['prefix']}/manifests/{digest}.json",
                Body=payload,
                ContentType="application/json",
            )
            pointer = self.client.put_object(
                Bucket=self.bucket,
                Key=f"{self.state['prefix']}/manifest.json",
                Body=payload,
                ContentType="application/json",
                **(
                    {"IfMatch": self.state["manifest_etag"]}
                    if self.state.get("manifest_etag")
                    else {"IfNoneMatch": "*"}
                ),
            )
            self.state["manifest_etag"] = pointer["ETag"]
            run_file = self.output / "run.json"
            run = json.loads(run_file.read_text()) if run_file.is_file() else {}
            summary_file = self.output / "summary.json"
            summary = json.loads(summary_file.read_text()) if summary_file.is_file() else {}
            goal = run.get("goal") or {}
            index = {
                "version": 1,
                "id": self.state["run_id"],
                "created_at": run.get("created_at"),
                "environment": self.state["env_id"],
                "goal": goal.get("id"),
                "revision": goal.get("revision"),
                "variant": goal.get("variant"),
                "goal_contract": goal.get("contract"),
                "goal_diff": goal.get("diff", []),
                "comparability": run.get("comparability"),
                "recipe_sha256": run.get("recipe", {}).get("sha256"),
                "wandb_url": (run.get("wandb") or {}).get("url"),
                "name": summary.get("name") or run.get("id") or self.state["run_id"],
                "approach": summary.get("approach"),
                "status": manifest["status"],
                "best_mse": summary.get("best_mse"),
                "checkpoint_count": sum(
                    name.endswith(".pt") and name != "resume.pt" for name in manifest["objects"]
                ),
                "has_final_prediction": "best.pt" in manifest["objects"],
                "manifest_key": f"{self.state['prefix']}/manifest.json",
            }
            projection = self.client.put_object(
                Bucket=self.bucket,
                Key=f"{self.catalog_prefix}/catalog/runs/{self.state['run_id']}.json",
                Body=_json(index),
                ContentType="application/json",
                **(
                    {"IfMatch": self.state["catalog_etag"]}
                    if self.state.get("catalog_etag")
                    else {"IfNoneMatch": "*"}
                ),
            )
            self.state["catalog_etag"] = projection["ETag"]
            self.state["status"] = manifest["status"]
            self.state.pop("error_type", None)
        except Exception as error:
            self.state["status"] = "pending"
            self.state["error_type"] = type(error).__name__
            if final:
                raise RuntimeError(
                    "R2 upload incomplete; local files are intact. Retry with "
                    f"upload_checkpoints.py {str(self.output)!r}"
                ) from error
            print(
                f"R2 upload pending ({type(error).__name__}); retrying at the next save.",
                flush=True,
            )
        finally:
            _write_json(self.output / "r2.json", self.state)

    def summary(self):
        if not self.enabled:
            return {}
        return {"r2_manifest_uri": self.uri, "r2_run_id": self.state["run_id"]}
