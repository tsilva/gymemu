"""Launch one resolved recipe on bounded dstack compute."""

import argparse
import base64
import hashlib
import json
import math
import re
import shlex
import subprocess
from pathlib import Path
from uuid import uuid4

import yaml
from omegaconf import OmegaConf

from gymemu.config import compose_config
from gymemu.research import GOALS
from gymemu.resources import RESOURCE_ROOT
from gymemu.storage import r2_client

IMAGE = re.compile(r"ghcr\.io/tsilva/gymemu/train@sha256:[0-9a-f]{64}\Z")
DURATION = re.compile(r"[1-9][0-9]*(?:m|h|d)\Z")
SECRETS = (
    "WANDB_API_KEY",
    "GYMEMU_MODELS_R2_ENDPOINT_URL",
    "GYMEMU_MODELS_R2_ACCESS_KEY_ID",
    "GYMEMU_MODELS_R2_SECRET_ACCESS_KEY",
)


def render_task(recipe, *, image, run_id, compute, max_duration, max_price):
    if not IMAGE.fullmatch(image):
        raise ValueError("Image must be an immutable Gymemu sha256 reference")
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("Invalid Run ID")
    if compute not in ("local", "spot", "on-demand"):
        raise ValueError("Compute must be local, spot, or on-demand")
    if not DURATION.fullmatch(max_duration):
        raise ValueError("A finite max duration is required")
    if compute != "local" and (max_price is None or not math.isfinite(max_price) or max_price <= 0):
        raise ValueError("Paid compute requires a positive hourly max price")
    encoded = base64.b64encode(recipe.encode()).decode()
    command = (
        f"printf %s {shlex.quote(encoded)} | base64 -d > /workspace/recipe.yaml && "
        f"gymemu-container run --recipe /workspace/recipe.yaml "
        f"output=/workspace/runs/{run_id}"
    )
    task = {
        "type": "task",
        "name": f"gymemu-{run_id[:12]}",
        "image": image,
        "working_dir": "/workspace",
        "env": [f"GYMEMU_IMAGE_REF={image}", f"GYMEMU_COMPUTE_PLACEMENT={compute}"]
        + [f"{key}=${{{{ secrets.{key} }}}}" for key in SECRETS],
        "commands": [command],
        "resources": {"cpu": "12..", "memory": "40GB..", "gpu": "1"},
        "max_duration": max_duration,
        "stop_duration": "5m",
    }
    if compute == "local":
        task.update(fleets=["b3"], creation_policy="reuse")
    else:
        task.update(spot_policy=compute, max_price=max_price)
    return task


def verify_image_source(image):
    """Reject a digest built from a different source commit."""
    inspected = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", "--format", "{{json .Image}}", image],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(inspected.stdout)
    revision = value.get("config", {}).get("Labels", {}).get("org.opencontainers.image.revision")
    checkout = subprocess.run(
        ["git", "-C", str(RESOURCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != checkout:
        raise ValueError(
            f"Image source revision {revision or 'missing'} differs from checkout {checkout}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("launch")
    launch.add_argument("--recipe-file", type=Path, required=True)
    launch.add_argument("--compute", choices=("local", "spot", "on-demand"), required=True)
    launch.add_argument("--image", default=None)
    launch.add_argument("--max-duration", default="1h")
    launch.add_argument("--max-hourly-price", type=float)
    launch.add_argument("--output-dir", type=Path)
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--follow", action="store_true")
    launch.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    if args.follow and args.dry_run:
        parser.error("--follow requires a submitted task")
    image = args.image or (RESOURCE_ROOT / "ops/dstack/verified-image.txt").read_text().strip()
    run_id = uuid4().hex
    selected_path = args.recipe_file.expanduser().resolve()
    if selected_path.is_relative_to(GOALS.resolve()):
        alias = RESOURCE_ROOT / "configs/recipe" / selected_path.name
        if not alias.is_file() or alias.read_bytes() != selected_path.read_bytes():
            parser.error("Goal recipe and compatibility alias differ")
        config = compose_config([f"recipe={selected_path.stem}", *args.overrides])
    else:
        config = compose_config(args.overrides, recipe=selected_path)
    config.run_id = run_id
    config.output = None
    config.launch_overrides = args.overrides
    if config.wandb.mode != "online" or not config.r2.enabled:
        parser.error("Queued Runs require online W&B and R2 publication")
    recipe = OmegaConf.to_yaml(config, resolve=True)
    task = render_task(
        recipe,
        image=image,
        run_id=run_id,
        compute=args.compute,
        max_duration=args.max_duration,
        max_price=args.max_hourly_price,
    )
    if args.dry_run:
        print(yaml.safe_dump(task, sort_keys=False), end="")
        return
    verify_image_source(image)
    client = r2_client()
    client.head_bucket(Bucket=config.r2.bucket)
    directory = args.output_dir or Path.home() / ".local/share/gymemu/launches" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "recipe.yaml").write_text(recipe)
    task_file = directory / "task.dstack.yml"
    task_file.write_text(yaml.safe_dump(task, sort_keys=False))
    receipt = {
        "id": run_id,
        "status": "submitted",
        "compute": args.compute,
        "image": image,
        "recipe_sha256": hashlib.sha256(recipe.encode()).hexdigest(),
        "dstack_name": task["name"],
        "catalog_status": None,
    }
    (directory / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    subprocess.run(
        [
            "dstack",
            "apply",
            "-f",
            str(task_file),
            "--project",
            "main",
            "--no-repo",
            "--yes",
            "--detach",
        ],
        check=True,
        cwd=directory,
    )
    print(f"Submitted Run {run_id} as {task['name']}")
    if args.follow:
        subprocess.run(
            ["dstack", "attach", task["name"], "--logs", "--project", "main"],
            check=True,
        )
        from gymemu.research_catalog import ResearchCatalog

        catalog = ResearchCatalog(bucket=config.r2.bucket, prefix=config.r2.prefix)
        state = catalog.snapshot(run_id=run_id, refresh=True)
        matches = [
            run
            for env in state["environments"]
            for goal in env["goals"]
            for revision in goal["revisions"]
            for run in revision["runs"]
            if run["id"] == run_id
        ]
        receipt["catalog_status"] = matches[0]["status"] if matches else "unpublished"
        (directory / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"Authoritative catalog state: {receipt['catalog_status']}")


if __name__ == "__main__":
    main()
