"""Render or submit a Gymemu task to the existing Beast-3 dstack fleet."""

import argparse
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

import yaml


def configuration(image, name, smoke, offline, overrides):
    if not re.fullmatch(r"ghcr\.io/tsilva/gymemu/train@sha256:[0-9a-f]{64}", image):
        raise ValueError("Use the immutable ghcr.io/tsilva/gymemu/train@sha256:... image reference")
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", name):
        raise ValueError("Task name must contain lowercase letters, digits and hyphens")
    args = ["gymemu-container"]
    if smoke:
        args += ["smoke", "--device", "cuda"]
        if not offline:
            args.append("--online")
    else:
        args += [
            "run",
            "recipe=breakout_ball",
            "trainer.frame_cache=/workspace/frame-cache/676ff638-lz4",
        ]
        if offline:
            args += ["wandb.mode=disabled", "r2.enabled=false"]
        args += overrides
    env = [f"GYMEMU_IMAGE_REF={image}"]
    if not offline:
        env += [
            f"{key}=${{{{ secrets.{key} }}}}"
            for key in (
                "WANDB_API_KEY",
                "GYMEMU_MODELS_R2_ENDPOINT_URL",
                "GYMEMU_MODELS_R2_ACCESS_KEY_ID",
                "GYMEMU_MODELS_R2_SECRET_ACCESS_KEY",
            )
        ]
    return {
        "type": "task",
        "name": name,
        "image": image,
        "working_dir": "/workspace",
        "env": env,
        "commands": [shlex.join(args)],
        "resources": {"cpu": "12..", "memory": "40GB..", "gpu": "1"},
        "volumes": [
            "/home/tsilva/.local/share/gymemu/container-workspace:/workspace",
            "/home/tsilva/.cache/huggingface:/workspace/huggingface",
            "/home/tsilva/.cache/gymemu:/workspace/frame-cache",
        ],
        "creation_policy": "reuse",
        "fleets": ["b3"],
        "max_duration": "1h" if smoke else "24h",
        "stop_duration": "5m",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.smoke and args.overrides:
        parser.error("smoke uses a fixed bounded configuration; overrides apply to training")
    config = configuration(args.image, args.name, args.smoke, args.offline, args.overrides)
    rendered = yaml.safe_dump(config, sort_keys=False)
    if not args.submit:
        print(rendered, end="")
        return
    with tempfile.TemporaryDirectory(prefix="gymemu-dstack-") as directory:
        path = Path(directory).resolve() / "task.dstack.yml"
        path.write_text(rendered)
        subprocess.run(
            [
                "dstack",
                "apply",
                "-f",
                str(path),
                "--project",
                "main",
                "--no-repo",
                "--yes",
                "--detach",
            ],
            check=True,
            cwd=path.parent,
        )


if __name__ == "__main__":
    main()
