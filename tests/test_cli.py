"""Exercise installed entry points from outside the checkout."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

CLI = Path(sys.executable).with_name("gymemu")


def run_cli(directory, *arguments):
    result = subprocess.run(
        [str(CLI), *arguments],
        cwd=directory,
        env={**os.environ, "HYDRA_FULL_ERROR": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_cli_help_and_errors(tmp_path):
    assert "cache-frames" in run_cli(tmp_path, "--help")
    assert run_cli(tmp_path, "--version").strip()
    for command in [
        "play",
        "compare",
        "cache-frames",
        "save-start-state",
        "upload-checkpoints",
        "dynamics",
    ]:
        assert "usage:" in run_cli(tmp_path, command, "--help")
    result = subprocess.run([str(CLI), "unknown"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert "unknown command" in result.stderr


@pytest.mark.parametrize("approach", ["direct", "latent"])
def test_cli_train_and_play(snapshot, tmp_path, approach):
    output = tmp_path / approach
    run_cli(
        tmp_path,
        "train",
        "game=custom",
        f"game.dataset={snapshot}",
        f"approach={approach}",
        "experiment=smoke",
        "wandb.mode=disabled",
        "r2.enabled=false",
        f"output={output}",
    )
    checkpoint = output / "best.pt"
    assert checkpoint.is_file()
    run_cli(
        tmp_path,
        "play",
        str(checkpoint),
        "--device",
        "cpu",
        "--headless-actions",
        "0,2",
        "--output",
        str(tmp_path / f"{approach}.png"),
    )
    assert (tmp_path / f"{approach}.png").is_file()
    assert (output / "reproduction.json").is_file()
