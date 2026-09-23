"""dstack placement may not change the resolved training recipe."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gymemu.commands import experiment
from gymemu.commands.experiment import render_task


def test_queued_task_preserves_recipe_and_bounds_paid_compute():
    image = "ghcr.io/tsilva/gymemu/train@sha256:" + "a" * 64
    recipe = "name: tiny\nrun_id: abc\n"
    local = render_task(
        recipe,
        image=image,
        run_id="a" * 32,
        compute="local",
        max_duration="1h",
        max_price=None,
    )
    spot = render_task(
        recipe,
        image=image,
        run_id="a" * 32,
        compute="spot",
        max_duration="1h",
        max_price=2.0,
    )
    assert local["image"] == spot["image"] == image
    assert local["commands"] == spot["commands"]
    assert local["fleets"] == ["b3"]
    assert spot["spot_policy"] == "spot" and spot["max_price"] == 2.0
    assert spot["max_duration"] == "1h"
    with pytest.raises(ValueError, match="price"):
        render_task(
            recipe,
            image=image,
            run_id="a" * 32,
            compute="on-demand",
            max_duration="1h",
            max_price=None,
        )


def test_queued_cli_preflights_both_buckets_and_keeps_run_identity(tmp_path, monkeypatch):
    image = "ghcr.io/tsilva/gymemu/train@sha256:" + "a" * 64
    recipe = Path(__file__).parents[1] / "experiments/goals/breakout/next-frame/recipes/direct.yaml"
    buckets = []
    submitted = []
    monkeypatch.setattr(experiment, "verify_image_source", lambda _: None)
    monkeypatch.setattr(
        experiment,
        "r2_client",
        lambda **kwargs: SimpleNamespace(
            head_bucket=lambda *, Bucket: buckets.append((kwargs.get("public", False), Bucket))
        ),
    )
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda args, **kwargs: submitted.append(args),
    )
    directory = tmp_path / "queued"
    experiment.main(
        [
            "launch",
            "--recipe-file",
            str(recipe),
            "--compute",
            "local",
            "--image",
            image,
            "--max-duration",
            "1h",
            "--output-dir",
            str(directory),
            "experiment=smoke",
        ]
    )
    receipt = json.loads((directory / "launch.json").read_text())
    task = yaml.safe_load((directory / "task.dstack.yml").read_text())
    saved = yaml.safe_load((directory / "recipe.yaml").read_text())
    assert receipt["id"] == saved["run_id"]
    assert task["name"] == receipt["dstack_name"]
    assert task["max_duration"] == "1h" and task["fleets"] == ["b3"]
    assert buckets == [(False, "gymemu"), (True, "gymemu-public")]
    assert len(submitted) == 1 and submitted[0][:2] == ["dstack", "apply"]
    assert all("${{ secrets." in value for value in task["env"] if "R2_" in value)
