"""The managed view has explicit, validated selectors and preserves GradLab conventions."""

import json
import re

import pytest

from gymemu.config import CONFIG_DIR
from gymemu.workspace import build_workspace


def test_workspace_groups_registered_metrics_and_keeps_primary_first():
    pytest.importorskip("wandb_workspaces")
    view = build_workspace("test", "test")
    primary, *others = view.sections
    assert primary.name == "Primary metrics" and primary.pinned and primary.is_open
    assert all(not section.is_open and not section.pinned for section in others)
    assert view.auto_generate_panels is False
    assert view.settings.smoothing_weight == 0
    assert view.settings.x_axis == "train/step"
    assert primary.panels[0].x == "eval/step"
    assert re.fullmatch(primary.panels[1].metric_regex, "probe/mse/h8")
    assert not re.fullmatch(primary.panels[1].metric_regex, "probe/mse/h80")
    assert all(p.x == "optimizer_steps" for p in others[-1].panels)


def test_workspace_rejects_misspelled_metrics():
    pytest.importorskip("wandb_workspaces")
    declaration = json.loads((CONFIG_DIR / "monitoring/workspace.json").read_text())
    declaration["sections"][0]["panels"][0]["metrics"] = ["eval/msee"]
    with pytest.raises(ValueError, match="Unregistered metric"):
        build_workspace("test", "test", declaration)
