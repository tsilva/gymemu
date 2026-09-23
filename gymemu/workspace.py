"""Create or update Gymemu's managed W&B view, following GradLab layout conventions."""

import argparse
import hashlib
import json
import math

from gymemu.config import CONFIG_DIR
from gymemu.metrics import METRICS, pattern, validate_metrics


def build_workspace(entity, project, declaration=None):
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    declaration = declaration or json.loads((CONFIG_DIR / "monitoring/workspace.json").read_text())
    sections = []
    for section in declaration["sections"]:
        panels = []
        for index, panel in enumerate(section["panels"]):
            keys = panel["metrics"]
            title = ", ".join(keys)
            layout = wr.Layout(x=(index % 2) * 12, y=(index // 2) * 8, w=12, h=8)
            if panel.get("kind") == "media":
                panels.append(wr.MediaBrowser(title=title, media_keys=keys, layout=layout))
                continue
            if not section.get("legacy", False):
                for key in keys:
                    if key not in METRICS:
                        validate_metrics({key: None})
            axis = panel.get("axis", "eval/step" if keys[0].startswith("eval/") else "train/step")
            selectors = "^(?:" + "|".join(pattern(key)[1:-1] for key in keys) + ")$"
            panels.append(
                wr.LinePlot(
                    title=title,
                    x=axis,
                    y=[],
                    metric_regex=selectors,
                    smoothing_type="none",
                    layout=layout,
                )
            )
        sections.append(
            ws.Section(
                name=section["title"],
                panels=panels,
                is_open=section.get("open", False),
                pinned=section.get("pinned", False),
                layout_settings=ws.SectionLayoutSettings(
                    columns=2, rows=math.ceil(len(panels) / 2)
                ),
                panel_settings=ws.SectionPanelSettings(
                    x_axis="train/step", smoothing_type="none", smoothing_weight=0
                ),
            )
        )
    return ws.Workspace(
        entity=entity,
        project=project,
        name=declaration["name"],
        sections=sections,
        settings=ws.WorkspaceSettings(
            x_axis="train/step",
            smoothing_type="none",
            smoothing_weight=0,
            sort_panels_alphabetically=False,
            group_by_prefix="first",
        ),
        runset_settings=ws.RunsetSettings(
            pinned_columns=["run:displayName", "run:state", "config:approach.kind", "config:seed"]
        ),
        auto_generate_panels=False,
    )


def sync_workspace(entity, project):
    from wandb_workspaces.workspaces import Workspace

    # Same deterministic managed identity convention as GradLab; never edit a personal view.
    token = hashlib.sha256(f"{entity}\0{project}\0gymemu-diagnostics-v2".encode()).hexdigest()[:11]
    url = f"https://wandb.ai/{entity}/{project}?nw={token}"
    try:
        existing = Workspace.from_url(url)
    except ValueError as exc:
        if "Workspace `" not in str(exc) or "not found" not in str(exc):
            raise
        existing = None
    desired = build_workspace(entity, project)
    desired._internal_name = f"nw-{token}-v"
    desired._internal_id = str(getattr(existing, "_internal_id", "") or "")
    desired.save()
    restored = Workspace.from_url(url)

    def signature(workspace):
        return [
            (
                section.name,
                section.is_open,
                section.pinned,
                [
                    (
                        panel.title,
                        getattr(getattr(panel, "x", None), "name", getattr(panel, "x", None)),
                        getattr(panel, "metric_regex", None),
                        getattr(panel, "media_keys", None),
                    )
                    for panel in section.panels
                ],
            )
            for section in workspace.sections
        ]

    expected, actual = signature(desired), signature(restored)
    if actual != expected:
        raise RuntimeError("Saved W&B view did not preserve declared section layout")
    return url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    print(sync_workspace(args.entity, args.project))


if __name__ == "__main__":
    main()
