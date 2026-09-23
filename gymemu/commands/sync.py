"""Publish a completed offline Run without changing its research identity."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from gymemu.storage import CheckpointStore
from gymemu.tracking import Tracker, project_name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args(argv)
    output = args.run.expanduser().resolve()
    run_path = output / "run.json"
    run = json.loads(run_path.read_text())
    if run["status"] not in ("complete", "publication_pending") or not (
        output / "summary.json"
    ).is_file():
        parser.error("Only completed local Runs can be synced")
    config = OmegaConf.to_container(OmegaConf.load(output / "resolved.yaml"), resolve=True)
    config.setdefault("r2", {})["enabled"] = True
    config["wandb"]["mode"] = "online"
    store = CheckpointStore(config, output)
    if store.state["run_id"] != run["id"]:
        raise ValueError("R2 receipt and local Run IDs differ")

    if not run.get("wandb_synced"):
        import wandb

        tracked = wandb.init(
            project=project_name(config),
            entity=config["wandb"].get("entity"),
            id=run["id"],
            resume="allow",
            dir=str(output),
            config=config,
        )
        try:
            tracker = Tracker(tracked)
            optimizer_steps = train_samples_seen = 0
            stages = {stage["name"]: stage for stage in config["approach"]["stages"]}
            for line in (output / "metrics.jsonl").read_text().splitlines():
                record = json.loads(line)
                optimizer_steps += record["train"]["batches"]
                train_samples_seen += record["train"]["samples"]
                tracker.log_epoch(
                    record,
                    optimizer_steps=optimizer_steps,
                    train_samples_seen=train_samples_seen,
                    learning_rate=stages[record["stage"]]["learning_rate"],
                    final=record["stage"] == run["stages"][-1],
                )
            diagnostics = output / "diagnostics.jsonl"
            if diagnostics.is_file():
                for line in diagnostics.read_text().splitlines():
                    tracked.log(json.loads(line))
            if hasattr(wandb, "Image"):
                for path in sorted((output / "diagnostics").glob("*.png"))[:12]:
                    tracked.log({"probe/frames": wandb.Image(str(path))})
            tracked.summary.update(json.loads((output / "summary.json").read_text()))
        except BaseException:
            tracked.finish(exit_code=1)
            raise
        else:
            tracked.finish(exit_code=0)
        run["wandb"] = {"id": run["id"], "url": getattr(tracked, "url", None)}
        run["wandb_synced"] = True
        run_path.write_text(json.dumps(run, indent=2) + "\n")
    prior_status = run["status"]
    prior_publication = run.get("publication")
    run["status"] = "complete"
    run["publication"] = "online"
    run_path.write_text(json.dumps(run, indent=2) + "\n")
    try:
        store.sync(final=True)
    except Exception:
        run["status"] = prior_status
        run["publication"] = prior_publication
        run_path.write_text(json.dumps(run, indent=2) + "\n")
        raise
    print(f"Published Run {run['id']}: {store.uri}")


if __name__ == "__main__":
    main()
