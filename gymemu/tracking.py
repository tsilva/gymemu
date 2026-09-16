"""Optional W&B logging for the shared runner, with one run across all stages."""

import json
import re
from contextlib import contextmanager

from gymemu.metrics import SCHEMA_VERSION, validate_metrics
from gymemu.training_state import TrainingInterrupted


def project_name(config):
    env_id = config["game"].get("env_id")
    if not isinstance(env_id, str) or not env_id.strip():
        raise ValueError("W&B logging requires game.env_id with the canonical environment ID")
    # W&B disallows these characters in project names. Preserve case and version.
    project = "gymemu-" + re.sub(r"[/\\#?%:]", "-", env_id.strip())
    if len(project) > 128:
        raise ValueError("W&B project name must be at most 128 characters")
    return project


def validate_tracking(config):
    mode = config.get("wandb", {}).get("mode", "disabled")
    if mode not in ("disabled", "offline", "online"):
        raise ValueError("wandb.mode must be disabled, offline, or online")
    if mode != "disabled":
        project_name(config)


class Tracker:
    def __init__(self, run=None):
        self.run = run

    def log_diagnostics(self, metrics, output, media_path=None):
        validate_metrics(metrics)
        # Keep the same scalar evidence when W&B is disabled.
        with (output / "diagnostics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        if self.run is not None:
            payload = dict(metrics)
            if media_path is not None and media_path.exists():
                import wandb

                payload["probe/frames"] = wandb.Image(
                    str(media_path),
                    caption=(
                        "Target above prediction. Rows are fixed starts; columns advance in time."
                    ),
                )
            self.run.log(payload)

    def log_epoch(self, record, *, optimizer_steps, train_samples_seen, learning_rate, final):
        if self.run is None:
            return
        prefix = f"stages/{record['stage']}"
        metrics = {
            "optimizer_steps": optimizer_steps,
            "train_samples_seen": train_samples_seen,
            "stage": record["stage"],
            "objective": record["objective"],
            f"{prefix}/epoch": record["epoch"],
            f"{prefix}/learning_rate": learning_rate,
        }
        for split in ("train", "validation"):
            result = record[split]
            metrics.update({f"{prefix}/{split}/{key}": value for key, value in result.items()})
            metrics[f"{prefix}/{split}/samples_per_second"] = result["samples"] / result["seconds"]
        metrics.update(
            {f"{prefix}/curriculum/{key}": value for key, value in record["curriculum"].items()}
        )
        metrics.update(
            {
                "train/step": optimizer_steps,
                "eval/step": optimizer_steps,
                f"train/{record['stage']}/loss/epoch": record["train"]["loss"],
                f"eval/{record['stage']}/loss": record["validation"]["loss"],
                "train/rate": record["train"]["samples"] / record["train"]["seconds"],
            }
        )
        if final:
            metrics["eval/mse"] = record["validation"]["mse"]
            metrics["evaluation/next_frame_rgb_mse"] = record["validation"]["mse"]
        self.run.log(metrics)

    def complete(self, summary):
        if self.run is not None:
            self.run.summary.update(summary)


@contextmanager
def track_run(config, output, *, evaluation, parameters, training_examples, recipe_sha256):
    settings = config.get("wandb", {})
    if settings.get("mode", "disabled") == "disabled":
        yield Tracker()
        return

    import wandb

    run = wandb.init(
        project=project_name(config),
        entity=settings.get("entity"),
        name=settings.get("name") or config["name"],
        group=settings.get("group"),
        tags=settings.get("tags", []),
        mode=settings["mode"],
        dir=str(output),
        config=config,
        reinit="create_new",
    )
    try:
        run.config.update(
            {
                "metrics_schema_version": SCHEMA_VERSION,
                "evaluation": evaluation,
                "parameters": parameters,
                "training_examples": training_examples,
                "recipe_sha256": recipe_sha256,
            }
        )
        run.define_metric("train/step")
        run.define_metric("eval/step")
        run.define_metric("train/*", step_metric="train/step", summary="last")
        run.define_metric("probe/*", step_metric="train/step", summary="last")
        run.define_metric("eval/*", step_metric="eval/step", summary="last")
        run.define_metric("optimizer_steps")
        run.define_metric("stages/*", step_metric="optimizer_steps")
        run.define_metric(
            "evaluation/next_frame_rgb_mse", step_metric="optimizer_steps", summary="min"
        )
        yield Tracker(run)
    except TrainingInterrupted:
        run.summary.update({"status": "interrupted", "resume_checkpoint": "resume.pt"})
        run.finish(exit_code=0)
        raise
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish(exit_code=0)
