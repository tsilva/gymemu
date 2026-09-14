"""Optional W&B logging for the shared runner, with one run across all stages."""

import re
from contextlib import contextmanager


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
        if final:
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
                "evaluation": evaluation,
                "parameters": parameters,
                "training_examples": training_examples,
                "recipe_sha256": recipe_sha256,
            }
        )
        run.define_metric("optimizer_steps")
        run.define_metric("stages/*", step_metric="optimizer_steps")
        run.define_metric(
            "evaluation/next_frame_rgb_mse", step_metric="optimizer_steps", summary="min"
        )
        yield Tracker(run)
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish(exit_code=0)
