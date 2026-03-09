import os
from datetime import datetime

import numpy as np


def _sanitize_run_name(value: str) -> str:
    cleaned = value.replace("/", "__").replace(" ", "_")
    return "".join(ch for ch in cleaned if ch.isalnum() or ch in {"_", "-", "."})


class TrainingTracker:
    def __init__(self):
        self.active = False
        self.run = None
        self.wandb = None
        self.global_step = 0

    def init_run(self, args, derived_stats):
        mode = os.getenv("WANDB_MODE", "").strip().lower()
        api_key = os.getenv("WANDB_API_KEY", "").strip()

        if mode == "disabled":
            return
        if mode not in {"online", "offline"}:
            if api_key:
                mode = "online"
            else:
                return

        try:
            import wandb
        except ImportError:
            print("Warning: W&B requested but the 'wandb' package is not installed. Continuing.")
            return

        project = os.getenv("WANDB_PROJECT", "gymemu").strip() or "gymemu"
        entity = os.getenv("WANDB_ENTITY", "").strip() or None
        run_name = os.getenv("WANDB_RUN_NAME", "").strip()
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = _sanitize_run_name(
                f"{args.dataset}-{args.dynamics_mode}-{timestamp}"
            )
        tags_env = os.getenv("WANDB_TAGS", "").strip()
        tags = [tag.strip() for tag in tags_env.split(",") if tag.strip()]
        notes = os.getenv("WANDB_NOTES", "").strip() or None

        config = {key: value for key, value in vars(args).items()}
        config.update(derived_stats)

        try:
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                tags=tags,
                notes=notes,
                mode=mode,
                config=config,
            )
        except Exception as exc:
            print(f"Warning: Failed to initialize W&B ({exc}). Continuing without it.")
            self.run = None
            return

        self.active = True
        self.wandb = wandb
        print(f"W&B enabled: project={project}, mode={mode}, run={run_name}")

    def log_run_metrics(self, metrics):
        if not self.active:
            return
        self.run.log(_to_wandb_scalars(metrics))

    def log_batch(self, phase, metrics, step):
        if not self.active:
            return
        payload = {f"{phase}/{key}": value for key, value in metrics.items()}
        payload[f"{phase}/optimizer_step"] = step
        self.global_step += 1
        payload["run/global_step"] = self.global_step
        self.run.log(_to_wandb_scalars(payload))

    def log_epoch(self, phase, metrics, epoch):
        if not self.active:
            return
        payload = {f"{phase}/{key}": value for key, value in metrics.items()}
        payload[f"{phase}/epoch"] = epoch + 1
        self.global_step += 1
        payload["run/global_step"] = self.global_step
        self.run.log(_to_wandb_scalars(payload))

    def log_media(self, phase, media, epoch):
        if not self.active:
            return
        payload = {f"{phase}/{key}": value for key, value in media.items()}
        payload[f"{phase}/epoch"] = epoch + 1
        self.global_step += 1
        payload["run/global_step"] = self.global_step
        self.run.log(payload)

    def image(self, image, caption=None):
        if not self.active:
            return None
        return self.wandb.Image(image, caption=caption)

    def finish(self, summary):
        if not self.active:
            return
        for key, value in _to_wandb_scalars(summary).items():
            self.run.summary[key] = value
        self.run.finish()
        self.active = False


def _to_wandb_scalars(metrics):
    payload = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, np.generic):
            payload[key] = value.item()
            continue
        payload[key] = value
    return payload


def frame_to_rgb_uint8(frame):
    frame_array = np.asarray(frame)
    if frame_array.ndim == 3 and frame_array.shape[-1] == 3:
        clipped = np.clip(frame_array, 0.0, 1.0)
        return (clipped * 255.0).astype(np.uint8)
    if frame_array.ndim == 3 and frame_array.shape[0] == 1:
        frame_array = frame_array[0]
    clipped = np.clip(frame_array, 0.0, 1.0)
    grayscale = (clipped * 255.0).astype(np.uint8)
    return np.repeat(grayscale[..., None], 3, axis=2)


def make_image_grid(rows, pad=2, pad_value=32):
    rendered_rows = []
    for row in rows:
        images = [frame_to_rgb_uint8(image) for image in row]
        if not images:
            continue
        height = max(image.shape[0] for image in images)
        width = sum(image.shape[1] for image in images) + pad * max(0, len(images) - 1)
        canvas = np.full((height, width, 3), pad_value, dtype=np.uint8)
        offset = 0
        for image in images:
            canvas[: image.shape[0], offset : offset + image.shape[1]] = image
            offset += image.shape[1] + pad
        rendered_rows.append(canvas)

    if not rendered_rows:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    total_height = sum(row.shape[0] for row in rendered_rows) + pad * max(
        0, len(rendered_rows) - 1
    )
    total_width = max(row.shape[1] for row in rendered_rows)
    grid = np.full((total_height, total_width, 3), pad_value, dtype=np.uint8)

    offset = 0
    for row in rendered_rows:
        grid[offset : offset + row.shape[0], : row.shape[1]] = row
        offset += row.shape[0] + pad
    return grid
