from __future__ import annotations

import os
from pathlib import Path

try:
    import modal
except ImportError as exc:  # pragma: no cover - only hit when Modal isn't installed locally
    raise RuntimeError("Install Modal first: pip install modal") from exc

REPO_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = "/root/gymemu"
REMOTE_CACHE_ROOT = "/mnt/cache"
REMOTE_MODELS_ROOT = "/mnt/models"

DEFAULT_GPU = os.environ.get("GYMEMU_MODAL_GPU", "A10G")
DEFAULT_TIMEOUT_S = int(os.environ.get("GYMEMU_MODAL_TIMEOUT_S", str(8 * 60 * 60)))
DEFAULT_CACHE_VOLUME = os.environ.get("GYMEMU_MODAL_CACHE_VOLUME", "gymemu-hf-cache")
DEFAULT_MODELS_VOLUME = os.environ.get("GYMEMU_MODAL_MODELS_VOLUME", "gymemu-models")

SOURCE_FILES = [
    "train.py",
    "dataset_utils.py",
    "device_utils.py",
    "game_config.py",
    "preprocessing.py",
    "rollout_dataset.py",
    "rollout_feedback.py",
    "spatial_model.py",
    "wandb_utils.py",
]

app = modal.App("gymemu-train")
cache_volume = modal.Volume.from_name(DEFAULT_CACHE_VOLUME, create_if_missing=True)
models_volume = modal.Volume.from_name(DEFAULT_MODELS_VOLUME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12").run_commands(
    "python -m pip install --upgrade pip",
    (
        "python -m pip install "
        "--extra-index-url https://download.pytorch.org/whl/cu118 "
        "torch numpy pillow python-dotenv huggingface_hub datasets tqdm wandb"
    ),
)
for relative_path in SOURCE_FILES:
    image = image.add_local_file(
        REPO_ROOT / relative_path,
        remote_path=f"{REMOTE_ROOT}/{relative_path}",
    )


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_S,
    volumes={
        REMOTE_CACHE_ROOT: cache_volume,
        REMOTE_MODELS_ROOT: models_volume,
    },
)
def train_remote(
    dataset: str = "tsilva/gymrec__BreakoutNoFrameskip_dash_v4_stack4_unroll8_train_ready",
    game: str = "breakout",
    history_length: int = 4,
    unroll_steps: int = 8,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    rollout_samples_per_epoch: int = 1024,
    feedback_mode: str = "soft",
    spatial_latent_channels: int = 32,
    spatial_refine_blocks: int = 4,
    output_subdir: str = "",
    wandb_mode: str = "disabled",
    model_compile: bool = True,
) -> dict[str, object]:
    import json
    import subprocess

    output_dir = REMOTE_MODELS_ROOT
    if output_subdir:
        output_dir = os.path.join(output_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(REMOTE_CACHE_ROOT, "hf"), exist_ok=True)
    os.makedirs(os.path.join(REMOTE_CACHE_ROOT, "datasets"), exist_ok=True)

    env = os.environ.copy()
    env["HF_HOME"] = os.path.join(REMOTE_CACHE_ROOT, "hf")
    env["HF_DATASETS_CACHE"] = os.path.join(REMOTE_CACHE_ROOT, "datasets")
    env["WANDB_MODE"] = wandb_mode

    checkpoint_path = os.path.join(
        output_dir,
        f"{dataset.replace('/', '__')}-spatial-dynamics.pt",
    )
    command = [
        "python",
        f"{REMOTE_ROOT}/train.py",
        "--dataset",
        dataset,
        "--game",
        game,
        "--history-length",
        str(history_length),
        "--unroll-steps",
        str(unroll_steps),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--rollout-samples-per-epoch",
        str(rollout_samples_per_epoch),
        "--feedback-mode",
        feedback_mode,
        "--spatial-latent-channels",
        str(spatial_latent_channels),
        "--spatial-refine-blocks",
        str(spatial_refine_blocks),
        "--output-dir",
        output_dir,
    ]
    command.append("--model-compile" if model_compile else "--no-model-compile")

    print("Running:", " ".join(command))
    subprocess.run(command, cwd=REMOTE_ROOT, env=env, check=True)

    models_volume.commit()
    cache_volume.commit()

    result = {
        "dataset": dataset,
        "checkpoint_path": checkpoint_path,
        "checkpoint_exists": os.path.exists(checkpoint_path),
        "models_volume": DEFAULT_MODELS_VOLUME,
        "cache_volume": DEFAULT_CACHE_VOLUME,
        "gpu": DEFAULT_GPU,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.local_entrypoint()
def main(
    dataset: str = "tsilva/gymrec__BreakoutNoFrameskip_dash_v4_stack4_unroll8_train_ready",
    game: str = "breakout",
    history_length: int = 4,
    unroll_steps: int = 8,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    rollout_samples_per_epoch: int = 1024,
    feedback_mode: str = "soft",
    spatial_latent_channels: int = 32,
    spatial_refine_blocks: int = 4,
    output_subdir: str = "",
    wandb_mode: str = "disabled",
    model_compile: bool = True,
) -> None:
    import json

    result = train_remote.remote(
        dataset=dataset,
        game=game,
        history_length=history_length,
        unroll_steps=unroll_steps,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        rollout_samples_per_epoch=rollout_samples_per_epoch,
        feedback_mode=feedback_mode,
        spatial_latent_channels=spatial_latent_channels,
        spatial_refine_blocks=spatial_refine_blocks,
        output_subdir=output_subdir,
        wandb_mode=wandb_mode,
        model_compile=model_compile,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
