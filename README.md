<div align="center">
  <img src="logo.png" alt="gymemu" width="512"/>

  # gymemu

  Spatial latent world-model training and playback for Gymnasium retro recordings.
</div>

## Overview

This repository now contains a single model family: the spatial latent world model in
`spatial_model.py`.

For Breakout, the pipeline:

- crops off the scoreboard
- filters for frames whose playfield background is still black
- binarizes the playfield to black/white
- uses 4 Atari actions: `NOOP`, `FIRE`, `RIGHT`, `LEFT`

Training and runtime share the same preprocessing so rollouts use the exact same frame
format the model was trained on.

Training can also corrupt the seed history with mild noise and foreground dropout so
the model learns to recover from slightly imperfect context during autoregressive
rollouts. This corruption is training-only; runtime inference is unchanged.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu118 -e .
```

If you want to download or publish checkpoints on Hugging Face:

```bash
cp .env.example .env
# add HF_TOKEN=...
```

`train.py` also reads `.env` for optional Weights & Biases logging:

- `WANDB_MODE=online`, `offline`, or `disabled`
- `WANDB_PROJECT=gymemu`
- optional `WANDB_ENTITY`, `WANDB_API_KEY`, `WANDB_RUN_NAME`, `WANDB_TAGS`, `WANDB_NOTES`

## Prepare Train-Ready Dataset

If you want to precompute the rollout windows used by training and publish them as a
new dataset, build a prepared dataset with `train` and `validation` splits:

```bash
python scripts/build_training_dataset.py \
  --source-dataset tsilva/gymrec__BreakoutNoFrameskip_dash_v4 \
  --target-dataset tsilva/gymrec__BreakoutNoFrameskip_dash_v4_stack4_unroll16_train_ready \
  --history-length 4 \
  --unroll-steps 16 \
  --sequence-stride 16
```

This applies the same filtering and preprocessing expected by `train.py`, then stores:

- `history`: stacked preprocessed input frames
- `action_seq`: one-hot actions for each rollout step
- `target_frames`: the target rollout frames for autoregressive training

Useful flags:

- `--skip-upload`: build and validate the dataset locally without pushing
- `--max-rows`: smoke-test the pipeline on only the first N raw rows
- `--private`: create the target dataset as private on Hugging Face

## Train On Modal

If you want CUDA without running training locally, use `modal_train.py`.

Install Modal locally first:

```bash
pip install modal
modal setup
```

Run a remote training job:

```bash
modal run modal_train.py \
  --dataset tsilva/gymrec__BreakoutNoFrameskip_dash_v4_stack4_unroll16_train_ready \
  --epochs 50 \
  --batch-size 16
```

The Modal app:

- installs the training dependencies in a Linux image with CUDA PyTorch wheels
- runs `train.py` on a GPU worker
- persists Hugging Face cache data in the `gymemu-hf-cache` volume
- persists checkpoints in the `gymemu-models` volume

Useful environment overrides before `modal run`:

- `GYMEMU_MODAL_GPU=L40S` to change the GPU class from the default `A10G`
- `GYMEMU_MODAL_TIMEOUT_S=43200` to increase the job timeout
- `GYMEMU_MODAL_CACHE_VOLUME=...` and `GYMEMU_MODAL_MODELS_VOLUME=...` to change volume names

## Inspect Datasets

To visually confirm that a deduped stacked dataset still contains the gameplay you
recorded, render a contact sheet plus a source-episode coverage view:

```bash
python scripts/inspect_dataset.py \
  --dataset tsilva/gymrec__BreakoutNoFrameskip_dash_v4_stack4_deduped \
  --source-dataset tsilva/gymrec__BreakoutNoFrameskip_dash_v4 \
  --emit-gifs \
  --num-samples 12 \
  --output-dir .cache/dataset_inspection
```

This writes:

- `.cache/dataset_inspection/samples.png`: evenly spaced `(history -> next)` strips from
  the inspected dataset, with `source_index` and action labels
- `.cache/dataset_inspection/gifs/*.gif`: one short animation per sampled transition,
  showing the history frames followed by the target next frame
- `.cache/dataset_inspection/coverage.png`: a timeline view of the source episode showing
  which rows were valid after preprocessing, which rows were kept in the deduped dataset,
  and which valid rows were dropped

The script also works on raw observation datasets. In that case it builds transition
windows with the same preprocessing used by training/runtime before rendering the same
spot-check images.

## Train

`train.py` is spatial-only. It only trains from a prepared rollout dataset created by
`scripts/build_training_dataset.py`.

Example:

```bash
python train.py \
  --dataset tsilva/gymrec__BreakoutNoFrameskip_dash_v4_stack4_unroll16_train_ready \
  --game breakout \
  --history-length 4 \
  --unroll-steps 16 \
  --epochs 50 \
  --early-stopping-patience 5 \
  --spatial-latent-channels 32 \
  --spatial-refine-blocks 4 \
  --image-width 80 \
  --image-height 96
```

Useful knobs:

- `--feedback-mode`: `soft`, `hard`, or `ste`
- `--history-corruption` / `--no-history-corruption`: enable or disable seed-history corruption
- `--history-corruption-max-strength`: cap the per-sequence Gaussian noise scale
- `--history-corruption-foreground-dropout-max`: cap foreground pixel dropout probability
- `--spatial-dynamics-path`: resume from an existing spatial checkpoint
- `--rollout-samples-per-epoch`: cap weighted rollout sampling per epoch
- `--model-compile`: enable or disable `torch.compile()` on CUDA

Checkpoints are saved as:

- `./models/<dataset-name>-spatial-dynamics.pt`

## Run

`main.py` is spatial-only. If you omit `--start-image`, runtime pulls the first valid
Breakout frame history from the dataset.

```bash
python main.py \
  --dataset tsilva/gymrec__BreakoutNoFrameskip_dash_v4 \
  --game breakout \
  --history-length 4 \
  --use-local-models \
  --models-dir ./models \
  --image-width 80 \
  --image-height 96

# or point directly at a checkpoint without renaming it
python main.py \
  --dataset tsilva/gymrec__BreakoutNoFrameskip_dash_v4 \
  --game breakout \
  --history-length 4 \
  --spatial-dynamics-path ./models/modal_real_20260310/tsilva__gymrec__BreakoutNoFrameskip_dash_v4_stack4_unroll8_train_ready-spatial-dynamics.pt \
  --image-width 80 \
  --image-height 96

# or only advance when an action key is pressed
python main.py \
  --dataset tsilva/gymrec__BreakoutNoFrameskip_dash_v4 \
  --game breakout \
  --history-length 4 \
  --runtime-mode on-action
```

You can also let runtime download the model from Hugging Face by omitting
`--use-local-models`.

Controls:

- `Space`: `FIRE`
- `Right Arrow`: `RIGHT`
- `Left Arrow`: `LEFT`
- no key: `NOOP`
- `Escape`: quit

`--runtime-mode realtime` is the default and keeps generating frames at 30 FPS using
the current held key state. `--runtime-mode on-action` waits until you press one of the
bound action keys, then advances exactly one model step for that action. Idle time does
not generate `NOOP` frames in `on-action` mode.

## Files

- `train.py`: trains the spatial latent world model from a prepared rollout dataset
- `main.py`: runs the spatial latent emulator in Pygame
- `modal_train.py`: launches CUDA training on Modal with persistent cache/model volumes
- `spatial_model.py`: model definition and checkpoint compatibility loader
- `preprocessing.py`: shared crop, validation, binarization, and action encoding
- `rollout_dataset.py`: shared rollout-window building and prepared-dataset helpers
- `rollout_feedback.py`: feedback modes for autoregressive rollout
- `game_config.py`: game-specific preprocessing and controls
- `scripts/build_training_dataset.py`: builds and uploads train-ready rollout datasets
- `scripts/inspect_dataset.py`: renders frame strips and source-coverage images for raw or
  deduped datasets

## Notes

- CUDA is still the intended fast path, but Apple Silicon uses `mps` automatically.
- Current defaults are tuned around Breakout and 16GB Apple Silicon machines.
- Legacy latent-delta and direct pixel-model codepaths have been removed from this repo.

## License

[MIT](LICENSE)
