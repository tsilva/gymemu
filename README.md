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

## Train

`train.py` is spatial-only. It expects a raw observation dataset so it can build
autoregressive rollout windows internally.

Example:

```bash
python train.py \
  --dataset tsilva/gymrec__BreakoutNoFrameskip_dash_v4 \
  --game breakout \
  --history-length 4 \
  --unroll-steps 8 \
  --sequence-stride 16 \
  --epochs 50 \
  --early-stopping-patience 5 \
  --spatial-latent-channels 32 \
  --spatial-refine-blocks 4 \
  --image-width 80 \
  --image-height 96
```

Useful knobs:

- `--feedback-mode`: `soft`, `hard`, or `ste`
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
```

You can also let runtime download the model from Hugging Face by omitting
`--use-local-models`.

Controls:

- `Space`: `FIRE`
- `Right Arrow`: `RIGHT`
- `Left Arrow`: `LEFT`
- no key: `NOOP`
- `Escape`: quit

## Files

- `train.py`: trains the spatial latent world model
- `main.py`: runs the spatial latent emulator in Pygame
- `spatial_model.py`: model definition and checkpoint compatibility loader
- `preprocessing.py`: shared crop, validation, binarization, and action encoding
- `rollout_feedback.py`: feedback modes for autoregressive rollout
- `game_config.py`: game-specific preprocessing and controls

## Notes

- CUDA is still the intended fast path, but Apple Silicon uses `mps` automatically.
- Current defaults are tuned around Breakout and 16GB Apple Silicon machines.
- Legacy latent-delta and direct pixel-model codepaths have been removed from this repo.

## License

[MIT](LICENSE)
