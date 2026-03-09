# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this
repository.

## Project Overview

Gymnasium Emulator is now centered on a single architecture: a spatial latent world
model for retro games recorded with Gymnasium/gymrec.

The system uses:

- **SpatialLatentWorldModel**: consumes a short history of preprocessed frames plus a
  one-hot action, predicts the next frame as a warped copy of the last frame plus a
  learned residual
- **Shared preprocessing**: crop scoreboard, filter invalid backgrounds, resize, and
  binarize frames identically in training and runtime
- **Pygame interface**: real-time visualization and keyboard control

Models can be loaded from local checkpoints or downloaded from Hugging Face at runtime.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu118 -e .
```

Optional Hugging Face setup:

```bash
cp .env.example .env
# add HF_TOKEN=...
```

## Running The Emulator

```bash
python main.py \
  --dataset tsilva/gymrec__BreakoutNoFrameskip_dash_v4 \
  --game breakout \
  --history-length 4
```

Controls:

- `Space`: `FIRE`
- `Right Arrow`: `RIGHT`
- `Left Arrow`: `LEFT`
- `Escape`: quit

## Code Architecture

### SpatialLatentWorldModel

Defined in `spatial_model.py`.

- Encodes `history` and temporal differences into a spatial latent map
- Applies action-conditioned residual blocks in local and downsampled context paths
- Predicts the next latent state as a residual update
- Decodes via a residual head plus a learned optical-flow-style warp of the last frame

### Runtime Loop

`main.py`:

1. builds the initial frame history from `--start-image` or the dataset
2. maps keyboard input to a 4-class Breakout action
3. runs the spatial model on `history + action`
4. feeds the predicted frame back into history
5. renders at 30 FPS

### Training

`train.py`:

- builds rollout windows directly from raw observation datasets
- trains the spatial world model autoregressively across `--unroll-steps`
- supports `soft`, `hard`, and `ste` rollout feedback
- saves the best checkpoint using validation loss and early stopping

## Important Instructions

- **README.md must be kept up to date** with any significant project changes
- Preserve compatibility between `train.py`, `main.py`, and `spatial_model.py`
- Prefer the spatial model path; latent/pixel legacy paths are intentionally removed
