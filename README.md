<div align="center">
  <img src="logo.png" alt="gymemu" width="512"/>

  # gymemu

  Neural emulator training and playback for retro games recorded with Gymnasium/gymrec.
</div>

## Breakout Status

This repo is now wired for the public dataset
`tsilva/gymrec__BreakoutNoFrameskip_dash_v4`.

The current Breakout pipeline applies these simplifications before training:

- Crop the scoreboard off the top of each frame.
- Keep only transitions whose playfield background is still black.
- Convert frames to monochrome with two values: black and white.
- Treat actions as Atari discrete actions with 4 classes: `NOOP`, `FIRE`, `RIGHT`, `LEFT`.

Those choices are implemented in shared preprocessing code so training and runtime use the same transform.

## Dataset Notes

The Hugging Face dataset currently contains:

- 108,001 frames
- 1 recorded episode
- 210x160 image observations
- `actions` stored as a one-element list containing the discrete action id

Because it is a single-episode dataset, `train.py` now does a temporal train/validation split within that episode instead of assigning the whole episode to validation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu118 -e .
```

On Apple Silicon, the code now prefers PyTorch `mps` automatically and uses more conservative default training batches to fit a 16GB unified-memory machine better.

If you plan to pull or publish model artifacts on Hugging Face:

```bash
cp .env.example .env
# add HF_TOKEN=...
```

## Train Breakout Models

The default training command is now Breakout-oriented:

```bash
python train.py \
  --dataset tsilva/gymrec__BreakoutNoFrameskip_dash_v4 \
  --game breakout \
  --epochs 50 \
  --latent-dim 32 \
  --image-width 80 \
  --image-height 96
```

For an M1 MacBook Pro with 16GB, the current defaults are tuned to be safer:

- training batch size defaults to `64`
- latent encoding batch size defaults to `64`
- DataLoader workers stay at `0` to avoid macOS multiprocessing memory duplication
- training and runtime both prefer `mps` over CPU when available

Output files are saved in `./models/` as:

- `tsilva__gymrec__BreakoutNoFrameskip_dash_v4-representation.pt`
- `tsilva__gymrec__BreakoutNoFrameskip_dash_v4-dynamics.pt`

## Run The Emulator

If you omit `--start-image`, the runtime fetches the first valid Breakout frame from the dataset and uses that as the initial latent state. You can still pass a local raw `210x160` Breakout frame if you want a specific start point.

```bash
python main.py \
  --dataset tsilva/gymrec__BreakoutNoFrameskip_dash_v4 \
  --game breakout \
  --use-local-models \
  --models-dir ./models \
  --image-width 80 \
  --image-height 96
```

Controls:

- `Space`: `FIRE`
- `Right Arrow`: `RIGHT`
- `Left Arrow`: `LEFT`
- no key: `NOOP`

If you omit `--use-local-models`, `main.py` downloads models from:

- `tsilva/gymrec__BreakoutNoFrameskip_dash_v4-representation`
- `tsilva/gymrec__BreakoutNoFrameskip_dash_v4-dynamics`

## Files

- `train.py`: trains the autoencoder and latent dynamics model
- `main.py`: runs the neural emulator in Pygame
- `game_config.py`: per-game preprocessing/action metadata
- `preprocessing.py`: shared crop, validation, binarization, and action encoding

## Notes

- CUDA is still the intended fast path for training and playback.
- Apple Silicon training/runtime use MPS automatically, with memory-oriented defaults for 16GB machines.
- The runtime predicts latent deltas, then advances state with `latent = latent + delta`.
- The corrupted magenta tail seen in the Breakout recording is filtered out during dataset ingestion.

## License

[MIT](LICENSE)
