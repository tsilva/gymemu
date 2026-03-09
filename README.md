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

## Build The Stacked Dataset

To create a filtered training dataset with 4 preprocessed history frames per sample,
run:

```bash
python scripts/build_stacked_dataset.py
```

The script:

- loads `tsilva/gymrec__BreakoutNoFrameskip_dash_v4`
- applies the same crop, black-background filter, and binarization used by runtime/training
- builds samples of `history[4] + action -> next_frame`
- removes exact duplicate `(history, action, next_frame)` tuples
- uploads the result to
  `tsilva/gymrec__BreakoutNoFrameskip_dash_v4_stack4_deduped`

The current full build produced:

- 108,001 source rows scanned
- 64,305 valid rows after background filtering
- 3,669 unique stacked transitions
- 60,620 exact duplicate stacked transitions removed

Generated samples have this schema:

- `episode_id`: source episode identifier as a hex string
- `source_index`: row index of the source action frame
- `history`: `uint8[4, 96, 80]`
- `action`: discrete Atari action id
- `next_frame`: `uint8[96, 80]`

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
  --early-stopping-patience 5 \
  --latent-dim 32 \
  --image-width 80 \
  --image-height 96
```

To train on the stacked, deduped dataset instead:

```bash
python train.py \
  --dataset tsilva/gymrec__BreakoutNoFrameskip_dash_v4_stack4_deduped \
  --game breakout \
  --history-length 4 \
  --epochs 50 \
  --early-stopping-patience 5 \
  --latent-dim 32 \
  --image-width 80 \
  --image-height 96
```

`train.py` now monitors validation loss for both phases, saves the best checkpoint, and
stops early after 5 non-improving epochs by default. Use
`--early-stopping-patience 0` to disable this.

For an M1 MacBook Pro with 16GB, the current defaults are tuned to be safer:

- training batch size defaults to `64`
- latent encoding batch size defaults to `64`
- DataLoader workers stay at `0` to avoid macOS multiprocessing memory duplication
- training and runtime both prefer `mps` over CPU when available

Output files are saved in `./models/` as:

- `tsilva__gymrec__BreakoutNoFrameskip_dash_v4-representation.pt`
- `tsilva__gymrec__BreakoutNoFrameskip_dash_v4-dynamics.pt`

### Pixel Dynamics Experiments

For direct frame prediction, train with `--dynamics-mode pixel`. The most useful
knobs during the Breakout experiments have been:

- `--pixel-unroll-steps`: number of autoregressive steps trained per sequence
- `--sequence-stride`: keep every `N`th rollout window to trade data for runtime
- `--pixel-dynamics-path`: resume from an existing pixel checkpoint
- `--learning-rate`: lower this when fine-tuning an existing pixel checkpoint
- `--pixel-feedback-mode`: `soft` for the current stable path, `ste` for binarized
  straight-through feedback experiments
- `--pixel-refine-blocks`: extra action-conditioned residual blocks that scale pixel-model
  capacity without changing the base checkpoint layout

Example long-horizon pixel run:

```bash
python train.py \
  --dataset tsilva/gymrec__BreakoutNoFrameskip_dash_v4 \
  --game breakout \
  --dynamics-mode pixel \
  --history-length 4 \
  --pixel-unroll-steps 8 \
  --pixel-refine-blocks 3 \
  --sequence-stride 16 \
  --learning-rate 0.0003 \
  --output-dir .cache/pixel_h4_unroll8_stride16 \
  --pixel-dynamics-path .cache/pixel_h4_unroll4_stride8/tsilva__gymrec__BreakoutNoFrameskip_dash_v4-pixel-dynamics.pt
```

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

For models trained on the stacked, deduped dataset, pass the matching dataset id and
history length:

```bash
python main.py \
  --dataset tsilva/gymrec__BreakoutNoFrameskip_dash_v4_stack4_deduped \
  --game breakout \
  --history-length 4 \
  --use-local-models \
  --models-dir ./models \
  --image-width 80 \
  --image-height 96
```

For direct pixel models, also pass the pixel dynamics mode. Use `--pixel-feedback soft`
for the current best checkpoints in this repo:

```bash
python main.py \
  --dataset tsilva/gymrec__BreakoutNoFrameskip_dash_v4 \
  --game breakout \
  --dynamics-mode pixel \
  --history-length 4 \
  --pixel-refine-blocks 3 \
  --pixel-feedback soft \
  --use-local-models \
  --models-dir ./.cache/pixel_h4_unroll8_stride16_refine3 \
  --image-width 80 \
  --image-height 96
```

Runtime now also defaults to a conservative `NOOP` stability hold for near-static
histories. That guard only applies when the action is `NOOP` and the predicted change is
tiny, which keeps idle states from drifting frame by frame.

For near-static `LEFT`/`RIGHT` states, runtime also applies a deterministic paddle shift in
the bottom playfield band instead of trusting the learned model. That keeps simple paddle
motion stable without changing the idle `NOOP` baseline.

For Breakout pixel playback, runtime now also tracks a small explicit ball state. When the
seed frame does not show a ball, it attaches one above the paddle; when the model predicts
scene changes that do not match the current stable playfield, runtime keeps the stable
background and advances just the ball. This preserves the current `NOOP` and paddle
stability fixes while making the ball visible during serve and launch states.

The Breakout pixel runtime now also treats paddle motion and brick hits explicitly:

- `LEFT` and `RIGHT` always shift the paddle deterministically, even while the ball is in
  flight.
- ball movement bounces off the first brick pixel it hits in the brick band and clears a
  small local patch, so the ball no longer phases through the wall of bricks while the
  rest of the stable-scene logic stays intact.

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
- `scripts/render_pixel_rollouts.py`: renders scripted pixel-model rollouts for quick visual checks
- `game_config.py`: per-game preprocessing/action metadata
- `preprocessing.py`: shared crop, validation, binarization, and action encoding
- `pixel_feedback.py`: shared helpers for soft, hard, and straight-through pixel feedback

## Notes

- CUDA is still the intended fast path for training and playback.
- Apple Silicon training/runtime use MPS automatically, with memory-oriented defaults for 16GB machines.
- The runtime predicts latent deltas, then advances state with `latent = latent + delta`.
- The corrupted magenta tail seen in the Breakout recording is filtered out during dataset ingestion.

## License

[MIT](LICENSE)
