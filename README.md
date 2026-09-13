<p align="center">
  <img src="logo.png" alt="gymemu" width="280" />
  <br />
  <strong>🎮 Train on recordings, play the predictions 🧠</strong>
</p>

<p align="center">
  <a href="https://github.com/tsilva/gymemu/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.11%E2%80%933.13-blue" alt="Python 3.11–3.13" /></a>
  <a href="https://github.com/tsilva/gymemu/blob/main/LICENSE"><img src="https://img.shields.io/github/license/tsilva/gymemu" alt="MIT license" /></a>
</p>

Gymemu is a Python tool for researchers experimenting with learned game emulators.
Train a neural network on recorded game frames and actions, then use the keyboard to
play inside its predictions. The default dataset contains Breakout trajectories.

The baseline predicts one RGB frame from a frame history and the current action using
a convolutional encoder/decoder and uniform pixel MSE. Playback feeds predictions back
into the model without loading a game engine or dataset.

## Install

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
git clone https://github.com/tsilva/gymemu.git
cd gymemu
uv sync --frozen
```

The locked environment uses Python 3.13. The manifest supports Python 3.11 through 3.13.

## Train and play

From the repository root:

```bash
uv run --frozen python train.py --output runs/breakout-001
uv run --frozen python play.py runs/breakout-001/best.pt
```

Training downloads the pinned
[Breakout dataset](https://huggingface.co/datasets/tsilva/gradlab-breakout-trajectories)
to the Hugging Face cache. Defaults are 8 history frames, batch size 32, and 10 epochs.
The device is selected in order of availability: CUDA, MPS, then CPU.
Use a new output directory for each run; nonempty directories are rejected.

By default, the play command opens a window. Press an action key once to generate the initial
frame. Each subsequent fresh press executes one action and predicts the next frame.
Holding a key does not advance the model.

| Key | Effect |
| --- | --- |
| Left arrow | Move left, action 2 |
| Right arrow | Move right, action 1 |
| Space | Button/serve, action 0 |
| R | Reset and wait for an action key |
| Escape | Quit |

For other integer action vocabularies, set bindings with `--key-action`, for example
`--key-action left=10 --key-action right=20 --key-action space=30`.

To start with visible bricks, paddle, and ball from a recorded scene, add
`--start-scene path/to/scene.npz` to the play command. The scene appears immediately;
the first fresh key press executes an action, and R restores the scene. All later
frames come from the model, with no corrections from recorded data.

Scene files are non-pickled NumPy archives containing `frames`: a uint8 array shaped
`[history, channels, height, width]`, ordered oldest to newest within one episode.
Use between one frame and the checkpoint's history length, at its original dimensions.
Short histories use left-zero padding. This start mode bypasses learned initialization.

## Commands

```bash
uv run --frozen python train.py --help  # training options
uv run --frozen python play.py --help   # playback options
uv run --frozen pytest -q               # pipeline tests
uv run --frozen ruff check .            # lint
```

For a bounded CPU smoke, use a fresh `runs/smoke` directory:

```bash
uv run --frozen python train.py --output runs/smoke --device cpu --threads 2 \
  --epochs 1 --batch-size 4 --limit-episodes 2 --train-batches 8 --eval-batches 2
uv run --frozen python play.py runs/smoke/best.pt --device cpu \
  --headless-actions 0,1,1,0,2 --output logs/smoke.png
```

Open `logs/smoke.png` to inspect the output. The first action value initializes the
frame; the remaining four advance it. This checks the pipeline, but the tiny training
budget does not produce a playable model. The smoke still downloads the dataset snapshot.

See the [training guide](docs/training.md) for CUDA settings, the dataset schema,
training defaults, and checkpoint details.

## Notes

- Runs write `config.json`, `metrics.jsonl`, and three inference checkpoints.
  `best.pt` has the lowest evaluation MSE, `last.pt` is the last completed epoch,
  and `latest.pt` is an atomic snapshot saved every 60 seconds and after training passes.
  Checkpoints do not support resuming optimizer progress and are incompatible with older
  gymemu experiments.
- Training uses recorded histories and keeps held-out episodes out of optimization.
  Playback uses generated histories, so errors can accumulate even with low evaluation MSE.
  Inspect ball motion, collisions, and brick persistence when comparing checkpoints.
- Training and playback share left-zero-padding and bootstrap rules. An empty history
  plus `START`, with no game action, predicts the initial frame. It cannot identify a
  reset seed and may average different starting images.
- Eight history frames are a practical baseline, not a guarantee of full observability.
  See the [history investigation](docs/history.md) for evidence and limits.
- Images keep their original dimensions and colors. The default dataset already masks
  the top 17 HUD rows. Each predicted transition follows its frame-skip-2 cadence.
- `--dataset` accepts a compatible Hub dataset or local snapshot directory.
  Generated data, runs, checkpoints, and diagnostics belong in ignored directories.

## Architecture

The diagram shows the default bootstrap path. Game thumbnails illustrate the data flow;
they are not measured model outputs.

![Training on recorded trajectories and playing through predicted frame feedback](architecture.png)

## License

[MIT](LICENSE)
