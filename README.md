<p align="center">
  <img src="logo.png" alt="gymemu" width="280" />
  <br />
  <strong>🎮 Train on recordings, play the predictions 🧠</strong>
</p>

<p align="center">
  <a href="https://github.com/tsilva/gymemu/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.11%E2%80%933.13-blue" alt="Python 3.11–3.13" /></a>
  <a href="https://github.com/tsilva/gymemu/blob/main/LICENSE"><img src="https://img.shields.io/github/license/tsilva/gymemu" alt="MIT license" /></a>
</p>

Gymemu is a Python toolkit for comparing learned game emulators. Train different models
or multi-stage pipelines on recorded game frames and actions, then play their predictions.
Hydra config files control the game, models, training stages, and experiment settings.

The default approach is the original direct RGB CNN. An experimental latent approach
first trains a frame autoencoder, freezes it, then trains a separate latent predictor.
Both use the same dataset adapter, player, and held-out RGB evaluation metric.

`recipe=breakout_ball` adds frame-aligned `ball_x_normalized` and
`ball_y_normalized` history and predicts the next coordinates alongside RGB. Its
coordinate loss trains the shared encoder, and predicted coordinates condition the
image decoder. See [the ball-coordinate recipe](docs/recipes.md#ball-coordinate-experiment).

## Install

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
git clone https://github.com/tsilva/gymemu.git
cd gymemu
uv sync --frozen
uv tool install . --editable --exclude-newer "7 days"
gymemu --help
```

The editable install adds `gymemu` to your shell. Use `gymemu train`, `gymemu play`,
and `gymemu compare` from any directory. Use absolute dataset/checkpoint paths when
working elsewhere. `uv tool install . --exclude-newer "7 days"` installs a self-contained copy.
The original `python train.py` and other repository scripts remain supported.

The locked environment uses Python 3.13. Python 3.11 through 3.13 is supported.

Training uses W&B and R2 by default. Run `uv run wandb login` and configure the
[R2 credentials](docs/training.md#r2-checkpoint-storage) before starting a run.
For local-only training, pass `wandb.mode=disabled r2.enabled=false`.

## Train and play

From the repository root:

```bash
# Default direct CNN on Breakout
uv run gymemu train output=runs/breakout-direct

# Frame autoencoder, then a separate latent prediction model
uv run gymemu train approach=latent output=runs/breakout-latent

# Play either approach using its checkpoint
uv run gymemu play runs/breakout-direct/best.pt
```

Use a new output directory for each run. Omit `output` for a timestamped directory
under `runs/`. Training downloads the pinned
[Breakout dataset](https://huggingface.co/datasets/tsilva/gradlab-breakout-trajectories)
and writes a recorded `start-scene.npz` beside the checkpoints.

On Apple Silicon, `play.py` can use MPS, including existing ball-position checkpoints.
Pass `--device mps` to select it explicitly, or `--device cpu` for CPU playback.

Training logs to Weights & Biases by default. Authenticate once before training:

```bash
uv run wandb login
uv run gymemu train output=runs/breakout-tracked
```

Projects use `gymemu-<canonical-env-id>`, so Breakout logs to
`gymemu-Breakout-Atari2600-v0`. Each run records stage losses, held-out RGB MSE,
throughput, learning rates, curriculum values, configuration, and the final summary.
Use `wandb.mode=offline` to collect logs locally or `wandb.mode=disabled` to turn tracking off.
See [tracking options](docs/training.md#weights--biases) for teams and custom environments.

Checkpoints also upload to the separate `gymemu` R2 bucket, together with the
recorded start scene, metrics, and reproduction files. Each run has a unique prefix;
immutable objects and manifests retain successfully uploaded checkpoint versions.
Use `r2.enabled=false` to keep artifacts local. Retry interrupted uploads with
`uv run gymemu upload-checkpoints runs/my-run`.

The player opens a local browser dashboard with that scene in step mode. Every fresh action key press
predicts one next frame. Press Tab to toggle continuous play at 30 predictions per
second, matching 60 Hz Atari with frameskip 2. Hold an action key to repeat it;
releasing all keys uses action 0, or the first checkpoint action if 0 is absent.
The most recently pressed held key wins. Slow inference reduces playback speed
without catch-up steps. R restores the scene and pauses; Escape pauses. Breakout uses Left, Right, and Space. Other games use checkpoint key bindings,
or numbered keys for their action vocabulary. The player prints its bindings.

The Input history widget shows every RGB history frame, oldest to newest,
left to right and then top to bottom, with frame numbers overlaid at each tile's
top-left corner. The compact grid has no bottom caption. Black frames retain the model's zero padding.
After inference it shows the exact stack used for the displayed prediction; before
the first prediction and after reset it shows the stack ready for the next step.
Closing or leaving the browser pauses playback. Ctrl+C in the terminal stops the server.

The player opens two synchronized browser tabs using Gradlab's paired workspace
approach. The first contains Original, Prediction, Prediction − original, and the
playbar. The Diagnostics tab contains Input history, Prediction error, Model context,
and custom metric widgets. Its default layout places Input history across the top,
with Prediction error below on the left and Model context on the right.
Both display snapshots from one inference session.
The header's Diagnostics/Player link reopens or focuses the companion tab.

The dashboard uses Gradlab's widget structure and theme, with a compact single-row
topbar and icon buttons for Panels, Add widget, and Reset layout. Comparison panels
automatically share the available space above the playbar with equal canvas sizes;
unused space around original, predicted, difference, and history frames matches the
widget background, keeping black image pixels distinct from display padding;
on narrow screens they stack vertically. In Diagnostics, drag a widget's grip to move
it or resize from its corner. Panel menus can hide or disable widgets. Panels restores
hidden widgets in the current tab. Add widget in Diagnostics creates editable metric
cards or RGB MSE charts. Layout changes synchronize between tabs and persist across
launches; Reset layout restores both tabs. Existing layouts migrate automatically.

Use the mode selector to switch between autoregressive play and teacher forcing.
Open the bottom bar's gear for playback settings and episode or starting-scene
selection. In teacher forcing, the bottom
slider seeks directly to a recorded target without replaying intervening predictions.
The bottom bar follows Gradlab's player styling: a purple episode/step label and
scrubber, purple play button (amber while playing), coral reset, and cyan settings.
The gear contains play/pause, reset, step navigation, and action controls. Playback
controls live in settings rather than a dashboard widget, including in saved layouts.
All three frame panels reserve the same footer height to keep the images aligned.
The difference footer shows current RGB MSE, independent of display gain. Diagnostics
charts every measured transition in the current episode: hover for a vertical cursor
and value, click to select that step in the player, drag to zoom, or double-click to
reset zoom. The playbar shows the shared zoom range with adjustable handles. While zoomed,
Diagnostics also shows the segment selector beneath the charts. Drag either bracket
or use its arrow keys to resize the range without moving the playback cursor.
Highest MSE jumps to the largest measured error. Seeking preserves measurements;
resetting, changing episode, or switching mode clears them. Unvisited steps are
unscored and gaps are left visible.
Original and Prediction have blank footers; the difference footer keeps its gain
control and legend. `--no-browser` prints both URLs without opening tabs;
`--port auto` chooses an unused port by default. Diagnostics observes playback and
does not send keyboard, heartbeat, or pause commands. Leaving the player tab pauses
playback; returning to either tab shows the current server snapshot.

To inspect one-step predictions without accumulated feedback errors, replay recorded
episodes with teacher forcing:

```bash
uv run gymemu play runs/breakout-direct/best.pt --teacher-forcing
```

The dashboard starts with Original, Prediction, and Prediction − original widgets
from left to right. The signed RGB difference uses gray for zero,
brighter channels for positive differences, and darker channels for negative ones.
Every step uses recorded RGB history, executed actions, and any required auxiliary
state. Space steps, Tab plays or pauses, R restarts the episode, and C selects the
next episode. Replay pauses at episode end. The timeline shows the frame position; the metrics widgets show
recorded action and float32 RGB MSE. The Input history widget shows the recorded inputs.
Replay starts with the real initial frame and predicts transitions only.

The dataset and pinned revision come from the checkpoint, with its evaluation split
selected by default. Use `--episode-id 5` to select an episode, `--split train` to
inspect training examples, or `--dataset /path/to/snapshot` to use a local copy.
See [teacher-forced replay](docs/training.md#teacher-forced-replay) for diagnostics
and headless output.

Use `--start-scene path/to/scene.npz` to select another scene, `--key-action left=10`
to set a binding, or `--empty-start` to test learned initialization. In empty-start
mode the first press generates the initial frame without executing a game action.
A missing recorded scene produces an error instead of silently switching modes.

For repeatable debugging, select a named frame/action snapshot:

```bash
uv run gymemu play runs/breakout-direct/best.pt --list-start-states
uv run gymemu play runs/breakout-direct/best.pt --start-state ball-up
```

Press R to reset the current state, or C to reset to the next named snapshot.
No flag is needed. The cycle begins with your selected start, then visits the
other compatible states in alphabetical order and wraps around. `--state-dir`
selects the library; the timeline shows the current state. Each reset restores
RGB, action, and any auxiliary state histories without inference. Incompatible
library snapshots are skipped with a message. With no other compatible states,
C resets the current state too.

Available Breakout starts are shown below. Each image is the last recorded frame of
its eight-frame snapshot. Use the name with `--start-state`.

| `ball-up` | `near-bricks` | `paddle-approach` |
| :---: | :---: | :---: |
| ![Ball moving up toward the bricks](start_states/breakout/ball-up.png) | ![Ball moving up near the brick wall](start_states/breakout/near-bricks.png) | ![Ball descending toward the paddle](start_states/breakout/paddle-approach.png) |
| Moving up after a paddle bounce | Approaching the brick wall | Descending toward paddle contact |

| `half-cleared` | `almost-cleared` | `above-bricks` |
| :---: | :---: | :---: |
| ![Partly cleared wall with 50 bricks remaining](start_states/breakout/half-cleared.png) | ![Almost cleared wall with eight bricks remaining](start_states/breakout/almost-cleared.png) | ![Ball above the wall after tunneling through its left side](start_states/breakout/above-bricks.png) |
| 50 bricks remain | Eight bricks remain | Ball above the wall; 86 bricks remain |

R restores the selected snapshot. These starts are shared across compatible
checkpoints so you can compare the same situation. See [the snapshot library](start_states/README.md)
for provenance and how to save more states with `save_start_state.py`.

## Saved recipes

For Docker image publishing, Beast-3 scheduling, and Runpod setup, see
[container training](containers/train/README.md).

The successful ten-epoch Breakout CNN run is saved as `recipe=breakout_cnn`.
`recipe=breakout_actions` adds seven previous executed actions alongside the current
action and eight RGB frames, using the same dataset, training budget, and MSE objective.
Generic `recipe=direct` and `recipe=latent` presets cover the original approaches.
`recipe=breakout_ball_region` keeps the action-history model and recorded training
histories, adding a separately normalized RGB loss around a detected target ball.
It uses a four-pixel margin and weight `0.3`, and logs detection coverage. See the
[ball-region experiment](docs/recipes.md#ball-region-loss-experiment) for the control
run, tuning, and detector limitations.
`recipe=breakout_scheduled` tests recovery from generated context using the action-history
CNN. It starts with two epochs of recorded context, then increases prediction feedback
to 80% by epoch 8. The [recipe guide](docs/recipes.md#scheduled-frame-feedback) explains
the sampling, compute cost, and tuning controls.

`recipe=breakout_scheduled_fast` accelerates the same curriculum with factored action
inputs, bfloat16 feedback buffers, and selective prediction at low feedback rates.
See the [matched Beast-3 benchmarks](docs/performance.md#scheduled-feedback-optimization)
for measured gains and reproduction commands. Existing recipes remain available as controls.

```bash
# Inspect a recipe without downloading data or starting training
uv run gymemu train recipe=breakout_cnn --cfg job --resolve

# Train on a CUDA host after building the frame cache described below
uv run gymemu train recipe=breakout_cnn output=runs/breakout-recipe

# Try the action-history variant with the same cache
uv run gymemu train recipe=breakout_actions output=runs/breakout-actions

# Replay the settings captured by a previous run
uv run gymemu train --recipe runs/breakout-recipe/recipe.yaml output=runs/replay
```

Every new run saves a standalone `recipe.yaml`, a source archive, and a code/environment
receipt. Saved recipes pin the actual dataset revision or content fingerprint and preserve
internal tuning links, such as model dimensions and stage learning rates. See the
[recipe guide](docs/recipes.md) for inheritance, cache setup, overrides, and reproduction limits.

## Configure and compare

Configs compose in layers: `game`, `model`, `approach`, `trainer`, `optimizer`, then
optional `recipe` and `experiment` presets. Command-line overrides take precedence.

```bash
# Inspect the complete configuration without loading data
uv run gymemu train approach=latent --cfg job --resolve

# Inherit a smaller CNN config and override training settings
uv run gymemu train model=direct_small history=4 trainer.epochs=5

# Reuse CUDA settings for either approach
uv run gymemu train approach=latent experiment=cuda

# Tune one stage independently
uv run gymemu train approach=latent approach.stages.0.epochs=5 approach.stages.1.epochs=20

# Sweep architectures and seeds; Hydra creates a separate directory per run
uv run gymemu train --multirun approach=direct,latent seed=47,48 hydra.sweep.dir=runs/comparison

# Rank runs with matching evaluation targets and export their budgets and metrics
uv run gymemu compare runs/comparison --csv logs/comparison.csv
```

The latent approach's `model` group configures its codec. Set
`approach.models.dynamics.width=128` to tune its predictor. See the
[training guide](docs/training.md) for config inheritance and artifacts, and
[approach guide](docs/approaches.md) for adding models or pipelines.

## Faster CUDA training

For large datasets, build a lossless frame cache once and reuse it across runs.
The cache preserves every RGB pixel and verifies source and cache checksums.
Use the same immutable dataset revision for caching and training.

```bash
uv run gymemu cache-frames \
  --dataset tsilva/gradlab-breakout-trajectories \
  --revision 676ff6388f4218d3c3a3ce9f2f33e075fa7314a3 \
  --output data/breakout-676ff638-lz4

uv run gymemu train experiment=cuda_cached \
  game.revision=676ff6388f4218d3c3a3ce9f2f33e075fa7314a3 \
  trainer.frame_cache=data/breakout-676ff638-lz4 trainer.epochs=10 \
  output=runs/breakout-cached
```

This preset uses two loading threads, compiled computation, and overlapping CUDA
transfers. It retains the direct CNN, full RGB frames, eight-frame history, and pixel
MSE. Compilation adds startup time; checkpoints also load on CPUs without compilation
or a frame cache. See [performance measurements](docs/performance.md) for results,
reproduction commands, and the synchronization tradeoff.

## Other games

The training path is game-independent. Supply a dataset with the supported RGB-frame,
episode, and scalar integer action schema:

```bash
uv run gymemu train game=custom game.name=my-game game.dataset=/absolute/path/to/snapshot
```

Hub IDs work too. Add a YAML file under `configs/game/` to save a dataset ID, immutable
revision, splits, key bindings, and recorded start selection. Images retain their
original dimensions and colors. Other dataset layouts, continuous actions, and compound
actions require an adapter; a dataset name alone does not establish compatibility.

## Checks

```bash
uv run pytest
uv run ruff check .

# Bounded training; still downloads the dataset snapshot
uv run gymemu train experiment=smoke wandb.mode=disabled r2.enabled=false output=runs/smoke
uv run gymemu play runs/smoke/best.pt --device cpu --headless-actions 0,1,2 --output logs/smoke.png
```

Open `logs/smoke.png` to inspect the result. A smoke checks the pipeline; its training
budget does not produce a playable model. Add `approach=latent` to exercise both stages.

## Notes

- Every run saves resolved config, dataset provenance, stage metrics, all inference
  weights, and a comparison summary. Checkpoints support playback, not optimizer resume.
  Existing version-1 direct CNN checkpoints still load.
- Representation fitting and prediction fitting use training episodes only. All
  approaches are evaluated in float32 using held-out next-frame RGB MSE. The comparison
  command separates different datasets and target sets, and reports training budgets.
- Playback feeds predictions back into the model, so errors can accumulate even with
  low evaluation MSE. Check ball motion, collisions, and brick persistence in rollouts.
- Eight history frames are a practical baseline, not a guarantee of full observability.
  See the [history investigation](docs/history.md). Empty-history initialization can
  average distinct recorded reset states.

## Architecture

The image shows the direct baseline. Alternative approaches replace its model and
training stages while sharing data alignment, checkpoint loading, and playback.
Game thumbnails illustrate the flow; they are not measured outputs.

![Direct CNN training and playback with Breakout frames](architecture.png)

## License

[MIT](LICENSE)

`recipe=breakout_scheduled_ball_region` combines generated history with the 0.03
ball loss for two epochs, using feedback probabilities 0.4 and 0.8. See the
[recipe details](docs/recipes.md#ball-loss-with-prediction-feedback).
