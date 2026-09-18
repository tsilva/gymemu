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

`gymemu dynamics` runs a separate state-only, single-ball experiment. It predicts
normalized ball/paddle state, brick occupancy, and termination without loading RGB
or modifying the dataset. A loss ends the simulation; serving is outside its
contract. It includes MLP/GRU models, independent state/action-history searches,
recursive state training, and headless playback. See
[state dynamics](docs/training.md#single-ball-state-dynamics).
New state models omit paddle velocity from their inputs, prediction heads, and
losses; original dataset columns and older checkpoints remain supported.
The first 48-run search and its collision, brick-removal, and termination failures
are documented in [the experiment results](docs/history.md#2026-09-17-single-ball-state-dynamics).
Use [isolated state probes](docs/training.md#isolated-state-targets-and-input-sufficiency)
to audit per-variable input ambiguity and separate target losses before joint training.
The probes also report event-specific errors and constant-motion baselines, with
optional event-balanced training samples.
[Internal-state input experiments](docs/training.md#reconstructed-internal-state-inputs)
compare reconstructed controller memory and paddle-hit counts with matched controls.
A [compact paddle predictor](docs/training.md#compact-paddle-learning-with-sufficient-inputs)
learns next position from current controller state, with a separate final-test evaluation.
Its controller inputs can be reduced to charge for the current position-prediction
experiment; [the ablation results](docs/history.md#2026-09-18-minimum-controller-inputs-for-paddle-position)
separate one-step accuracy from the state needed for repeated controller updates.
A [charge predictor](docs/training.md#learning-the-paddle-charge-update) now learns
the other part of `[paddle x, charge] + action`, with recorded validation and
recursive paddle-only results.
[Horizontal-velocity probes](docs/training.md#discrete-horizontal-velocity-probes)
compare continuous and discrete outputs using controller data and collision memory
reconstructed in RAM, without adding dataset columns.
The [focused paddle-region experiment](docs/training.md#focused-paddle-region-velocity-probes)
tests a smaller input contract and separates bounce-direction errors from speed errors.
[Direction-only probes](docs/training.md#isolating-horizontal-direction) compare
two-class direction learning with the eight-class velocity objective on those same inputs.
The [coverage audit](docs/training.md#auditing-direction-error-coverage) compares
remaining errors with training support and native bounce-direction boundaries.
The [direction learning curve](docs/training.md#direction-learning-with-more-training-episodes)
compares 128, 256, and 512 training episodes with fixed validation and update budgets.
The [full-data geometry experiment](docs/training.md#full-data-direction-geometry-experiments)
reaches 99.94% paddle-direction accuracy on an independent test after perfect
development validation. This is a direction-only diagnostic, not recursive play.
The [speed and direction combination](docs/training.md#isolated-horizontal-speed-and-frozen-direction)
now predicts complete near-paddle horizontal velocity, with 99.72% exact accuracy
on paddle hits in a fresh independent test.
The [full-game vx integration](docs/training.md#full-game-horizontal-velocity-with-a-frozen-paddle-component)
combines the frozen paddle component with a full-field predictor. The
[spatial acceleration head](docs/training.md#isolated-brick-triggered-horizontal-acceleration)
raises fresh-test horizontal-velocity accuracy to **99.9913%** and catches
**47 / 48** actual brick-triggered speed increases. The
[vertical-velocity model](docs/training.md#isolated-vertical-ball-velocity) reaches
**99.9929%** on a separate fresh test, with 13 errors in 182,441 transitions,
while preserving the horizontal predictor. These remain one-step results.
The [state-model status table](docs/state-model-status.md) lists each target,
its model and inputs, measured accuracy, and remaining gaps.
[Controller history probes](docs/training.md#inferring-controller-state-from-history)
test whether observed paddle/action histories can recover those internal values.
The [paddle dataset annotator](docs/training.md#paddle-controller-dataset-columns)
stores reconstructed controller values directly in transition and episode tables,
with exact paddle replay checks and unchanged original columns.

`recipe=breakout_reconstruction` trains only a single-frame autoencoder with RGB
MSE. Open its `best.pt` with `gymemu play` to compare held-out recorded frames with
their reconstructions. Validation runs after every epoch; every evaluated checkpoint
is retained, and the lowest validation reconstruction MSE selects `best.pt`.
See [representation training](docs/training.md#single-frame-reconstruction).
New W&B runs log the validation score once as `eval/mse`, against optimizer updates
on `eval/step`. The run configuration identifies whether the score measures
reconstruction or next-frame prediction.

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

# Browse environment IDs, training runs, then checkpoints
uv run gymemu play

# Play either approach using its checkpoint
uv run gymemu play runs/breakout-direct/best.pt
```

Omitting the checkpoint opens a navigator for local runs under `runs/` and published
R2 runs. Select an
environment ID, a training run, and a checkpoint to open the paused player. Use
`--runs-dir /path/to/runs` to browse another directory. The navigator supports
expandable Search with a clear-and-close button, breadcrumbs, browser Back, and
a Refresh icon for newly saved checkpoints.
R2 checkpoints download with their saved starting scene when selected; retained
versions appear in the checkpoint list. Use `--local-only` to browse offline.
The player's **Checkpoints** link returns to the selected run. Runs without saved
environment IDs appear under **Unknown environment**.

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
Frequent gradient health checks and fixed held-out rollouts expose saturation,
dead gradients, and ball loss during an epoch. See the [diagnostic metrics and
project view](docs/metrics.md) for chart meanings and configuration.
Use `wandb.mode=offline` to collect logs locally or `wandb.mode=disabled` to turn tracking off.
See [tracking options](docs/training.md#weights--biases) for teams and custom environments.

Checkpoints also upload to the separate `gymemu` R2 bucket, together with the
recorded start scene, metrics, and reproduction files. Each run has a unique prefix;
immutable objects and manifests retain successfully uploaded checkpoint versions.
Use `r2.enabled=false` to keep artifacts local. Retry interrupted uploads with
`uv run gymemu upload-checkpoints runs/my-run`.

Periodic checkpoints default to every 10 minutes (`trainer.checkpoint_seconds=600`).
New training runs also save `resume.pt`, including optimizer, RNG, and batch progress.
Saves before validation, after each epoch, and on graceful stop remain enabled.
Stop with Ctrl+C and wait for the checkpoint message before shutting down. Continue
in a new output directory with:

```bash
uv run gymemu train --resume runs/my-run/resume.pt output=runs/continued
```

See [stop and resume training](docs/training.md#stop-and-resume-training) for recovery
limits and compatibility. Older weights-only checkpoints cannot resume exactly.

The player opens paused in **teacher forcing** mode, using recorded dataset history
and actions. This is also the default when selecting a checkpoint in the navigator.
Use the mode selector or `--autoregressive` to play with predicted frames feeding
back into history. `--start-state`, `--start-scene`, `--empty-start`, `--key-action`,
and `--headless-actions` also select autoregressive mode.

Autoregressive play starts with the recorded scene in step mode. Every fresh action key press
predicts one next frame. Press Tab to toggle continuous play at 30 predictions per
second, matching 60 Hz Atari with frameskip 2. Hold an action key to repeat it;
releasing all keys uses action 0, or the first checkpoint action if 0 is absent.
The most recently pressed held key wins. Slow inference reduces playback speed
without catch-up steps. R restores the scene and pauses; Escape pauses. Breakout uses Left, Right, and Space. Other games use checkpoint key bindings,
or numbered keys for their action vocabulary. The player prints its bindings.

The Input history widget shows every RGB history frame, oldest to newest,
left to right and then top to bottom, with frame numbers overlaid at each tile's
top-left corner. Black frames retain the model's zero padding.
After inference it shows the exact stack used for the displayed prediction; before
the first prediction and after reset it shows the stack ready for the next step.
While paused after a prediction, drag a history tile onto another to move it there
and rerun inference for the same target. Alt + arrow keys also move a focused tile.
The temporary shuffle changes only RGB order; actions and auxiliary state stay fixed.
Scrubbing or stepping restores the original history. The shuffled prediction and
current MSE update together without changing the saved MSE chart or rollout history.
Each history tile also has a pencil button that opens a zoomed pixel editor.
The pencil pauses running playback. Before the first prediction it opens for
inspection and explains that you must step once before applying edits.
Choose a color already present in that frame and a square brush size from 1 to 32
image pixels, paint, then select **Apply and
predict**. Zoom, Alt-click color picking, undo, and reset are available. Painted
frames retain their edits when reordered; scrubbing or stepping clears all edits.
An edited tile shows a revert button in its bottom-right corner. Revert restores
that frame and reruns inference while keeping the other edits and current order.
Switching tabs or windows keeps playback running and clears held keys. Closing the player page pauses playback. Ctrl+C in the terminal stops the server.

The player opens two synchronized browser tabs using Gradlab's paired workspace
approach. The first contains Original, Prediction, Diff, and the
playbar. The Diagnostics tab contains Input history, Prediction error,
and custom metric widgets. Its default layout places Input history across the top,
with a full-width Prediction error chart below.
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
charts every measured transition on a labeled Step X axis. Hover moves a shared
cursor across all charts and shows tooltips with the nearest recorded step, metric
label, and value. Drag to zoom; a single click resets an active zoom. When fully
zoomed out, click a point to select that step in the player. The playbar shows the shared zoom range with adjustable handles. While zoomed,
Diagnostics also shows the segment selector beneath the charts. Drag either bracket
or use its arrow keys to resize the range without moving the playback cursor.
Seeking preserves measurements;
resetting, changing episode, or switching mode clears them. Unvisited steps are
unscored and gaps are left visible.
Original and Prediction have blank footers; the difference footer keeps its gain
control and legend. `--no-browser` prints both URLs without opening tabs;
`--port auto` chooses an unused port by default. Diagnostics observes playback and
does not send keyboard, heartbeat, or pause commands. Leaving the player tab keeps playback running; returning to either tab shows the current server snapshot.

To inspect one-step predictions without accumulated feedback errors, replay recorded
episodes with teacher forcing:

```bash
uv run gymemu play runs/breakout-direct/best.pt --teacher-forcing
```

The dashboard starts with Original, Prediction, and Diff widgets
from left to right. The signed RGB difference uses gray for zero,
brighter channels for positive differences, and darker channels for negative ones.
Every step uses recorded RGB history, executed actions, and any required auxiliary
state. Space steps, Tab plays or pauses, R restarts the episode, and C selects the
next episode. Replay pauses at episode end. The timeline ends at the furthest generated frame, keeping the scrubber at the end as new frames arrive. Scrubbing backward preserves that range. The metrics widgets show
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

For differentiable eight-step training, `recipe=breakout_autoregressive_fast`
selects the measured CUDA variant with compiled loss, bf16 feedback histories,
and a tuned batch size. It retains the full rollout loss
and diagnostics. The larger batch changes the number of optimizer updates per
epoch; this is a throughput recipe, not a demonstrated improvement in rollout
quality. See [autoregressive measurements](docs/performance.md#autoregressive-training).

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
  weights, and a comparison summary. `resume.pt` also preserves optimizer and training state.
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

### Train through predicted sequences

```bash
uv run python train.py recipe=breakout_autoregressive_ball_region r2.enabled=false
```

This experimental recipe trains on 1, 2, 4, then 8 autoregressive steps, with RGB
and auxiliary ball-region loss at every valid step. Gradients flow through predicted
frames. Episode endings mask unavailable targets; the direct baseline and held-out
one-step RGB metric are unchanged. See [the training guide](docs/training.md#differentiable-autoregressive-training)
for configuration and memory costs.

For generated rollouts with gradients limited to each prediction, use
`recipe=breakout_detached`. It keeps the per-step losses and detaches the entire history
before each forward pass. `recipe=breakout_detached_fast` selects the measured Beast-3
execution settings. See [detached rollout training](docs/training.md#detached-rollout-training)
for the full experiment and [throughput measurements](docs/performance.md#detached-rollout-training).

To measure whether ball position is linearly decodable from a frozen encoder, use
[`probe_ball_latents.py`](probe_ball_latents.py). It fits current/next-position
readouts on recorded training episodes and scores held-out episodes; see
[linear ball-position probes](docs/training.md#linear-ball-position-probes).

[`prototype_brick_grid.py`](prototype_brick_grid.py) tests deterministic brick-grid
labels from recorded Breakout RGB, with count checks and diagnostic images. See
[brick-grid extraction](docs/training.md#brick-grid-extraction-prototype).

[`augment_brick_dataset.py`](augment_brick_dataset.py) adds complete-dataset brick
matrices, quality flags, and initial-layout flags while checking that all original
columns survive unchanged. See [dataset annotations](docs/training.md#brick-dataset-annotations).

For the four-frame context / four-step rollout variant, use
`recipe=breakout_detached_h4_r4`. It keeps detached feedback, four action-history
slots, and the same dataset and objective, with a 1 → 2 → 4-step curriculum.


`probe_paddle_history.py` studies the frame/action context needed to predict paddle
position and native velocity from deterministic RGB-derived paddle features.
A compact next-state probe reached 0.225-pixel position MAE with features from
one frame and four actions on held-out episodes. See the
[experiment results](docs/history.md#2026-09-17-trained-paddle-state-history-probes)
and [reproduction commands](docs/training.md#standalone-paddle-history-experiment).

`probe_paddle_rgb.py` repeats this experiment using full RGB images and the
same CNN encoder as the frame autoencoder. It compares direct state regression
with a learned position head supervised by recorded normalized paddle labels.
See [full RGB probe training](docs/training.md#full-rgb-paddle-probe).

For **current-state estimation**, use the same script with `--current` and prepare
the recorded width with `--include-width`. The plain CNN directly outputs
normalized paddle position, velocity, and width for the newest observed frame.
It uses past actions, with no transition model or auxiliary classification head.
See [direct current paddle state](docs/training.md#direct-current-paddle-state).
The trained two-frame/one-action model reached 0.055-pixel current position MAE,
0.215-pixel/native-tick velocity MAE, and 0.043-pixel width MAE on held-out episodes.
See [the current-state results](docs/history.md#2026-09-17-direct-current-paddle-state-from-rgb-and-past-actions).

Preparing with `--include-ball` adds the existing normalized ball x, y, vx, and vy
labels to the same direct current-state regressor. The seven-output experiment
with two frames and one action achieved 0.713/0.815-pixel ball x/y MAE, but did
not meet the earlier accuracy criterion across all variables; paddle accuracy
also decreased. See [joint paddle/ball results](docs/history.md#2026-09-17-joint-current-paddle-and-ball-state)
and [training options](docs/training.md#joint-current-paddle-and-ball-state).
