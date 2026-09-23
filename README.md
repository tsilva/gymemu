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

Published experiments follow Environment → Research Goal → Goal Revision or Variant →
Run → Checkpoint → Player. The private R2 catalog records Run state and comparable
held-out RGB MSE. See [training and publication](docs/training.md) for direct training,
offline sync, and the checked-in recipe workflow.

## Desktop player

`uv run gymemu play` opens the checkpoint navigator in a dedicated Gymemu window.
Selecting a checkpoint opens the Player in that window; **Diagnostics** opens or
focuses a second window. Player and Diagnostics have separate app and browser-tab
icons. Closing both windows stops the local server. `--no-browser` prints complete
URLs for use in a normal browser or Codex's in-app Browser. The dedicated windows
use a pinned, SHA-256-verified Neutralinojs runtime cached under
`~/.cache/gymemu/neutralino`; the first launch downloads it. Installed Gymemu
includes the compiled Svelte UI and needs no Node.js at runtime.

For source UI development, run `pnpm install --frozen-lockfile`, `pnpm check:web`,
and `pnpm build:web`. The generated `gymemu/web_assets/dist/` files are included in
the Python package. Run `pnpm test:web` for the browser-side logic tests.

Play the published state dynamics and RGB decoder together:

```bash
uv run gymemu play-state
# Interactive generated play also starts from a complete recorded state.
uv run gymemu play-state --autoregressive
```

The player opens paused in teacher-forcing mode. Space predicts from the recorded
state and action, with Prediction, Original, and Diff panels. Select Autoregressive
for interactive play from the episode's starting state, using Left/Right/Space.
Tab toggles continuous play and R resets. `--autoregressive` starts directly from
the selected episode's complete recorded state. See
[state playback](docs/training.md#interactive-state-dynamics-and-decoder) for details.

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
while preserving the horizontal predictor. The
[position heads](docs/training.md#isolated-ball-positions-and-fractional-y) reach
**99.9888% exact x** and **99.9963% exact y including its fractional component**
on another fresh test. The
[brick-layout classifier](docs/training.md#isolated-next-brick-layout) makes
**zero errors on 45,245 fresh-test layouts**, including all 1,184 removals.
Wall clears/refills are absent from the audited data and remain unsupported.
The [contact-memory head](docs/training.md#isolated-next-brick-contact-memory)
also makes **zero errors on those 45,245 reused-test transitions**, including
when its predictions feed the following contact input. Other state fields remain
supplied, so full-state rollouts are still untested.
The [paddle-hit counter](docs/training.md#isolated-next-paddle-hit-count) reaches
**99.9978% exact count accuracy** on that reused test and **99.9912%** with count
and contact fed back. Its one false hit affects four count updates before a
reference segment boundary.
The [paddle-width predictor](docs/training.md#isolated-next-paddle-width) makes
**zero width errors on all 45,245 reused-test transitions**, including all 18
narrowings, and stays exact with width, count, and contact fed back together.
The [life-loss stop head](docs/training.md#isolated-life-loss-termination) makes
**zero errors across 45,283 reused-test transitions**, stopping on the exact
step of all 38 deaths with no premature stops. This uses recorded current ball
state; full-state feedback remains untested.
The [combined y/vy pair](docs/training.md#paired-vertical-position-and-velocity)
uses a small learned velocity correction to reconcile the two predictions.
On the reused test, fully exact 128-step vertical rollouts improve from
**98.6965% to 99.4286%** across 39,203 overlapping windows. Other state fields
remain supplied, and rare remaining errors can still produce large drift.
A subsequent [collision-displacement refinement](docs/training.md#vertical-collision-displacement-refinement)
improved development results but regressed on the reused held-out test and was
rejected. The existing combined pair remains the current model.
An [explicit collision-timing experiment](docs/training.md#explicit-vertical-collision-timing)
also regressed on the reused held-out test despite exact development predictions;
it was rejected too. A subsequent [split audit](docs/training.md#vertical-development-split-with-untrained-episodes)
establishes 400 development episodes excluded from every component's training,
while preserving 64 untouched final-test episodes. The development episodes were
previously evaluated, so their scores guide tuning rather than final reporting.
The [development feedback experiment](docs/training.md#development-vertical-feedback-and-factorized-paddle-outcomes)
finds paddle collisions are the largest first-failure group. A timing/speed
factorization reduces failed 128-step development windows by 21.8%, but introduces
one original-validation error and is not promoted. The existing pair remains current.
Adding [explicit paddle-edge distances](docs/training.md#paddle-edge-feature-experiment)
gives a small further drift improvement but retains that validation error and is
also rejected.
A [matched dataset comparison](docs/training.md#checkpoint-trajectory-dataset-comparison)
finds that trajectories from ten policy checkpoints reduce paddle y/vy errors
by 73% on old development data and 78% on new validation data, averaged over two
seeds with the same model and training budget. This tests the paddle branch;
the current reference remains unchanged pending promotion checks.
The subsequent [upper-screen continuation](docs/training.md#upper-screen-dataset-continuation)
improves new-data accuracy and recursive drift but regresses on old data.
A [fixed-size 50/50 mixture](docs/training.md#fixed-size-mixed-upper-screen-fitting)
recovers much of the old-data accuracy while retaining most new-data gains.
Legacy validation and rollout drift still prevent promotion; the reference stays unchanged.
The [horizontal pairing experiment](docs/training.md#horizontal-pair-on-new-checkpoint-trajectories)
reaches 99.9476% joint x/vx accuracy with independent predictors on new validation.
Sharing hidden layers nearly halves trainable parameters but slightly reduces
one-step and exact-path accuracy. Horizontal paddle interactions remain the main gap.
The next [ball-state merge](docs/training.md#four-variable-ball-state-merge)
combines x, y with its fractional part, vx and vy in one checkpoint. It reaches
99.9400% exact joint validation accuracy and 93.3265% entirely exact 128-step
ball-feedback windows.
The [ball and brick-layout merge](docs/training.md#ball-and-brick-layout-merge)
now emits both from one checkpoint, with 99.9337% joint accuracy and 92.5407%
exact coupled 128-step windows. Joint continuation regressed bricks, so the
selected candidate preserves the parent weights.

The [ball/bricks/contact merge](docs/training.md#ball-bricks-and-contact-merge-on-migrated-data)
adds contact memory: **99.9293%** exact joint validation accuracy and
**92.2644%** exact 128-step windows while feeding all these
fields back. It uses the verified schema-v1 migration at revision `8f9838c`,
with unchanged train/validation records. Paddle/count state and life boundaries
remain supplied.

The [count integration](docs/training.md#paddle-hit-count-added-to-the-transition-container)
adds capped paddle-hit count while keeping the existing branches frozen. It reaches
**99.9171%** exact combined validation accuracy, **99.9834%** count accuracy,
and **91.5089%** entirely exact 128-step windows with count fed back too.
Paddle x/width/charge and reference life boundaries remain supplied.

The [width integration](docs/training.md#paddle-width-added-to-the-transition-container)
adds paddle width with **100.0000%** exact validation accuracy, including all
115 width-change cases. The combined container scores **99.9171%** one-step
accuracy and **91.5089%** entirely exact128 windows with width fed back.
Paddle x, charge and reference life boundaries remain supplied.

The [charge integration](docs/training.md#paddle-charge-added-to-the-transition-container)
adds current action input and predicts next paddle charge with **100.0000%**
validation accuracy. Combined one-step accuracy is **99.9171%**, with
**91.5089%** entirely exact128 windows when charge is fed back too.
Paddle x and reference life boundaries remain supplied.

The [paddle-position integration](docs/training.md#paddle-position-added-to-the-transition-container)
feeds every compact state field back, including paddle position and charge.
Paddle-position validation accuracy is **99.9834%**, combined accuracy
**99.9005%**, and entirely exact128 windows **90.0675%**.
That experiment used recorded life boundaries; the next integration adds learned stopping.

The [termination integration](docs/training.md#life-loss-termination-added-to-the-transition-container)
adds learned stopping and masks terminal successor states. All 239 validation
deaths are detected without false stops on recorded inputs. Full state feedback
stops on the exact death step in **169/239**
life segments; **141/246** entire segments
are exact. Recursive trajectory errors remain; actions are recorded, and missed
deaths are censored at the reference end.

A [first-error audit](docs/training.md#first-errors-in-full-state-rollouts)
finds that 67 of the 105 divergent segments first fail in ball prediction,
52 near the paddle. Existing training errors provide a mining pool; vertical
paddle predictions also show a generalization gap. No weights changed in this audit.

A [separate unified MLP benchmark](docs/training.md#unified-residual-mlp-benchmark)
trains one shared residual network from scratch for 20 uniform epochs. It reaches
**98.1852%** exact joint validation accuracy and **13/246** completely exact
segments, versus **99.9006%** and **141/246** for the container. The training probe
reaches 99.9969%, indicating a substantial generalization gap. The container stays
selected; no mining, oversampling or dataset changes were introduced.

A [frameskip-1 retraining run](docs/training.md#unified-mlp-on-frameskip-1)
uses the same shared architecture and 32,080 updates on the newly balanced dataset.
It reaches **99.6056%** exact one-step validation. Over 256 native frames,
completely exact sampled feedback windows improve from **20.38% to 51.83%**.
The recordings and policies also differ, so this does not isolate frameskip alone.
The new checkpoint is separate, and test remains reserved.
The [published model](https://huggingface.co/tsilva/gymemu-breakout-unified-dynamics-fs1) includes standalone PyTorch inference code and the full state contract.

State playback initializes from a complete recorded state, including controller
charge. A [controller fine-tuning experiment](docs/training.md#recorded-initialization-and-controller-fine-tuning)
improves controller stress accuracy but still exhibits some rollout failures;
its separate local playback configuration is documented with the results.

A [longer training run](docs/training.md#longer-training-for-the-frameskip-1-unified-mlp)
adds 32,080 updates with the same architecture and uniform sampling. Exact
one-step validation improves to **99.6585%**, and completely exact sampled
256-frame windows reach **55.49%**. These continued weights are saved separately;
the published model remains unchanged.

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
`uv sync` updates the checkout's `.venv`, while the installed `gymemu` command uses
its own uv tool environment. After pulling a change that adds dependencies, refresh
the installed command with `uv tool install . --editable --reinstall --exclude-newer "7 days"`.
Alternatively, run `uv run gymemu play` from the checkout to use `.venv` directly.
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

Omitting the checkpoint opens the private R2 research catalog. Select an
Environment, Research Goal, Revision, Variant, Run, and Checkpoint to open the
paused Player. Use `--local-only --runs-dir /path/to/runs` to browse unpublished
local runs. The navigator supports
expandable Search with a clear-and-close button, breadcrumbs, browser Back, and
a Refresh icon for newly saved checkpoints.
Published inference Checkpoints download from the public `gymemu-public` bucket
with their saved starting scene and are checked against the Run manifest by size
and SHA-256. Recovery files stay private. Use `--local-only` to browse offline.
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

Run artifacts upload to the private `gymemu` R2 bucket. Approved inference and
playback files also upload to the public `gymemu-public` bucket. Each Run has a unique prefix;
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

The player uses Gradlab's paired workspace approach. Its native Player window
contains Original, Prediction, Diff, and the playbar. The Diagnostics window contains Input history, Prediction error,
and custom metric widgets. Its default layout places Input history across the top,
with a full-width Prediction error chart below.
Both display snapshots from one inference session.
The header's Diagnostics/Player link opens or focuses the companion window.
With `--no-browser`, the same link opens a browser tab.

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
control and legend. `--no-browser` prints both URLs without opening native windows;
`--port auto` chooses an unused port by default. Diagnostics observes playback and
does not send keyboard, heartbeat, or pause commands. Switching windows keeps playback running; returning to either window shows the current server snapshot.

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

A separate [recorded-state decoder experiment](docs/training.md#recorded-state-decoder)
reconstructs Breakout playfields from ball/paddle positions and brick occupancy.
It uses explicit spatial features and a small palette-classification MLP; HUD
rendering and integration with dynamics playback remain outside this benchmark.
The decoder is available on [Hugging Face](https://huggingface.co/tsilva/gymemu-breakout-state-decoder-fs1)
with standalone PyTorch inference, validation metrics and reconstruction examples.
