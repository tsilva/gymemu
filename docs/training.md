# Training guide

For the locked Docker image, Beast-3 scheduling, Runpod, and bounded GPU checks,
see [container training](../containers/train/README.md).

The image in [verified-image.txt](../ops/dstack/verified-image.txt) passed Beast-3
verification on 2026-09-14 using its RTX 4090, PyTorch 2.14.0, and CUDA 13.0.
Direct, two-stage latent, and ball-state GPU smokes passed with bf16 and compilation.
A dstack job then trained `breakout_ball` on the pinned real dataset for eight
batches and evaluated two held-out batches, with the normal 210×160 RGB frames,
eight-frame history, and width-32 model. Checkpoint reload and 12-step playback
passed from both recorded and empty starts. W&B synced and the uploaded R2
checkpoint passed a SHA-256 read-back check. This proves execution, not rollout quality
or sustained throughput. See the [verification run](https://wandb.ai/tsilva/gymemu-Breakout-Atari2600-v0/runs/91iwybk4)
and [successful image build](https://github.com/tsilva/gymemu/actions/runs/34838073845).

With the private dstack coordinator connection configured, launch full training using:

```bash
uv run --frozen python ops/dstack/launch.py \
  --image "$(cat ops/dstack/verified-image.txt)" --name gymemu-ball --submit
```

The verified reference remains pinned when newer images are published. Update it
only after verifying the replacement on a GPU.

See the [README](../README.md) for setup, playback, and comparison commands.

For a ball-region RGB loss with recorded histories, run:

```bash
uv run python train.py recipe=breakout_ball_region
```

This inherits the `breakout_actions` data, model, cache, and training budget. Each
example requires one predictor forward pass; there are no generated training
prefixes or coordinate inputs. The loss is whole-frame RGB MSE plus `0.3` times
the mean error in a padded ball region detected from the recorded target. Tune
`approach.options.ball_region_weight` and `approach.options.padding`. Training and
validation log `rgb_mse`, `ball_region_mse`, and `ball_detection_coverage`; the region
metric includes zero for samples without a unique detection. Final checkpoint
selection and comparison still use ordinary held-out float32 RGB MSE. Playback
still feeds back predicted frames. See the [experiment guide](recipes.md#ball-region-loss-experiment)
for matched controls, throughput interpretation, and missed-detection limitations.

For joint RGB and ball-coordinate prediction, run
`uv run python train.py recipe=breakout_ball`. This uses the action-history recipe's
data and training budget, with frame-aligned normalized x/y inputs and successor
coordinate targets. The loss adds masked coordinate MSE with weight `0.01`; tune it
using `approach.options.coordinate_loss_weight`. The comparison metric stays RGB
MSE. See [alignment, initialization, and playback details](recipes.md#ball-coordinate-experiment).

Ball-position checkpoints trained on CUDA also play on Apple Silicon with
`play.py <checkpoint> --device mps`. The coordinate head handles non-divisible
adaptive-pooling bins on MPS using equivalent regional means. Existing checkpoint
weights load directly; no retraining or conversion is needed. CPU and CUDA retain
PyTorch's native pooling operation.

## Hierarchical configuration

[Hydra](https://hydra.cc/docs/intro/) composes YAML defaults and command-line overrides.
`configs/config.yaml` selects the defaults. Named `recipe` files collect complete
experiment settings; `experiment` presets apply afterward for runtime or smoke overrides. `hydra.job.chdir` is false, so relative dataset and scene paths remain relative
to the directory where you launch the command.

| Group | Controls |
| --- | --- |
| `game` | Canonical environment ID, dataset, revision, splits, bindings, and recorded scene |
| `model` | Direct predictor or latent codec architecture and dimensions |
| `approach` | Named models and ordered stages with objectives, epochs, and learning rates |
| `trainer` | Batch size, runtime device, precision, workers, and smoke limits |
| `optimizer` | Registered Adam or AdamW settings; each stage supplies its learning rate |
| `recipe` | Named experiment assembled from the other groups |
| `experiment` | Reusable overrides across groups, such as smoke or CUDA settings |

Top-level `history`, `seed`, `name`, and `output` apply to the whole experiment.
Saved recipes populate `expected_dataset` to reject changed data on replay. See the
[recipe guide](recipes.md) for `recipe=breakout_cnn` and `--recipe path/to/recipe.yaml`.
The latent approach selects the frame codec model group automatically. A command-line
`model=...` selection overrides that choice and must satisfy the codec interface.
Configs do not download data until training begins. Use `--cfg job --resolve` to inspect
a composition, or `--info defaults-tree` to inspect inheritance.

For example, a custom experiment can inherit CUDA settings:

```yaml
# configs/experiment/my_run.yaml
# @package _global_
defaults:
  - cuda
  - _self_
history: 4
trainer:
  epochs: 3
  batch_size: 32
```

Select it with `experiment=my_run`. Model configs can inherit another model in the
same way; `configs/model/direct_small.yaml` is a working example.
Hydra also supports external config search directories through `--config-dir`.

```bash
uv run python train.py model.width=16 trainer.learning_rate=0.0003
uv run python train.py --multirun model.width=16,32 history=4,8 seed=47,48
```

Multiruns execute sequentially with Hydra's default local launcher. Use
`hydra.sweep.dir=runs/my-sweep` to select their parent directory. Do not set `output`
during a sweep: a fixed directory would collide, and the runner rejects it. Each job
gets its own numbered directory, config, metrics, and checkpoints. Existing run
artifacts are never overwritten.

The old `--dataset`, `--output`, and other argparse training flags remain supported
for the direct baseline. Use the Hydra `key=value` interface for new experiments;
do not mix the two syntaxes. The Python interface is `compose_config(overrides)` from
`gymemu.config` followed by `train(cfg)` from `gymemu.engine`, with an explicit output.

## Defaults and precision

The direct baseline uses 8 history frames, width 32, batch size 32, Adam at 0.001, and
10 epochs. The latent example uses 10 epochs for each of its two stages. Stage epochs
and learning rates inherit trainer values unless explicitly overridden.

Device selection tries CUDA, MPS, then CPU. `experiment=cuda` sets CUDA, bfloat16,
six loader workers, two main-process CPU threads, batch size 64, and one epoch per stage.
Tune these settings for available resources; they are not benchmark guarantees.

CUDA loading uses pinned memory and nonblocking transfers. Workers keep image pixels
as exact uint8 values; normalization happens on the device. Nonzero workers use spawn
and persistent workers. `trainer.threads` limits main-process PyTorch CPU threads.
`experiment=cuda_cached` uses a verified LZ4 frame cache, two in-process loading
threads, compiled loss, CUDA prefetch, and finiteness checks every 100 batches and
before checkpoint writes. Set `trainer.frame_cache` to the cache directory. See
[performance measurements](performance.md) for construction and validation.

Training bfloat16 requires CUDA support. Validation and checkpoint playback use float32,
so reported RGB MSE has the same precision across approaches.

`experiment=smoke` bounds epochs, episodes, and batches and uses a small model on CPU.
It still fetches the snapshot and reads the episode tables. For a truly small local
smoke, provide a small compatible dataset with `game=custom`.

`recipe=breakout_scheduled` trains the action-history model with a progressive frame
feedback curriculum. It warms up on recorded context for two epochs and ramps to 80%
predicted-frame selection by epoch 8. Set `approach.options.schedule.warmup_epochs=0`
when a one-epoch smoke must exercise feedback. The extra sequential forwards make
later epochs more expensive. See [scheduled frame feedback](recipes.md#scheduled-frame-feedback).

`recipe=breakout_scheduled_fast` preserves that schedule, eight-step prefix, model
parameters, and final-target MSE while optimizing execution. It retains feedback in
the autocast dtype, factors the first convolution's constant action planes, and
groups selected predictions at low feedback probabilities. Its saved recipe records
these execution choices. Floating-point rounding and random-number streams can differ
from the reference; compare rollout quality after training. See
[performance measurements](performance.md#scheduled-feedback-optimization).

## Weights & Biases

W&B tracking defaults to `wandb.mode=online`. Set `wandb.mode=disabled` to turn it off.
The disabled path
does not initialize or import the SDK, require credentials, or create W&B files.
Authenticate once, then train any approach or recipe with the Hydra interface:

```bash
uv run wandb login
uv run python train.py recipe=breakout_cnn

# Select an existing W&B team and label related runs
uv run python train.py wandb.entity=my-team wandb.group=history-study

# Save W&B logs locally without credentials or an upload
uv run python train.py experiment=smoke wandb.mode=offline r2.enabled=false output=runs/tracked-smoke
```

Each experiment creates one W&B run across all training stages. Hydra multiruns
create a separate run for each job. `wandb.name` defaults to the experiment name;
`wandb.tags=[baseline,history8]` adds labels. Project names are derived from
`game.env_id`, the environment's canonical ID, with a `gymemu-` prefix. Both Breakout
dataset configs use `Breakout-Atari2600-v0`, producing
`gymemu-Breakout-Atari2600-v0`. Custom games must supply `game.env_id` when tracking
is enabled. The short `game.name` and dataset repository name do not determine the
project. W&B-forbidden characters `/`, `\`, `#`, `?`, `%`, and `:` become hyphens;
case and version suffixes are preserved.

Legacy argparse commands also default to online tracking. Use `--wandb-mode disabled`
or `--wandb-mode offline` to override it and `--env-id MyGame-v0` for a custom dataset.
Standalone recipes retain their saved mode; recipes created before tracking support
remain disabled unless a `wandb` configuration is added.

After each completed epoch, the run logs:

- `stages/<name>/train/*` and `stages/<name>/validation/*`: losses, samples, batches,
  elapsed seconds, throughput, and validation MSE, kept separate for each stage.
- `stages/<name>/epoch`, `stages/<name>/learning_rate`, and
  `stages/<name>/curriculum/*`, including scheduled feedback probability when present.
- `optimizer_steps` and `train_samples_seen`, accumulated across all stages.
- `evaluation/next_frame_rgb_mse` from the final predictive stage, with its minimum
  tracked for comparison. Representation-stage reconstruction scores do not enter
  this metric.

Charts use cumulative optimizer steps, so the axis continues when stage epochs restart.
Validation retains the shared float32 held-out evaluation path. The run configuration
includes resolved settings, pinned dataset identity, evaluation identity, parameter
count, and recipe hash. The W&B summary mirrors `summary.json`, including `best_mse`
and training budget. Logging happens once per epoch and adds no per-batch device
synchronization. W&B also collects its standard system metrics.

W&B files live in `<output>/wandb/`. To upload an offline run later, use
`uv run wandb sync <output>/wandb/offline-run-...` with its actual directory name.
Local metrics and checkpoints are still saved. Model and dataset artifacts are not
uploaded to W&B; checkpoint storage in R2 is configured separately below.
Configure W&B credentials through `wandb login` or
`WANDB_API_KEY`, never in a recipe. Enabled tracking errors are surfaced; a training
exception or interruption finishes the W&B run with a nonzero exit code.
See the [W&B SDK reference](https://docs.wandb.ai/models/ref/python/functions/init)
for authentication and run settings.

## R2 checkpoint storage

Training uploads run artifacts to the `gymemu` R2 bucket by default,
independently of the W&B mode. This is a separate bucket from Gradlab's model storage.
`r2.enabled=false` disables all R2 access. Old standalone recipes without an `r2`
section retain local-only storage.

Provision the bucket once in the same Cloudflare account you use for Gradlab:

1. In **R2 object storage**, create a bucket named `gymemu`. Keep it private.
2. Create an R2 API token with **Object Read & Write** permission scoped to that bucket.
3. Export its account endpoint, access key ID, and secret access key in the training
   process as `GYMEMU_MODELS_R2_ENDPOINT_URL`, `GYMEMU_MODELS_R2_ACCESS_KEY_ID`, and
   `GYMEMU_MODELS_R2_SECRET_ACCESS_KEY`.

See Cloudflare's [bucket creation](https://developers.cloudflare.com/r2/buckets/create-buckets/)
and [R2 token instructions](https://developers.cloudflare.com/r2/api/tokens/).
The trainer checks access to the existing bucket before loading data. It does not
create buckets or use Gradlab's credentials. Credentials stay outside Hydra configs,
checkpoints, source archives, and W&B. Store them in your secret manager or process
environment. The [.env.example](../.env.example) file lists the required names;
environment files are not loaded automatically.

On macOS, Gymemu can instead read a local `~/.config/gymemu/r2.toml` profile containing
the account endpoint and references to credentials stored in Keychain. The profile
contains no secret values and is not part of the repository or run artifacts:

```toml
endpoint_url = "https://ACCOUNT_ID.r2.cloudflarestorage.com"

[keychain]
account = "ACCOUNT_ID/gymemu"
access_key_id = "eu.tsilva.gymemu.r2.access-key-id"
secret_access_key = "eu.tsilva.gymemu.r2.secret-access-key"
```

`GYMEMU_R2_CONFIG` selects another profile path. If any of the three R2 environment
variables is set, all three must be supplied; the client never mixes environment
values with Keychain credentials. On remote training hosts, inject the three variables
through the host's secret manager or process environment.

```bash
# With W&B authentication and the three R2 variables already configured
uv run python train.py output=runs/breakout-stored

# Choose another dedicated bucket or object prefix
uv run python train.py r2.bucket=gymemu r2.prefix=experiments

# Local-only smoke with no W&B or R2 credentials
uv run python train.py experiment=smoke wandb.mode=disabled r2.enabled=false

# Retry publication using an existing run's saved R2 destination
uv run python upload_checkpoints.py runs/breakout-stored
```

Legacy argparse commands accept `--no-r2` to disable uploads and `--env-id` for a
custom environment's canonical ID. Use Hydra for bucket and prefix overrides.

Each run uploads under `runs/<canonical-env-id>/<unique-run-id>/` by default.
Environment characters outside letters, digits, dots, underscores, and hyphens become
hyphens in object prefixes. The upload receipt preserves the original environment ID.
The files include root and stage `best.pt`, `last.pt`, and `latest.pt`, the recorded
`start-scene.npz`, resolved config, recipe, source archive, reproduction receipt,
metrics, and completed-run summary. Dataset shards and W&B files are excluded.

Files are stored at `objects/<sha256>` within the run prefix. Identical aliases share
one object. After every file's upload succeeds and its remote size and SHA-256 metadata
match, `manifest.json` maps local relative paths to their immutable object keys,
hashes, and sizes. Previous successful manifests remain in `manifests/`, so later
checkpoint saves do not erase earlier published versions. Downloads should use the
manifest's paths and verify the recorded hashes; restore `start-scene.npz` beside the
root checkpoint before running the existing player.

Uploads run synchronously after periodic checkpoints and completed epochs. Network
transfer can extend save time. SDK requests have bounded retries and timeouts;
transient upload failures are reported and retried at the next save. A failed upload
leaves the previous remote manifest intact. The local `r2.json` receipt records the
destination and pending status. The retry command reuses that run prefix and uploads
the currently available local artifacts. A superseded local checkpoint that never
uploaded successfully is not retained separately.

At completion, the trainer requires a successful final upload before marking the W&B
run successful. A final upload failure raises an error while preserving the local
training results for retry. `summary.json` and the W&B summary include
`r2_manifest_uri` and `r2_run_id`. A later retry updates R2 and its local receipt; it
does not reopen or backfill a finished W&B run. These are inference checkpoints, with
the same playback and optimizer-resume limits as local files.

## Run artifacts

| Artifact | Contents |
| --- | --- |
| `recipe.yaml` | Standalone training config with a fresh output, pinned data, and preserved internal tuning links |
| `source.tar.gz` | Training/player Python sources, configs, Python version selection, and dependency lock |
| `reproduction.json` | Recipe/source hashes, dataset identity, Git state, packages, hardware, and numerical settings |
| `resolved.yaml` | Fully resolved configuration, including the absolute output path |
| `.hydra/` | Hydra composition and override metadata for Hydra-launched runs |
| `config.json` | Inference contract, architecture, action vocabulary, and dataset provenance |
| `metrics.jsonl` | Stage, epoch, curriculum, training loss, validation loss, RGB MSE, sample counts, and time |
| `summary.json` | Completed-run RGB score, evaluation identity, parameter count, and training budget |
| `wandb/` | Local W&B run files when online or offline tracking is enabled |
| `r2.json` | R2 run identity, destination, acknowledged objects, and upload status |
| `start-scene.npz` | Recorded RGB history and the executed actions between frames for playback |
| `stages/<name>/best.pt` | Selected checkpoint for this stage, including all models |
| `stages/<name>/last.pt` | Last completed epoch in this stage |
| `stages/<name>/latest.pt` | Periodic stage snapshot, which may precede evaluation |
| `best.pt`, `last.pt`, `latest.pt` | Corresponding checkpoints from the final predictive stage |

`trainer.checkpoint_seconds` defaults to 60. Checkpoint replacement is atomic. An
interrupted run can lose work after the latest snapshot and does not get a completed
summary. All checkpoints contain model weights and configuration, not optimizer state.
See [approaches.md](approaches.md) for stage selection and version compatibility.

`compare.py` only includes runs with completed summaries. It sorts within groups with
identical dataset identity, evaluated target sequence, RGB geometry, and metric.
Local datasets are content-hashed once per run; Hub datasets use immutable revisions.
Changing evaluation limits can produce a different comparison group. Changes in history
length or model architecture do not, provided the evaluated targets stay the same.
Comparison CSV includes parameters, steps, samples seen, and stage wall time. Match
budgets and seeds when drawing conclusions; lower MSE alone does not show better rollouts.

## Dataset contract

`game.dataset` accepts a Hub ID or local snapshot directory. Hub revisions resolve to
immutable commits before download. Breakout defaults to
`tsilva/gradlab-breakout-trajectories` at
`b8091d248295eb5135011dd9b943c75f4a7d50be`.

| Location | Required columns |
| --- | --- |
| `frames/assets/*.parquet` | `frame_id`, embedded HF `image` with `bytes` and `path` |
| `transitions/<split>/*.parquet` | `episode_id`, `step`, `source_frame_id`, `successor_frame_id`, `native_action_json` |
| `episodes/<split>/*.parquet` | `episode_id`, `initial_frame_id`, `length` |

Images must be RGB with consistent dimensions. Actions must decode to scalar integers;
the vocabulary is inferred from training episodes and evaluation cannot introduce new
actions. Frame IDs are joined by value, never interpreted as offsets. Episodes must have
unique IDs, contiguous steps, consistent lengths, and complete frame chains. Training and
evaluation episode IDs must be disjoint even when smoke limits are enabled.

Every episode supplies a bootstrap example with zero history and a reserved `START`
category meaning no game action. Later examples use preceding recorded frames in
chronological order with zeros on the left. Histories never cross episode boundaries.
The default direct and latent models receive only the current executed action.
`approach=direct_actions` also receives previous actions, oldest to current, with
`START` padding on the left. At target frame position `p`, an action context of length
`A` uses `episode.actions[max(0, p-A):p]`, where action `i` links frame `i` to `i+1`.
This is the same alignment in the standard and cached loaders. See the
[recipe guide](recipes.md#action-history-experiment) for the Breakout experiment.
Scheduled sampling extends only training windows with earlier frames and action
sequences, preserving every supervised target. Negative action tokens mark nonexistent
prefix steps before an episode; they never reach the model as game actions. Generated
context is built inside the approach on the training device. Evaluation always uses
the ordinary recorded-history windows for comparable next-frame MSE.
Only encoded images, trajectory arrays, and a bounded decoded-frame cache stay in memory;
windows are assembled on demand.

A generic game config can inherit the custom contract:

```yaml
# configs/game/my_game.yaml
defaults:
  - custom
  - _self_
name: my-game
env_id: MyGame-v0
dataset: owner/recorded-trajectories
revision: null
key_actions:
  left: 10
  right: 20
  space: 30
start:
  split: train
  episode_id: null
  frame_position: 0
```

Replace the dataset and actions with real values. `episode_id: null` selects the first
episode in the chosen split; `frame_position` is the chronological frame index, where
zero is the episode's initial frame. The runner takes up to `history` frames ending at
that position. It validates the selection rather than silently choosing another scene.

The Breakout config selects held-out episode 5, frame position 23, preserving the
previous full-brick-wall scene with the paddle and ball visible. This history is only
exported for playback and never enters optimization. Custom games default to the first
training episode's initial frame; choose a later position if startup animation is blank.
`game.start_scene=/path/to/scene.npz` overrides dataset-based scene selection.

Scene files are non-pickled NPZ archives with uint8 `frames` shaped
`[history, channels, height, width]`, oldest to newest, containing one to `history`
frames of the checkpoint's RGB geometry. New scenes also contain a one-dimensional
integer `actions` array of native action values between consecutive scene frames,
oldest to newest, with no action toward a future frame. An action-history model requires
at least the last `min(action_history - 1, frame_count - 1)` recorded actions. Legacy
frame-only scenes remain valid for models that use only the current action.
Reset restores both frames and actions. Subsequent frames come entirely from the model,
and each prediction uses the previous executed actions plus the current action.
The Input history widget shows every RGB history frame, oldest to newest,
left to right and then top to bottom. Black frames retain the model's zero padding.
After inference it shows the exact stack used for the displayed prediction; before
the first prediction and after reset it shows the stack ready for the next step.
Closing or leaving the browser pauses playback. Ctrl+C in the terminal stops the server.

The player starts in single-step mode. Tab toggles continuous playback, capped at
30 predictions per second for the 60 Hz Atari simulation with frameskip 2. Held
keys repeat on each tick; the most recently pressed held action key wins. With no
held keys, playback uses action 0, or the first checkpoint action if 0 is absent.
Slow inference lowers the effective rate without catch-up steps. R, C, or losing
window focus pauses continuous playback and clears held keys. In empty-start mode,
the first tick initializes the frame without executing a game action.

## Browser player workspace

`play.py` serves the player on loopback and opens its browser URL. No JavaScript
installation, build, CDN, or Gradlab installation is needed. `--no-browser` prints
the complete URL for opening manually or in Codex's in-app Browser. `--port auto`
selects an unused port; `--port NUMBER` selects a specific port. Each server has one
playback session and a random access token in its printed URL. Multiple tabs share
that session. Close the browser or lose focus to pause; Ctrl+C stops the server.

The UI reuses Gradlab's panel module lifecycle, registry, GridStack layout, fonts,
and theme. Each widget can be dragged, resized, hidden, or disabled. Hidden widgets
are restored from Panels. Disabled widgets remain in place but stop rendering.
Prediction widgets can expand to fullscreen. Add widget creates a metric widget;
its menu supports editing the title, selecting metrics, duplicating, and deleting
custom instances. The built-in error chart keeps the last 300 measured transitions.

Layout and widget configuration are stored in
`~/.config/gymemu/player-workspace.json`, with browser storage as a secondary copy.
Reset layout restores defaults. These preferences do not change inference inputs.
Extension instructions live in `gymemu/web_assets/panels/README.md`.

The mode selector switches between autoregressive play and teacher forcing using
the same checkpoint. Each switch resets the selected mode and pauses. Dataset
loading uses the same provenance and CLI overrides as startup. Switching to ordinary
play requires a compatible starting scene unless launched with `--empty-start`;
a missing scene remains an error. Episode and starting-scene selection are in
Playback controls. The timeline slider seeks teacher-forced targets directly using
the aligned dataset window. Seeking pauses and clears the displayed error history.

Frame images, input history, and metrics are committed together from one inference
revision. The browser only controls and visualizes playback; Python owns all model
inference and input history. Slow inference lowers playback rate without catch-up
steps. A lost browser heartbeat pauses playback and clears held actions.

## Teacher-forced replay

```bash
uv run python play.py runs/my-run/best.pt --teacher-forcing
uv run python play.py runs/my-run/best.pt --teacher-forcing --episode-id 5
uv run python play.py runs/my-run/best.pt --teacher-forcing --dataset /path/to/snapshot --split heldout
```

Replay uses the checkpoint's dataset and saved revision, and defaults to its evaluation
split, or `heldout` for checkpoints without split metadata. `--dataset`, `--revision`,
and `--split` override those selections. Selecting another dataset does not inherit
the original dataset's revision. Episode IDs are the recorded IDs in the selected
split, not row offsets. C cycles through episodes in ID order and wraps around.
An unknown episode, incompatible image geometry, missing required state labels, or
action outside the checkpoint vocabulary produces an error.

The dashboard opens paused with the real initial frame in Original. Prediction and
Prediction − original show a pending message until the first prediction.
Space predicts the next recorded transition, Tab toggles continuous
play at 30 predictions per second, R restarts the current episode, and C selects
the next episode. Reset, focus loss, and episode end pause replay. Keyboard presses
never replace recorded actions. Start-scene options and action overrides cannot be
combined with `--teacher-forcing`.

Each prediction uses the same aligned dataset windows as training: recorded RGB
frames ordered oldest to newest, left-zero-padding, and the executed action leading
to the displayed target. Models with action history receive the recorded action
sequence with START padding. Models with auxiliary state receive recorded state
history, including the unavailable initial-state marker. Predictions never feed back
into either RGB or state history. The Input history widget shows the exact
recorded RGB stack used for the displayed prediction.

The default layout shows the original on the left, prediction in the middle, and
prediction minus original on the right. The compact topbar shows the checkpoint,
playback mode, connection status, and icon buttons for Panels, Add widget, and Reset
layout; hover over an icon for its label. The timeline shows episode and target frame
position. The bar uses Gradlab's purple scrubber and icon controls: purple play,
amber pause, coral reset, and cyan playback settings. Open settings for previous-step
navigation in replay, single stepping, next episode or scene, selection, and action
buttons. The Playback controls widget provides the same selection and action controls.
Metric widgets show executed action and float32 next-frame RGB MSE over
pixels normalized to `[0, 1]`. Replay starts from a recorded initial frame;
it does not score empty-history bootstrap predictions. Its displayed MSE is for the
current transition, not the full held-out evaluation metric.

The difference subtracts the original from the prediction in float32, before display
quantization. Each signed RGB channel maps `[-1, 1]` to `[0, 255]`, with zero at gray
128, positive differences brighter, and negative differences darker. The display
uses a fixed scale across frames and preserves both signs. The browser difference
widget can amplify contrast by 2×, 4×, or 8× without changing the MSE. Mixed channel differences
appear colored. This visualization does not change the prediction or MSE.

Errors visible here occur even with recorded history. If a failure disappears here
but appears during autoregressive playback on comparable frames and actions,
feedback error is a plausible contributor. This diagnostic alone does not establish
that feedback is the only cause or that low pixel MSE implies playable rollouts.
For state-conditioned models, teacher forcing also removes accumulated state errors.

For a bounded replay smoke, save the comparison after at most N transitions:

```bash
uv run python play.py runs/my-run/best.pt --teacher-forcing --headless-steps 60 --output logs/replay.png
```

The image labels all three panels; stdout records the episode, frame position, action,
and MSE. A shorter episode stops at its last frame without crossing into another.

## Named debug start states

`play.py CHECKPOINT --start-state NAME` loads a named snapshot from
`start_states/<game>/NAME.npz`. `--list-start-states` lists names and descriptions for
the checkpoint's game. `--start-state`, `--start-scene`, and `--empty-start` are mutually
exclusive; omitting all three preserves the checkpoint's normal recorded start.
The repository's library is found regardless of the current working directory.
Use `--state-dir /path/to/library` to select a different library.

Press R to reset the current state, or C to reset to the next named snapshot.
No flag is needed. The cycle begins with your selected start, then visits the
other compatible states in alphabetical order and wraps around. `--state-dir`
selects the library; the timeline shows the current state. Each reset restores
RGB, action, and any auxiliary state histories without inference. Incompatible
library snapshots are skipped with a message. With no other compatible states,
C resets the current state too.

For later Breakout situations, use `half-cleared` for 50 remaining bricks,
`almost-cleared` for eight remaining bricks, or `above-bricks` for a ball that has
emerged through the left side of the wall and is moving above it.

Each named snapshot stores exact uint8 RGB frames, the native actions between them,
and a name, description, and available source provenance. R restores both histories.
No recorded frames are injected after startup. These are visual/action contexts for
the learned emulator, not serialized simulator states.

```bash
# Name an existing recorded scene without changing its frame or action values
uv run python save_start_state.py runs/my-run/best.pt --name opening \
  --scene runs/my-run/start-scene.npz --description "Recorded opening scene"

# Export another point in the checkpoint's recorded dataset
uv run python save_start_state.py runs/my-run/best.pt --name my-debug-state \
  --episode-id 5 --frame-position 45 --split heldout \
  --frame-cache data/breakout-676ff638-lz4
```

Dataset export defaults to the checkpoint's dataset/revision and held-out split;
`--dataset` can select a local copy. Snapshot export never fits weights. Small curated
states in the default library are versioned and included in new run source archives.
Use an ignored directory under `artifacts/` for disposable snapshots. Existing names
are never overwritten. See [the library](../start_states/README.md) for included scenes.

### Ball loss with prediction feedback

`recipe=breakout_scheduled_ball_region` combines the existing scheduled history
generation with `RGB MSE + 0.03 * ball-region MSE`. It trains from scratch for two
epochs, replacing eligible history frames with detached predictions at probability
0.4 in epoch 1 and 0.8 in epoch 2. The rollout prefix spans the configured history.
The model, dataset, seed, and other trainer settings match `breakout_ball_region`.
Masks and supervised targets always come from recorded frames. Validation uses
recorded histories and reports the common RGB MSE separately from the joint loss.

```bash
uv run --frozen python train.py recipe=breakout_scheduled_ball_region
```

This uses `approach=scheduled_ball_region`, which reuses scheduled context generation
and the ball objective. Generated context adds inference work to every training
batch. Evaluate ball survival and duplicates during playback as well as held-out
MSE; this objective does not enforce exactly one ball.
