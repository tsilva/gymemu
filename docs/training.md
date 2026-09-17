# Training guide

Install the command with `uv tool install . --editable --exclude-newer "7 days"` from the checkout.
Run `gymemu --help` to list subcommands. The `uv run gymemu` examples below use
the locked project environment; the installed `gymemu` command also works outside
the checkout. Relative dataset, output, and checkpoint paths use your current directory.

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
uv run gymemu train recipe=breakout_ball_region
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
`uv run gymemu train recipe=breakout_ball`. This uses the action-history recipe's
data and training budget, with frame-aligned normalized x/y inputs and successor
coordinate targets. The loss adds masked coordinate MSE with weight `0.01`; tune it
using `approach.options.coordinate_loss_weight`. The comparison metric stays RGB
MSE. See [alignment, initialization, and playback details](recipes.md#ball-coordinate-experiment).

Ball-position checkpoints trained on CUDA also play on Apple Silicon with
`play.py <checkpoint> --device mps`. The coordinate head handles non-divisible
adaptive-pooling bins on MPS using equivalent regional means. Existing checkpoint
weights load directly; no retraining or conversion is needed. CPU and CUDA retain
PyTorch's native pooling operation.

## Browse saved checkpoints

```bash
uv run gymemu play
uv run gymemu play --runs-dir /absolute/path/to/runs
```

Without a checkpoint argument, the browser opens an environment → training run →
checkpoint navigator for local and R2 runs. It discovers local run directories recursively, including
Hydra timestamped runs, and sorts runs and checkpoints by their latest save time.
Each run lists its root checkpoint files and `stages/<stage>/*.pt` checkpoints.
Stage checkpoints use `start-scene.npz` from the run directory. Selecting a
checkpoint loads it into the existing player in paused mode; the **Checkpoints**
link returns to the catalog. Only one checkpoint is active per navigator server.

Open Search at each level to filter the list. The field focuses automatically;
the × button clears and closes it. Use the breadcrumbs and browser Back to navigate.
The Refresh icon spins and stays disabled while the list reloads.
Refresh discovers new files. Runs without checkpoints display an empty state;
unreadable runs display a notice. Older runs missing `config.json` use safely loaded
checkpoint metadata for their environment ID, with **Unknown environment** when no
ID was saved. Discovery does not construct inference models.

R2 uses the existing credential profile and defaults to bucket `gymemu`, prefix
`runs`. Override these with `--r2-bucket` and `--r2-prefix`. Each row identifies its
local or R2 source. Opening an R2 run lists its current checkpoints and retained
versions from immutable manifests, deduplicating repeated publications of the same
file contents. Earlier versions show a short content hash. A remote run's count
initially covers current checkpoint files and expands when its history is loaded.

Selection downloads the checkpoint and the starting scene from the same manifest
into `~/.cache/gymemu/checkpoints/`. Both downloads and cached files are checked
against the published SHA-256 hashes and sizes. Selecting another checkpoint reuses
verified cached files. Checkpoint loading still uses `weights_only=True` and the
model registry. It does not execute Python targets from manifests or metadata.

Use `--local-only` to avoid contacting R2. If R2 is unavailable, the navigator shows
a notice and keeps local runs accessible. Refresh retries remote discovery.

Playback options such as `--device cpu`, `--autoregressive`, `--start-state`, and
`--empty-start` apply to each selected checkpoint. In autoregressive mode, missing
starting scenes remain an error unless an explicit alternative is supplied. Headless playback and
`--list-start-states` still require a checkpoint argument. Supplying a checkpoint
directly continues to open the player and diagnostics tabs.

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
uv run gymemu train model.width=16 trainer.learning_rate=0.0003
uv run gymemu train --multirun model.width=16,32 history=4,8 seed=47,48
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
uv run gymemu train recipe=breakout_cnn

# Select an existing W&B team and label related runs
uv run gymemu train wandb.entity=my-team wandb.group=history-study

# Save W&B logs locally without credentials or an upload
uv run gymemu train experiment=smoke wandb.mode=offline r2.enabled=false output=runs/tracked-smoke
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
and training budget. The legacy aliases above remain available for existing charts.
New runs also use metrics schema v2, with short `train/`, `eval/`, and `probe/` names
and latest-value summaries. `eval/mse` uses `eval/step`; training and fixed probes
use `train/step`. Both axes count cumulative optimizer updates. W&B also collects
its standard system metrics.

Read-only diagnostics are enabled by default. Global gradient norms are measured
on every update and the window maximum preserves single-update spikes. Layer
gradients, actual relative parameter updates, ReLU activity, and pre-sigmoid
statistics are sampled on the first ten updates of each epoch and every 100 batches.
A fixed held-out probe runs before each epoch, every 1,000 batches, and at epoch end.
It compares up to 32 recursive predictions against episode-local recorded targets,
with recorded actions and no correction of predicted history. It also evaluates
clean histories against those targets. Stateful approaches feed back predicted
state and omit the clean-history aggregate. Representation-only stages omit
predictive probes. Probes preserve module modes, buffers, and Torch RNG state,
and never call the training objective or alter gradients.

Settings live under `trainer.diagnostics`: `enabled`, `log_every`, `probe_every`,
`samples`, and `horizon`. Diagnostics add compute and sampled device synchronization;
set `trainer.diagnostics.enabled=false` for an explicit uninstrumented benchmark.
Saved recipes without this block preserve their old behavior. Scalars are saved
to `diagnostics.jsonl` even with W&B disabled; `probe.json` identifies fixed starts
and target frame IDs; `diagnostics/*.png` contains target/prediction comparisons.
W&B receives the same scalars and images. These probes are monitoring subsets,
not replacements for full held-out evaluation or checkpoint selection.

See [the metric reference](metrics.md) for interpretation and the managed project
view. The view follows GradLab conventions: a pinned, open primary accordion;
separate closed diagnostic accordions; unsmoothed plots; explicit axes; registry-
validated selectors; stable view identity; and no automatic panel generation.
To create or update it after authenticating:

```bash
uv sync --frozen --extra monitoring
uv run --frozen --extra monitoring python -m gymemu.workspace \
  --entity tsilva --project gymemu-Breakout-Atari2600-v0
```

The declaration is `configs/monitoring/workspace.json`. Re-running updates the same
managed view. Existing personal views are separate. Old runs retain their original
metrics in the Legacy runs accordion; they cannot gain new diagnostics retroactively.

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
uv run gymemu train output=runs/breakout-stored

# Choose another dedicated bucket or object prefix
uv run gymemu train r2.bucket=gymemu r2.prefix=experiments

# Local-only smoke with no W&B or R2 credentials
uv run gymemu train experiment=smoke wandb.mode=disabled r2.enabled=false

# Retry publication using an existing run's saved R2 destination
uv run gymemu upload-checkpoints runs/breakout-stored
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
does not reopen or backfill a finished W&B run. R2 also stores `resume.pt`, including
the optimizer and progress required to continue training after a restart.

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
| `resume.pt` | Full training state: weights, Adam state, stage/epoch/batch progress, RNG, metrics, and stage-best weights |

`trainer.checkpoint_seconds` defaults to 60. Checkpoint replacement is atomic. An
interrupted run can lose work after the latest snapshot and does not get a completed
summary. The playable `best.pt`, `last.pt`, and `latest.pt` files contain weights and
configuration. The separate `resume.pt` also includes training state.
See [approaches.md](approaches.md) for stage selection and version compatibility.

### Stop and resume training

New runs save `resume.pt` every `trainer.checkpoint_seconds` (60 by default), before
validation, and after each completed epoch. It contains all model weights, Adam
moments and step counters, the active stage and curriculum epoch, completed batch
and sample counts, accumulated metrics, stage-best weights, and Python/NumPy/PyTorch
RNG state (including CUDA or MPS when used). The current bf16 path has no gradient
scaler or learning-rate scheduler to restore.

Ctrl+C or SIGTERM requests a stop at the next completed optimizer update, followed
by a local checkpoint and an R2 upload attempt. During validation, the pre-validation
checkpoint is already safe; stopping there repeats validation on resume. Wait for
`Training stopped; resume.pt saved...` before shutting down. SIGKILL, power loss,
or a forced scheduler timeout can only recover the last successfully saved checkpoint.
Local checkpoint writes are flushed and atomically replaced. R2 failures leave the
local file intact and can be retried with `gymemu upload-checkpoints`.

```bash
uv run gymemu train --resume runs/original/resume.pt output=runs/continued
```

The checkpoint supplies the resolved recipe. The continuation writes to a **new,
empty output directory** and starts a new W&B/R2 run; `resumed_from` in `config.json`
links it to the source checkpoint. Omit `output` for a timestamped directory. Keep
both directories if you want the complete log history. Global optimizer/sample
counters continue; old diagnostic and epoch log files are not copied or rewritten.
The same syntax works with `python train.py` and `gymemu-container run`.

The sampler reconstructs the saved epoch's deterministic shuffle and starts at the
first unconsumed sample, independent of loader prefetch. This applies to both the
standard and cached loaders and the built-in deterministic trajectory datasets.
New runs use a dedicated epoch seed, so their shuffled order differs from versions
that used the global RNG even with the same experiment seed.
No already-completed optimizer updates are replayed. Resuming a completed stage
restores its best weights before preparing the next stage, including its frozen models.

Resume requires the same training recipe, dataset identity, device type and PyTorch
version. Output/tracking/storage settings and loader workers, threads, cache path,
and checkpoint interval may change. Use the saved source and locked environment
for reproducibility. State restoration does not guarantee bitwise GPU equality when
kernels are nondeterministic; CPU interruption tests compare weights and Adam state
exactly, including scheduled sampling and multi-stage training.

Older runs only saved inference weights. They **cannot** be converted into exact
resumable checkpoints: their optimizer and RNG state were never stored. Passing
`latest.pt` to `--resume` is rejected rather than silently restarting Adam.

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
After pausing on a prediction, drag one history tile onto another to insert it at
that position and rerun inference for the same target. Alt + arrow keys move the
focused tile too. Labels identify each frame's original slot while shuffled.
This temporary experiment changes only RGB frame order; action tokens, auxiliary
state, and the target stay fixed. The prediction, difference image, and current
float32 RGB MSE update together. Recorded MSE chart points and autoregressive
rollout history remain unchanged. Scrubbing, stepping, resetting, or switching
episodes or modes restores normal history. A changed prediction demonstrates
sensitivity to the chosen order; an unchanged prediction alone does not prove
the model ignores history.
The pencil button on each history tile opens a zoomed pixel editor, pausing playback
if needed. Before the first prediction, the popup opens for inspection and explains
that you must step once before applying edits. Its swatches contain only colors
present in the frame when
opened. Choose a square brush size from 1 to 32 image pixels; zoom does not change
its painted area. Click or drag to paint, Alt-click to pick an existing
color, and use zoom, Undo stroke, or Reset painting as needed. Apply and predict
reruns the same target using the painted frame; Cancel discards unapplied strokes.
The server validates colors against the original frame palette and preserves
untouched float pixels. Edits follow the original frame through reordering, and
scrubbing or stepping clears both painting and shuffling. The current prediction,
difference, and MSE update together without changing recorded data, chart points,
action/state inputs, or future rollout history.
Edited tiles show a revert button in their bottom-right corner. It restores that
frame's original float pixels and reruns inference for the same target. Other
painted frames and the current frame order stay intact.
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
installation, build, CDN, or Gradlab installation is needed. Playback opens a player
tab and a Diagnostics tab. `--no-browser` prints both complete URLs for opening
manually or in Codex's in-app Browser. `--port auto`
selects an unused port; `--port NUMBER` selects a specific port. Each server has one
playback session and a random access token shared by its printed URLs. Both tabs
read the same atomic frame/metric revisions from that session, without duplicate
inference. Leaving or closing the player tab pauses playback; Diagnostics only sends explicit chart selections as seek commands; it sends no
keyboard, heartbeat, or blur commands. Ctrl+C stops the server.

The UI reuses Gradlab's panel module lifecycle, registry, GridStack layout, fonts,
and theme, including paired views with per-tab panel placement. Original, Prediction,
and difference belong to the player tab with its playbar; input history, error charts,
and custom widgets belong to Diagnostics. Its default arrangement puts
Input history across the full top row, with a full-width Prediction error chart
below. History tiles use minimal spacing, overlaid top-left frame numbers,
and no bottom caption. Older default arrangements migrate; custom placements remain. Header links open or focus
the companion tab. Layout updates use BroadcastChannel with a storage-event fallback
and ordered revisions, following Gradlab's workspace synchronization strategy.
The comparison panels automatically fill the available height above the playbar in
equal-width columns. All three reserve equal-height footers so the image viewports
align; Original and Prediction footers are blank. Narrow screens stack the panels.
Unused space around original, predicted, difference, and history frames uses the widget
background color so black image pixels remain distinct from display padding.
Diagnostic widgets can be dragged and resized. All widgets can be hidden or disabled. Hidden widgets
are restored from Panels. Disabled widgets remain in place but stop rendering.
Original and Prediction show frames without footer labels or expand buttons; the
difference widget retains its gain control, current RGB MSE, and legend. Add widget creates a metric widget;
its menu supports editing the title, selecting metrics, duplicating, and deleting
custom instances. The error chart keeps all measured transitions in the current episode, sorted by
timestep, with labeled Step X axes. Hover moves a synchronized cursor across every
chart; tooltips show the nearest recorded step, metric label, and value. Drag
horizontally to zoom. A single click resets an active zoom without seeking. When
fully zoomed out, clicking a point pauses and selects the same target in both tabs
and the playbar. Double-click or Reset zoom also resets the range. While zoomed, Diagnostics also shows a segment selector beneath the charts.
The playbar in both tabs highlights the shared zoom range, with handles
that can be dragged or adjusted with arrow keys (Shift moves ten steps), Home, and End.
On a focused chart, arrow keys or Home/End inspect points; Enter or Space selects.
Unvisited steps remain unscored, with gaps
rather than interpolated errors. The chart cursor marks the selected step until hover takes over; the amber
line marks the hovered step. Reset, episode changes, and mode changes clear the series
and zoom. Stale chart selections from a previous episode or reset are rejected.

Layout and widget configuration are stored in
`~/.config/gymemu/player-workspace.json`, with browser storage as a secondary copy.
Reset layout restores defaults for both tabs. Old layouts retain their widget
settings, with diagnostic placements moved into the second tab. These preferences
do not change inference inputs.
Extension instructions live in `gymemu/web_assets/panels/README.md`.

New launches default to paused teacher forcing, including checkpoints opened from
the navigator. Use `--autoregressive` to start with predicted frames feeding back
into history. Scene options (`--start-state`, `--start-scene`, `--empty-start`) and
action overrides (`--key-action`, `--headless-actions`) also select autoregressive
mode. `--teacher-forcing` remains available as an explicit choice; dataset replay
options work without it on a default launch.

The mode selector switches between autoregressive play and teacher forcing using
the same checkpoint. Each switch resets the selected mode and pauses. Dataset
loading uses the same provenance and CLI overrides as startup. Switching to ordinary
play requires a compatible starting scene unless launched with `--empty-start`;
a missing scene remains an error. Episode and starting-scene selection are in
the playback settings panel opened by the bottom bar's gear. Play/pause, reset,
step navigation, action buttons, and keyboard help are also in settings. Saved layouts
automatically drop the former Playback controls widget while preserving other widgets.
The timeline slider spans the initial frame through the furthest generated frame,
so the scrubber stays at the end while generating new frames. Scrubbing backward
preserves that range; resetting or selecting another episode clears it. The slider
seeks teacher-forced targets directly using the aligned dataset window. Seeking pauses and preserves measured errors, replacing the value if a step is measured again.

Frame images, input history, and metrics are committed together from one inference
revision. The browser only controls and visualizes playback; Python owns all model
inference and input history. Slow inference lowers playback rate without catch-up
steps. A lost browser heartbeat pauses playback and clears held actions.

## Teacher-forced replay

```bash
uv run gymemu play runs/my-run/best.pt --teacher-forcing
uv run gymemu play runs/my-run/best.pt --teacher-forcing --episode-id 5
uv run gymemu play runs/my-run/best.pt --teacher-forcing --dataset /path/to/snapshot --split heldout
```

Replay uses the checkpoint's dataset and saved revision, and defaults to its evaluation
split, or `heldout` for checkpoints without split metadata. `--dataset`, `--revision`,
and `--split` override those selections. Selecting another dataset does not inherit
the original dataset's revision. Episode IDs are the recorded IDs in the selected
split, not row offsets. C cycles through episodes in ID order and wraps around.
An unknown episode, incompatible image geometry, missing required state labels, or
action outside the checkpoint vocabulary produces an error.

The dashboard opens paused with the real initial frame in Original. Prediction and
Diff show a pending message until the first prediction.
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
buttons. Playback controls are contained in settings rather than a dashboard widget.
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
uv run gymemu play runs/my-run/best.pt --teacher-forcing --headless-steps 60 --output logs/replay.png
```

The image labels all three panels; stdout records the episode, frame position, action,
and MSE. A shorter episode stops at its last frame without crossing into another.

## Named debug start states

`play.py CHECKPOINT --start-state NAME` loads a named snapshot from
`start_states/<game>/NAME.npz`. `--list-start-states` lists names and descriptions for
the checkpoint's game. `--start-state`, `--start-scene`, and `--empty-start` are mutually
exclusive; `--autoregressive` uses the checkpoint's normal recorded start.
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
uv run gymemu save-start-state runs/my-run/best.pt --name opening \
  --scene runs/my-run/start-scene.npz --description "Recorded opening scene"

# Export another point in the checkpoint's recorded dataset
uv run gymemu save-start-state runs/my-run/best.pt --name my-debug-state \
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

## Differentiable autoregressive training

`recipe=breakout_autoregressive_ball_region` starts each example with recorded RGB
history and generates future frames sequentially using the aligned recorded actions.
Every subsequent input uses predicted RGB, with gradients preserved through feedback.
Every valid target receives whole-frame RGB MSE plus ball-region MSE weighted by 0.03.
The region term is zero when the target-image detector cannot locate a ball; this
objective does not supervise invisible ball coordinates.

The recipe uses horizons 1, 2, 4, and 8 in epochs 1 through 4, then keeps 8 through
epoch 10. `approach.options.rollout_schedule` lists one horizon per epoch; its last
value is repeated and must equal `approach.options.rollout_steps`, the maximum.
Use `approach.options.rollout_schedule=[8]` for eight steps from the start. Future
horizon and RGB history length are independent; the horizon can exceed history.

Every recorded frame remains a possible first target, including bootstrap and final
frames. Future targets stop at the episode boundary, with negative action tokens
marking invalid right padding. Loss is averaged over valid steps within each example,
then over examples, giving every sampled start equal weight. The diagnostic RGB and
ball-region means pool valid frames, so they can differ from the example-weighted
training objective when remaining sequence lengths differ. Bootstrap has zero RGB
history and no game action. Both standard and cached loaders preserve this alignment.

This recipe uses batch size 8 and disables compilation initially to limit rollout
memory and compilation costs. `recipe=breakout_autoregressive_fast` is the explicit
throughput variant, using compilation, a tuned larger batch, and bf16 feedback
histories. Both retain the same full-resolution targets, rollout
curriculum, objective, and diagnostic frequency. The larger batch changes the
optimization trajectory and number of updates per epoch. See
[performance measurements](performance.md#autoregressive-training) before comparing
throughput or training quality.
Training loss is a sequence objective; checkpoint selection and held-out comparison
remain float32 one-step RGB MSE on the same recorded targets as other approaches.
The ordinary player and checkpoint contract remain unchanged. Better collision
recovery must be checked in held-out recursive playback; it is not guaranteed.

```bash
uv run python train.py recipe=breakout_autoregressive_ball_region r2.enabled=false
```

Use `approach.options.feedback_dtype=autocast` in the fast recipe to retain generated
history in the active mixed-precision dtype. Loss reduction remains float32, and
feedback remains differentiable. Float32 evaluation/playback use the original model
arithmetic. Old recipes default to float32 history. Compilation may change numerical
rounding; sampled activation diagnostics run eagerly to avoid recompiling whenever
hooks are installed. Scalar health statistics transfer together at reporting time;
every-update gradient norms and the existing collapse signals remain enabled.
The fast recipe sets `trainer.compile_layout_optimization=false` to retain the
reference convolution layouts. Automatic layout conversion and factored action
inputs were faster in pilot tests but changed gradients substantially on a trained
eight-step diagnostic batch; neither is enabled in this recipe.

```bash
uv run python train.py recipe=breakout_autoregressive_fast \
  trainer.frame_cache=data/breakout-676ff638-lz4
```

## Detached rollout training

`recipe=breakout_detached` trains from initialization for ten epochs on the pinned
Breakout revision `676ff6388f4218d3c3a3ce9f2f33e075fa7314a3`. It uses eight RGB history
frames, eight action tokens, width 32, Adam at 0.001, batch size 32, and a horizon
curriculum of 1, 2, 4, 8 in epochs 1 through 4, retaining 8 afterward. Its ball-region
weight is 0.03. The model generates all feedback frames. It detaches the entire input
history before each next-frame prediction and averages the losses over valid steps
within each example, then over examples. The runner applies one optimizer update per
batch. No gradient clipping is enabled.

The recipe retains cached input loading, bf16 feedback, CUDA prefetch, compiled loss
with the reference convolution layout, two loader workers, and two PyTorch threads.
`recipe=breakout_detached_fast` uses batch size 64, automatic compiler layout
optimization, and factored action inputs. It retains the same horizon curriculum,
optimizer, loss weights, and diagnostic cadence. Consult
[performance results](performance.md#detached-rollout-training) for its execution
differences and limits. Compilation and alternative convolution arithmetic can change
rounding; a larger batch also changes updates per epoch and the optimization trajectory.

```bash
uv run python train.py recipe=breakout_detached_fast --cfg job --resolve
uv run python train.py recipe=breakout_detached_fast \
  trainer.frame_cache=data/breakout-676ff638-lz4
```

These commands start a new full training run, not a continuation of the diagnostic
checkpoint. Keep the recorded-context float32 held-out evaluation and generated-rollout
probes enabled. The default diagnostics record health every 100 updates and held-out
32-frame probes every 1,000 updates. `best.pt` is still selected by held-out one-step
RGB MSE, so inspect rollout and ball-region metrics before choosing a playback model.
The short detached-feedback comparison improved both batch-sequence repeats, but does
not establish the quality or stability of this full ten-epoch experiment. See
[the experiment evidence](history.md#detached-feedback-at-the-horizon-8-transition-2026-09-16).

Checkpoint metadata and saved recipes preserve `detach_feedback`. Weights checkpoints
do not contain Adam state; replaying a recipe restarts training and is not an exact
optimizer resume; use `--resume resume.pt` for new runs with full training checkpoints.
Use a source snapshot or image containing the new recipe when
launching remotely; the earlier verified container image alone does not contain it.

## Linear ball-position probes

`probe_ball_latents.py` fits two independent affine readouts of a frozen
checkpoint's complete final encoder feature map: current ball position (last
recorded input frame) and next ball position (the prediction target). The input
includes the checkpoint's action history. It uses recorded native state labels,
not coordinates inferred from predicted images. Missing reset labels and the
native `ball_y=0` sentinel are excluded from fitting and scoring; occluded balls
with recorded coordinates remain included.

```bash
uv run python probe_ball_latents.py CHECKPOINT \
  --dataset /path/to/pinned/dataset/snapshot \
  --frame-cache /path/to/verified/frame-cache \
  --output logs/ball-position-probe --device cuda
```

The default experiment streams every training window twice and every held-out
window once; features are not retained as a large matrix. Train-only feature
normalization is fixed before fitting. The two output pairs share an inference
pass but have independent linear coefficients and target masks. Encoder weights
remain frozen in evaluation mode, and extraction uses float32 inference. Saved
artifacts include the checkpoint hash, episode IDs, normalization, readout weights,
training curve, and held-out MAE, RMSE, R², distance quantiles and baseline scores.

Errors use native coordinate units: normalized X multiplied by 160 and normalized
Y multiplied by 255. These are not claimed to be rendered sprite-center errors.
Mean-position baselines use training labels only. Persistence and constant-velocity
baselines use ground-truth past positions, so they are privileged references, not
image-only competitors; comparisons use matching valid subsets. The script uses a
fixed Adam schedule, not a closed-form least-squares optimum. Weak results can
reflect optimization or nonlinear encoding; good results establish linear
decodability, not causal use of ball information by the original decoder. These
are recorded-history probes; they do not establish decodability from corrupted
autoregressive histories.

`--resume PATH` continues after the last completed probe pass, keeping the saved
feature normalization and linear weights. New probe bundles also save optimizer
state. Continuing an older weights-only bundle initializes a fresh optimizer and
records that fact in the manifest. Incomplete passes are repeated. Checkpoint
writes are atomic. The report also compares next-position predictions against
reusing the current readout, and scores the displacement obtained by subtracting
the two readouts.

## Four-frame context and four-step rollouts

```bash
uv run gymemu train recipe=breakout_detached_h4_r4
```

This separate recipe inherits `breakout_detached_fast` and changes RGB history
and action history to four slots, with `rollout_steps=4` and curriculum `[1, 2, 4]`.
Epochs 3–10 use four-step detached rollouts. Dataset revision, model width, Adam,
learning rate, full-resolution losses, float32 evaluation, and 32-step probes are
inherited. New runs save full `resume.pt` training checkpoints.

This jointly changes context and training horizon; it cannot identify which change
caused any quality difference. Starting from recorded history, the fifth prediction
is the first with entirely generated context, so retain longer playback probes when
judging this variant. Training throughput settings are measured separately on Beast-3.

Beast-3 measurements retain batch 64, two loader workers and two CPU threads,
with BF16 and compiled automatic layout: about 3,566 training windows/s at the
four-step horizon, excluding full validation and remote uploads. See
[performance measurements](performance.md#four-frame-context-and-four-step-rollouts)
for the sweep and timing scope.
