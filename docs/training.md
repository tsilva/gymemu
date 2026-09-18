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

## Single-frame reconstruction

```bash
uv run gymemu train recipe=breakout_reconstruction
uv run gymemu play /absolute/path/to/run/best.pt --reconstruction
```

The Breakout recipe uses CUDA, bfloat16 training, compilation, and the lossless
frame cache. Set `trainer.frame_cache` to the existing cache directory. Validation
always runs in float32. For other devices, start with `approach=reconstruction`
instead of the CUDA-tuned Breakout recipe.

This recipe trains only the existing convolutional frame codec, using ordinary
float32 RGB MSE between each recorded image and its reconstruction. The codec
receives one image, with no action, frame history, dynamics model, auxiliary loss,
or latent regularizer. The width-32 codec produces a 32×27×20 latent map from a
210×160 RGB image. Padding preserves the original image dimensions after decoding.

Training uses frames referenced by training episodes, including their initial
frames. Every epoch evaluates the held-out episode split without fitting it.
The default recipe evaluates that full split. The lowest validation reconstruction
MSE selects `best.pt`; `stages/representation/epoch-XXXX.pt` retains every evaluated
epoch with its scores. `last.pt` is the last completed epoch, regardless of score.
The evaluation contract names `reconstruction_rgb_mse`, so comparison never groups
these scores with next-frame prediction scores. Train/validation loss curves can
reveal a generalization gap; reconstruction quality alone does not establish that
the latent map preserves every variable needed for future use.

Reconstruction checkpoints open paused in reconstruction mode automatically.
Original, Reconstruction, and Diff show the same recorded timestep, including
frame zero. Space advances one recorded frame; Tab plays or pauses. Reset, episode
selection, and seeking always reconstruct the newly displayed recorded frame.
There is no generated-frame feedback. History editing and autoregressive playback
are unavailable for these checkpoints. A latent-pipeline checkpoint can also use
`--reconstruction` to inspect its codec independently.

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

New runs use metrics schema v3. After each completed epoch, W&B logs:

- `eval/mse`: the final stage's full validation RGB MSE, published once.
- `train/<stage>/loss/epoch` and `eval/<stage>/loss`: stage objectives. These remain
  separate from the comparison score because an objective may include other losses.
- `train/<stage>/<metric>/epoch` and `eval/<stage>/<metric>`: sample counts, batches,
  elapsed seconds, and additional stage measurements, excluding duplicate loss/MSE aliases.
- `train/<stage>/curriculum/*`: stage-specific curriculum settings.
- `train/epoch`, `train/stage`, `train/objective`, `train/lr`, `train/samples`,
  `train/rate`, and `eval/rate`: progress and throughput.

`eval/mse` uses `eval/step`; training and fixed probes use `train/step`.
Both axes count cumulative optimizer updates across stages. New runs do not emit
`evaluation/*`, `stages/*`, `optimizer_steps`, or `train_samples_seen` aliases.
Historical runs retain their original charts and data. W&B also collects its
standard system metrics.

The run configuration records the score's meaning in `evaluation.metric`, such as
`reconstruction_rgb_mse` or `next_frame_rgb_mse`. Filter by that field when comparing
runs. Validation remains float32, and best-checkpoint selection is unchanged.
The W&B summary mirrors `summary.json`, including `best_mse` and the training
budget. Local `metrics.jsonl` and checkpoint metadata retain their full stage
records independently of the W&B naming scheme.

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
| `stages/<name>/epoch-XXXX.pt` | Every evaluated epoch, with its validation scores |
| `stages/<name>/latest.pt` | Periodic stage snapshot, which may precede evaluation |
| `best.pt`, `last.pt`, `latest.pt` | Corresponding checkpoints from the final stage |
| `resume.pt` | Full training state: weights, Adam state, stage/epoch/batch progress, RNG, metrics, and stage-best weights |

`trainer.checkpoint_seconds` defaults to 600 seconds, or 10 minutes. Checkpoint
replacement is atomic. An interrupted run can lose work after the latest snapshot
and does not get a completed
summary. The playable `best.pt`, `last.pt`, and `latest.pt` files contain weights and
configuration. The separate `resume.pt` also includes training state.
See [approaches.md](approaches.md) for stage selection and version compatibility.

### Stop and resume training

New runs save `resume.pt` every `trainer.checkpoint_seconds` (600 by default), before
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
Switching tabs or windows keeps playback running and clears held keys. Closing the player page pauses playback. Ctrl+C in the terminal stops the server.

The player starts in single-step mode. Tab toggles continuous playback, capped at
30 predictions per second for the 60 Hz Atari simulation with frameskip 2. Held
keys repeat on each tick; the most recently pressed held action key wins. With no
held keys, playback uses action 0, or the first checkpoint action if 0 is absent.
Slow inference lowers the effective rate without catch-up steps. R or C pauses continuous playback and clears held keys. Losing
window focus clears held keys while playback continues. In empty-start mode,
the first tick initializes the frame without executing a game action.

## Browser player workspace

`play.py` serves the player on loopback and opens its browser URL. No JavaScript
installation, build, CDN, or Gradlab installation is needed. Playback opens a player
tab and a Diagnostics tab. `--no-browser` prints both complete URLs for opening
manually or in Codex's in-app Browser. `--port auto`
selects an unused port; `--port NUMBER` selects a specific port. Each server has one
playback session and a random access token shared by its printed URLs. Both tabs
read the same atomic frame/metric revisions from that session, without duplicate
inference. Leaving the player tab keeps playback running; closing it pauses playback; Diagnostics only sends explicit chart selections as seek commands; it sends no
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
steps. A lost browser heartbeat clears held actions while playback continues.

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
the next episode. Reset and episode end pause replay. Focus loss keeps replay running. Keyboard presses
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
## Brick-grid extraction prototype

`prototype_brick_grid.py` extracts a 6-by-18 visible brick matrix from existing
lossless Breakout RGB. It calibrates row colors and geometry on a training frame,
then freezes them. A brick needs at least 75% matching pixels in its 8-by-6 cell;
small patches fitting inside 2-by-4 pixels are treated as ball pixels. Other
partial patches remain unknown. The detector uses RGB alone, not recorded counts.

```bash
uv run --frozen python prototype_brick_grid.py \
  --dataset /path/to/pinned-dataset \
  --cache /path/to/matching-lz4-cache \
  --output logs/brick-grid-prototype/final
```

The default sample covers 64 consecutive transitions from each train shard and
64 from each held-out shard starting at offset 64. The separate held-out prefix
was used for initial exploration; the final offset sample checks the frozen rule.
Outputs include `labels.parquet`, geometry/provenance JSON, a Markdown report, and
RGB/matrix diagnostic images. Sources remain unchanged. `grid` and `state_grid`
flatten in row-major order, top-to-bottom and left-to-right; values are 0 absent,
1 present, and -1 unknown. Pixel-support fractions are not probabilities.

Recorded successor-state brick counts check extracted matrices. If count or
certainty checks fail, `state_grid` is entirely unknown while `grid` retains
visible occupancy. Startup can show an incomplete wall despite 108 internal
bricks; accepting those visible absences as true state labels would be incorrect.
Temporal checks use adjacent steps within episodes and validate frame-ID joins.

The September 17 prototype matched counts on all 10,624 fresh held-out examples
and all 286 sampled single-brick destruction transitions. In 42,496 training
examples, 318 startup frames failed the internal-state count check. This is
sampled count/temporal validation, not independent per-brick ground truth. Later
wall resets were not represented. No full dataset augmentation has been run.
## Brick dataset annotations

The complete annotated dataset is published at revision
`9a22e4c0b6b9796a1358f36a6854f2569cbee0af` of
`tsilva/gradlab-breakout-trajectories`. Existing recipe pins remain unchanged;
select this revision explicitly when consuming the new columns.

`augment_brick_dataset.py` applies the validated detector to all unique RGB frames,
then joins results to each successor state and every episode's initial frame.
It builds a separate output tree and validates every rewritten shard against all
original columns. It does not change images, splits, rewards, actions, or training
filters. The dataset's prior revisions remain usable.

```bash
uv run --frozen python augment_brick_dataset.py \
  --dataset /path/to/pinned-dataset \
  --cache /path/to/matching-lz4-cache \
  --output logs/brick-grid-augmentation --workers 4
```

The output includes `dataset/transitions`, `dataset/episodes`, publication receipts
under `dataset/annotations/breakout-bricks-v1`, and resumable scratch arrays. Resume
requires unchanged code and source identity. Publication is a separate step after
full validation, using the source Hub commit as the required parent.

Transition columns describe the successor frame:

| Column | Meaning |
|---|---|
| `brick_grid` | Six rows of 18 cells: 0 absent, 1 present, -1 unknown |
| `brick_grid_suspect` | At least one quality flag is set |
| `brick_grid_quality_flags` | Integer bitmask with the reasons below |
| `brick_count_visible` | Confidently present cells |
| `brick_count_mismatch` | Visible count differs from recorded native brick count |
| `brick_grid_unknown_cells` | Number of unresolved cells |
| `brick_grid_min_present_support` | Minimum matching pixels among present cells, out of 48; 48 if none |
| `is_initial_brick_layout` | Frame belongs to the episode's initial wall animation |
| `source_is_initial_brick_layout` | The input/source frame belongs to that animation |
| `source_brick_grid_suspect` | Source state's grid has a quality issue |

Episode tables have corresponding `initial_brick_*` columns and
`initial_is_brick_layout`. Initial native counts are unavailable, so
`initial_brick_count_mismatch` is null and the count-unavailable bit is set.
Matrices retain their visual values even when suspect; flags never overwrite
the inferred pattern to force agreement with a native count.

Quality bits are: 1 unknown cell, 2 count mismatch, 4 unexpected wall color,
8 partially supported present brick, 16 unexplained brick reappearance,
32 unvalidated later-wall reset, 64 native count unavailable, 128 initial layout.
Temporal reappearance checks mark both endpoints. Startup is the initial prefix
of a fresh full-wall episode up to the first complete visible wall or evidence
of gameplay. It can contain decreasing visible brick counts. It never restarts
after life loss or on later walls. This is a documented visual/state heuristic,
not a native animation flag.

To exclude animation from next-frame pairs, reject rows where either source or
target layout flag is true. For longer frame histories, check every included
state's flag, including the episode's initial frame. No automatic exclusion is
enabled by this annotation work.


## Standalone paddle history experiment

`probe_paddle_history.py` trains compact paddle position/native-velocity probes
from RGB-derived paddle features and independently sized executed-action histories.
It keeps training, validation, and official held-out episodes separate and writes
standalone probe checkpoints, not emulator checkpoints for the player.

```bash
uv run python probe_paddle_history.py prepare
uv run python probe_paddle_history.py train --categorical --samples 0 --epochs 80 --contexts 1:4,2:2,2:4
uv run python probe_paddle_history.py evaluate --checkpoints logs/paddle-history-20260917/next-f1-a4-s47-n0-e80-w128-cat.pt
```

Defaults use the locally cached `676ff6388f4218d3c3a3ce9f2f33e075fa7314a3`
recording and its verified LZ4 cache. Override `--dataset`, `--frame-cache`, and
`--output` for other local paths. `--current` predicts the newest observed state;
the default predicts the successor. Import `infer_rgb` from the script to run a
saved probe on oldest-to-newest uint8 CHW RGB history and executed actions. Read
[the experiment results](history.md#2026-09-17-trained-paddle-state-history-probes)
for alignment, measured context tradeoffs, and the limits of the conclusion.

### Full RGB paddle probe

`probe_paddle_rgb.py` repeats the next-state experiment with the entire recorded
210×160 RGB image. Its `paddle_state_cnn` model uses the exact `FrameCodec.encoder`
architecture: three stride-2 convolutions with channels 3→32→64→32, kernel 4,
padding 1, and ReLU after the first two. Images are divided by 255 and padded
at the bottom to 216 pixels, as in the codec. Each historical image passes through
the same encoder. The regression head combines their learned features and
independently sized one-hot action histories.

```bash
uv run python probe_paddle_rgb.py prepare
uv run python probe_paddle_rgb.py train --contexts 1:4,2:2,2:4 --samples 512 --validation-samples 1024 --epochs 80
uv run python probe_paddle_rgb.py evaluate --checkpoints /path/to/probe.pt
```

Preparation reuses the feature study's episode and frame identities, then reads
`paddle_x_normalized` and `paddle_vx_normalized` directly from the original
recordings. Extracted paddle features are never inputs to this probe. The latest
input image is frame t, the last included action is action t, and the target is
state t+1. Native-unit reporting multiplies both normalized outputs by 160;
velocity is pixels per native emulator tick, not captured-frame displacement.

`--position-head` selects `paddle_position_cnn`, an explicit alternative readout
with the same CNN. A learned 160-class position head receives auxiliary
cross-entropy supervision from the recorded position of each observed frame.
Those labels are training targets only. At inference, learned position
probabilities and actions predict the next state. `heads --resume-checkpoints
/path/to/position-probe.pt --head-epochs 80 --contexts 1:3,1:4,2:2,2:4` freezes this
learned perception in evaluation mode and compares temporal heads efficiently
using cached CNN probabilities. Every saved model still accepts full RGB.

Training initializes from scratch unless `--resume-checkpoints` is explicit.
It does not load the existing autoencoder weights, whose training episodes overlap
this probe's validation set. `--fp32` disables encoder autocast and TF32; final
evaluation always uses float32 without TF32. These are standalone state probes,
not emulator checkpoints accepted by the player. Generated data, launch receipts,
checkpoints, and metrics remain under `logs/paddle-rgb-20260917/`.

### Direct current paddle state

For state estimation, align the output with the newest observed image. This is a
different objective from the successor-state experiment above. The plain
`paddle_state_cnn` can output all three recorded normalized paddle variables
directly, without a position-classification head or transition model:

```bash
uv run python probe_paddle_rgb.py prepare --include-width --output logs/paddle-current-rgb-20260917
uv run python probe_paddle_rgb.py train --current --output logs/paddle-current-rgb-20260917 --contexts 1:0,1:4,2:0,2:1,2:4 --samples 256 --epochs 20
```

The output order is `paddle_x_normalized`, `paddle_vx_normalized`,
`paddle_width_normalized`. The last input frame and the target describe state t;
the action history ends at action t−1, which produced that frame. Recorded states
serve only as targets. Current position and width are visible, while current
native velocity may require temporal or action information.

The dataset normalizes x and velocity by 160, but width by 16. In this recording,
width targets are 1.0 and 0.75, corresponding to 16 and 12 pixels. Preparation
checks this conversion against the recorded native width. The model outputs the
original normalized values. Training weights squared normalized errors by their
native scales, so each native-unit error contributes equally. Reports retain both
normalized and native MAE, including separate results for each width.

### Joint current paddle and ball state

`--include-ball` prepares seven targets directly from the normalized dataset fields
and includes paddle width automatically:

```bash
uv run python probe_paddle_rgb.py prepare --include-ball --output logs/paddle-ball-current-20260917
uv run python probe_paddle_rgb.py train --current --contexts 2:1 --output logs/paddle-ball-current-20260917 --epochs 24
```

The output order is paddle x, paddle vx, paddle width, ball x, ball y, ball vx,
and ball vy. Each uses the corresponding existing `_normalized` field. Preparation
does not infer labels from RGB or calculate velocity labels from displacement.
The native reporting scales are `[160, 160, 16, 160, 255, 2, 3.375]`; ball velocity
scales differ from paddle velocity, and ball y uses 255 rather than image height.
The source conversion is checked against native records while preserving the
original normalized training targets.

The same CNN and frame/action history feed a seven-output direct regression head.
`--expand-outputs --resume-checkpoints /path/to/paddle-model.pt` retains existing
paddle output rows and initializes new rows, then trains all weights jointly.
It does not freeze the paddle encoder or add an auxiliary classifier.

`--loss-weighting native` retains the earlier equal-native-unit residual weights.
`normalized` uses unweighted MSE on the recorded normalized values. `standardized`
weights each residual by the inverse standard deviation of that normalized target
in the sampled training set. These options change loss weights, not stored labels
or model output representations. Reports include each target's normalized/native
MAE and tails. Evaluation also separates cases with changed recorded ball velocity,
unchanged recorded ball position, and ball y outside the RGB canvas; these are
diagnostic slices, never exclusions or model inputs.

`--regression-loss huber` retains the same weighted quadratic loss for errors
below one native unit and uses linear growth beyond that point. All labels and
examples remain included. `--selection worst-native-mae` selects the checkpoint
with the lowest maximum per-output native MAE, useful when the goal is accuracy
on every variable and a few coordinate outliers dominate squared error. The
default remains MSE training and native-MSE checkpoint selection for reproducing
the earlier experiments. Each run records its loss and selection settings.

## Single-ball state dynamics

`gymemu dynamics` isolates dynamics from perception and rendering. The input is
recorded normalized ball x/y/vx/vy, paddle x/width, a 6-by-18 binary brick grid,
and requested policy actions. The outputs are the next state and a terminal
probability. Lives, score, serve phase, RGB, frame IDs, and episode IDs are not model
inputs. Lives are used only to identify loss boundaries. This experiment has its
own state metrics and is not ranked against RGB approaches by pixel MSE.

New configs set `model.predict_paddle_velocity=false`. Both MLP and GRU exclude
that input and output, its training loss, and its reported prediction metrics.
Context probes also exclude velocity from the retained paddle history. The cache
keeps its original 115-slot layout for compatibility; slot 5 is a reserved zero in
new generated states, not a velocity estimate. Recorded dataset columns remain
unchanged. Older checkpoints without the option retain their original behavior.
Use `model.predict_paddle_velocity=true` only to reproduce legacy dynamics fits.
Isolated probes omit velocity from their default targets; an explicit legacy
velocity target also requires `--legacy-paddle-velocity`.

The read-only adapter uses annotated dataset revision
`9a22e4c0b6b9796a1358f36a6854f2569cbee0af`. It joins successor labels to the next
state and uses the following row's requested action for that state's outgoing
transition. Missing reset labels never become fabricated states. The adapter
validates the original frame chain using IDs, never treating IDs as offsets.

A lives decrease or active-to-zero ball-y transition ends a logical single-ball
segment. A loss followed by a serve inside one frameskip also ends the segment.
The new active successor can independently begin another segment. Waiting/startup
states and suspect brick annotations are excluded. Quality gaps and recording
truncation censor segments but do not become death labels. No context or training
rollout crosses a segment boundary. Split assignment happens at the original
episode level, before segmentation. Training and validation come from disjoint
original training episodes; the original held-out split supplies the reserved test.

Prepare a reusable cache, without downloading or decoding image assets:

```bash
uv run gymemu dynamics prepare cache=data/state-dynamics
# Or use the existing local annotated snapshot:
uv run gymemu dynamics prepare \
  dataset=/absolute/path/to/annotated-dataset cache=data/state-dynamics
```

Null episode counts select the available data, reserving 20% of original training
episodes for validation. For a bounded experiment set `data.train_episodes=128`,
`data.validation_episodes=32`, and `data.test_episodes=64`. The cache saves rules,
original episode IDs, segment boundaries, and the dataset manifest hash. Its
`manifest.json` is complete only after every split is written; incomplete caches
cannot train. Cache and run destinations must be new directories.

Train the one-state baseline, or run the context/capacity/memory search:

```bash
uv run gymemu dynamics train cache=data/state-dynamics output=runs/state-baseline
uv run gymemu dynamics search cache=data/state-dynamics output=runs/state-search
```

Hydra composes `configs/state_dynamics.yaml` with the dataset identity and starting
state selection in `configs/game/breakout_state.yaml`. Dotted overrides reject
unknown keys. `model.history` counts observed states including the current one;
`model.past_actions` counts prior requested actions and excludes the always-present
current action. Histories are ordered oldest to newest and left-padded with explicit
state validity and a separate non-action token. State and action lengths are
independent. Each model receives the same eligible source/target identities.

The search first fits all configured context combinations over the configured
seeds. It selects contexts by mean validation score, then compares MLP widths and
GRUs over the same seeds. Finalists include generated-state training. The winning
configuration is selected by mean seed score, with parameter count breaking exact
ties; the best validation seed supplies the final checkpoint. Only then does the
search evaluate that checkpoint on the reserved test split. Results identify the
best tested configuration within the saved budget, not a universal optimum.

The MLP predicts residual motion. Brick/width heads start with a persistence bias;
all transitions and collision effects remain learned. The GRU processes the warm-up
context and then carries recurrent memory through generated steps. Brick and width
feedback commits discrete values in both training and inference. Training uses a
straight-through gradient for these decisions. Signed velocity outputs have no
sigmoid or `[0,1]` clamp. Motion losses use documented native-unit tolerances,
not an assumption that every normalized field has the same scale.

Training begins with one-step supervision, then uses configured rollout horizons.
Every valid generated step receives the corresponding recorded target. Ground-truth
segment boundaries control masks: predicted termination never hides later training
losses. Terminal transitions supervise only termination; next-state losses are
masked. Brick change weighting supplements BCE on every cell, including false
appearances. Termination BCE uses a training-only positive weight capped at 100.
Rollout stages use half the initial learning rate and begin from the best validation
checkpoint, including its matching optimizer state. Every evaluated epoch is saved.

Validation reports native-unit per-field errors, collision and brick-change subsets,
brick-removal precision/recall, spurious reappearances, and termination confusion
counts. Collision subsets are label-derived velocity-change diagnostics, not native
collision event annotations. Threshold selection maximizes validation terminal F1,
then prefers fewer false positives. Test evaluation reuses the saved threshold.

The fixed selection score is the mean across configured available rollout horizons
of capped ball-position MAE, capped paddle-position MAE, and 20 times brick-cell
error rate, plus 20 times early-terminal rollout fraction and 10 times terminal
F1 error. Ball MAE is capped at 100 and paddle MAE at 50 only in this ranking score;
unclipped errors remain in the report. Inspect event-specific results before calling
a model playable. State rollout metrics deliberately continue diagnostic prediction
after a predicted early stop so premature termination cannot conceal error.

Each run saves resolved configuration, the data/segmentation manifest, per-epoch
metrics and checkpoints, the selected checkpoint, and a summary. Checkpoints load
with `weights_only=True` through the fixed state-model registry. No Python target
from checkpoint metadata is executed. This experiment writes local artifacts; it
does not start W&B sessions or upload checkpoints automatically.

Evaluate or replay a state checkpoint:

```bash
uv run gymemu dynamics evaluate cache=data/state-dynamics \
  checkpoint=runs/state-baseline/best.pt output=logs/state-validation
uv run gymemu dynamics play cache=data/state-dynamics \
  checkpoint=runs/state-baseline/best.pt output=logs/state-playback \
  playback.start_index=0 playback.steps=128
```

Playback uses recorded requested actions with recursively generated state and stops
on predicted termination. It also stops at the source segment/step limit because
recorded actions beyond that segment are not part of this task. The JSON distinguishes
those stop reasons and includes aligned reference states for diagnosis. It renders
no images. `StatePlayer` can also be driven by fresh requested actions; it rejects
steps after termination. There is no respawn, serve-phase estimator, or lives output.
A new player instance starts a new scene. `checkpoint=...` on `train` initializes
fine-tuning in a new directory; it is not an exact interrupted-run resume command.

## Isolated state targets and input sufficiency

Before combining state losses, audit identical inputs for contradictory next-state
labels and train each target independently:

```bash
uv run state_observability.py --cache data/state-dynamics-20260917 \
  --output logs/state-observability-new --contexts 1:7,8:7,16:15,32:31,full
uv run state_probes.py --cache data/state-dynamics-20260917 \
  --output runs/state-isolated-new --contexts 1:7,8:7 --epochs 20 --samples 100000
```

Context notation is observed states followed by past requested actions. The current
action is always included. `full` hashes the entire observed prefix since the current
virtual segment began. The audit reads training data only. It reports contradictions
per target, explicit witness episode/step pairs, empirical error floors, and a
separate count for examples with fully available context. A contradictory pair
proves that exact prediction from those inputs is impossible. No matching
counterexample does not prove that a finite history is sufficient.

Each isolated run has independent weights, one selected loss, and no generated
feedback. Scalar motion targets use their native-unit squared errors with the same
per-field tolerance as the state experiment. Paddle width, brick occupancy, and
terminal events use their respective BCE objectives. Brick occupancy is one target
family with 108 output cells; this does not isolate its individual cells. Choose a
subset with `--targets ball_y_normalized,terminal`. Every run saves its target,
configuration, seed, cache/evaluation identities, checkpoint, training diagnostic,
and validation metrics. The command does not evaluate the reserved test split.

Checkpoints use `gymemu-state-probe-v1` and are diagnostics, not playable joint
emulators. Scalar checkpoints minimize validation native RMSE. Terminal and brick
checkpoints maximize validation terminal/removal F1; width uses validation BCE.
The terminal threshold is chosen on validation only and reused for the training
sample diagnostic. High width/cell accuracy is not sufficient: inspect width-change
accuracy and brick-removal precision/recall. Failed isolated fits do not prove
missing information; use exact conflicts and small training-set fit checks to
separate that hypothesis from optimization and sampling problems.

The optional retained paddle context addresses an input problem found in the audit:
startup brick quality had excluded valid paddle observations from history. It reads
paddle x/vx/width and requested actions from the original records, with explicit
validity masks. It preserves history through brick-only quality gaps but clears it
at life loss, a lifecycle reset, or missing paddle measurements. No reset labels are
invented. It uses no future observations and does not change target eligibility:

```bash
uv run state_probes.py --cache data/state-dynamics-20260917 \
  --output runs/state-retained-paddle-new --contexts 1:7 \
  --targets paddle_x_normalized,paddle_vx_normalized \
  --paddle-context 32 --epochs 20 --samples 100000
```

This explicitly selects the registered `state_paddle_context_probe` model. Its
inputs combine the ordinary context with the additional paddle observations,
validity flags, and actions. It verifies dataset-manifest provenance and equality
between recovered current paddle values and the existing cache. The original
state cache and joint training/playback contract remain unchanged. These probes
run on CPU. Results and limitations are in
[the experiment history](history.md#2026-09-17-loss-scaling-and-isolated-state-targets).


### Event diagnostics and longer isolated fits

Every isolated fit now saves `event_diagnostics` and `training_event_diagnostics`
in `summary.json`. These compare native scalar errors with constant motion during
ordinary flight, velocity changes, brick changes, inferred paddle/wall collisions,
paddle reversals, and width changes. Event groups overlap. Wall/paddle labels are
geometric proxies inferred from recorded transitions, not native collision flags.
Terminal successor states remain excluded from all state-error groups. Terminal
classification and brick-removal errors are reported separately. Event labels are
used for evaluation or sampling, never as predictor inputs.

To repeat the corrected-context convergence experiment:

```bash
uv run state_probes.py --cache data/state-dynamics-20260917 \
  --output runs/state-convergence-new --contexts 8:7 --paddle-context 32 \
  --targets paddle_x_normalized,ball_x_normalized,ball_y_normalized \
  --epochs 60 --samples 100000
```

The command saves `best-through-020.pt`, then equivalent snapshots every twenty
epochs, so a single run supports matched budget comparisons. Each snapshot is the
lowest validation-score checkpoint reached by that budget, not necessarily the
weights at its final epoch. `best.pt` remains the overall selected checkpoint.

`--sampling balanced_events` draws half of each training epoch from transitions
with a velocity change, brick change, or terminal event, and half from the rest.
Rare events repeat when necessary. Nonterminal target models continue excluding
terminal transitions. Validation sampling and its episode identities are unchanged.
This deliberately changes the training objective's effective event weights; it is
not importance-corrected sampling. Compare it with uniform sampling from the same
initial checkpoint and budget before adopting it. The first experiment improved
collision errors but worsened aggregate ball-y accuracy.

`--learning-rate` controls AdamW's learning rate. For matched continuations, the
Python `train_target(..., initial_checkpoint=path)` API checks target, model config,
and cache identity, records the checkpoint hash, and initializes a fresh optimizer.
This is fine-tuning, not exact optimizer-state resume. Development event diagnostics
reject the reserved test split. See the
[convergence investigation](history.md#2026-09-17-event-errors-and-isolated-convergence).


### Reconstructed internal-state inputs

The `state_hidden_probe` registry entry adds five inputs to the isolated context
MLP: controller charge, controller measurement, input-repeat state, held-input
state, and cumulative paddle-hit count. Each input describes the source state,
before the action being predicted. These experiments measure whether known internal
variables help an isolated predictor. They are not joint emulators and do not
provide a way to obtain ground-truth internal values during learned playback.

Build the separate input cache with the pinned native Rust source snapshot used
by the audit:

```bash
uv run python -m gymemu.state_hidden_context \
  --cache data/state-dynamics-20260917 \
  --output data/state-hidden-new \
  --native-source logs/state-event-diagnosis-20260917/native-source.rs
uv run state_probes.py --cache data/state-dynamics-20260917 \
  --output runs/state-hidden-new --contexts 8:7 --paddle-context 32 \
  --targets paddle_x_normalized --hidden-inputs data/state-hidden-new \
  --hidden-mode controller --epochs 60 --samples 100000
```

The builder reads training/validation episode metadata and actions. It reproduces
the pinned controller from reset, verifies every recorded paddle position/velocity,
and saves its state before each current action. Controller memory persists across
life loss. The separate hit counter includes only earlier inferred paddle bounces,
is capped at twelve, and resets at segment starts with verified initial-serve
velocity. A non-serve segment start fails preparation instead of guessing the count.
The original dataset and ordinary history boundaries stay unchanged.

The input-cache manifest binds dataset/state-cache identities, source hash, feature
scales, verification counts, and array checksums. Loading rejects changed arrays or
an incompatible state cache. The fixed source snapshot hash prevents a different
native version from silently receiving the same provenance label.

`--hidden-mode none` supplies zeros in the same five positions, preserving model
parameter count. `controller`, `hits`, and `both` enable the corresponding inputs.
All modes receive the same history and action inputs. Use matched seeds, training
samples, budgets, and validation targets for comparisons. Continuation experiments
also copy the same prior weights and initialize added first-layer columns to zero,
so predictions initially agree within float32 precision. New input weights remain
trainable. Their initial equality does not imply equal behavior after training.

See [the internal-state experiment](history.md#2026-09-17-known-internal-state-inputs)
for the completed comparisons and the separate exact paddle-input sufficiency check.


### Paddle controller dataset columns

Build a dataset version containing the four reconstructed paddle controller values:

```bash
uv run python -m gymemu.augment_paddle_dataset \
  --dataset logs/brick-grid-augmentation-20260917-v1/dataset \
  --output data/breakout-paddle-controller-20260918 \
  --native-source logs/state-event-diagnosis-20260917/native-source.rs
```

`--frames /path/to/existing/frames` optionally links unchanged image assets when
the input is a local table-only dataset. The output must not exist. The build uses
a `.incomplete` directory and renames it only after all validation passes. It does
not upload to the Hub or modify the source dataset.

Each transition has `paddle_charge`, `paddle_measure`, `paddle_repeat`, and
`paddle_held` for its successor state, aligned with `record_json`. The same names
with `source_` describe the state before its action. Episode tables include the
four `initial_` values after reset no-ops and before the first recorded action.
Charge, measurement, and repeat are int32; held is boolean. Divide by
`[3856, 235, 60, 1]` when the existing normalized diagnostic input format is needed.

Reconstruction starts from known reset values, reproduces the seeded reset no-ops,
and applies recorded executed actions using the pinned native controller. It
retains controller memory across life losses. The pinned final-life frame-stop
rule uses preceding source labels; current successor labels only verify replay.
This path requires the original fixed frameskip-2, sticky-0 collection contract.
Every successor paddle position and velocity must match exactly. A missing,
duplicate, out-of-order, or incomplete episode stops the build. Each output file
is read back and compared against every original column and every new value.

`annotations/breakout-paddle-controller-v1/` contains schema/timing information,
source and output checksums, pinned code, and validation counts. Earlier brick
receipts remain historical records of the input dataset; the new receipts describe
the rewritten files. No paddle-hit count is added. Train and held-out memberships
stay unchanged. Annotation is deterministic and fits no model on either split.
Existing dynamics loaders ignore the new columns until an explicit input adapter
uses them; creating this dataset does not change existing models or checkpoints.

### Compact paddle learning with sufficient inputs

`gymemu.paddle_transition_training` isolates next paddle x using only current x,
controller charge/measurement/repeat/held values, and the current requested action.
It uses the verified internal-input cache. Ball state, bricks, paddle velocity,
width, and all history are omitted. Life-loss targets remain excluded. All eligible
training transitions are visited once per epoch; complete validation episodes
remain separate from training.

```bash
uv run python -m gymemu.paddle_transition_training \
  --cache data/state-dynamics-20260917 \
  --hidden data/state-hidden-inputs-20260917 \
  --output runs/paddle-compact-new \
  --encoding hybrid --objective classification --epochs 100 --seed 2026
```

The registered `paddle_transition_mlp` uses two SiLU hidden layers by default.
`scalar` encoding scales the five state values and appends a one-hot action.
`hybrid` also supplies ordinary binary digits of the same five integers. This adds
no information or controller rules; it changes the input representation. The bit
widths are 8, 12, 8, 6, and 1. The successful model has 43 input features and 25,111
parameters with width 128 and two hidden layers.

`regression` predicts a native-pixel displacement using MSE. `classification`
selects an integer displacement with cross-entropy and argmax. Displacement classes
are determined from training targets only; this cache contains -11 through +11.
Validation classes outside that range cause an error, rather than clipping labels.
Reported exact-pixel accuracy rounds regression output to the nearest integer;
raw regression MAE and rounded MAE are saved separately. Classifier output is
already integer-valued.

Training uses AdamW without weight decay, native-unit losses, gradient clipping,
and cosine learning-rate decay. The selected checkpoint minimizes validation
integer errors, then raw RMSE. Run configs record all settings, input identities,
and internal-state provenance. `best.pt` uses `gymemu-paddle-transition-v1` and is
loaded through a fixed registry kind. It is not compatible with the joint player.
Actual inference is `model.delta(model(source))`, followed by adding current x.
It never executes the native controller calculation.

`--controller-fields charge` restricts the predictor to current paddle x, charge,
and requested action. `--controller-fields charge repeat held` removes only
measurement. Omitting this option retains all four controller fields, including
for older checkpoints. Input selection happens inside the registered model before
scalar/binary encoding; omitted fields cannot affect its output. The checkpoint
stores the selected fields. `repeat_unsaturated` evaluation separately reports
cases where the repeat counter is below 60, which overall accuracy can obscure.
For this pinned controller, reset no-op counts 1 through 30 and two native frames
per executed action, exhaustive reachable-state checks confirm that x plus charge
is sufficient to determine both next x and next charge. This includes startup
under that contract. The position probe only learns next x; it still requires
correct current charge. See [the minimum-input results](history.md#2026-09-18-minimum-controller-inputs-for-paddle-position).

The default data-loading path permits train/validation only. After model selection
is frozen, a separate internal cache can be prepared with
`python -m gymemu.state_hidden_context ... --test-only`; explicit
`load_data(..., 'test', allow_test=True)` enables final evaluation. These test values
are correct source-time inputs, not model-predicted controller state. The native
reconstruction accounts for a last-life action ending after one native frame, using
only preceding source ball/life labels. Ordinary nonterminal predictions advance
the configured two frames.

The [sufficient-input experiment](history.md#2026-09-17-accurate-paddle-learning-from-sufficient-inputs)
reports the controlled encoding/objective comparison, two classifier seeds, and
frozen-checkpoint final-test results. This establishes accurate one-step paddle-x
learning with known controller state; it does not train controller-state updates,
ball dynamics, or an end-to-end emulator. The later charge experiment below tests
the remaining paddle update separately.

### Learning the paddle charge update

The compact training path also supports `--target paddle_charge`. It reads source
and successor charge directly from the annotated transition tables, joins by
original episode ID and step, checks provenance and file hashes, and excludes the
same life-loss targets as the position experiment. Current inputs are still x,
charge, and requested action. No native controller code runs inside the predictor.

```bash
uv run python -m gymemu.paddle_transition_training \
  --cache data/state-dynamics-20260917 \
  --hidden data/state-hidden-inputs-20260917 \
  --annotations data/breakout-paddle-controller-20260918 \
  --target paddle_charge --controller-fields charge \
  --encoding hybrid --objective classification \
  --epochs 100 --seed 1591 --output runs/paddle-charge-new
```

The classifier predicts a charge change, which is added to current charge. Its
ordered classes come from training targets only and are stored in the checkpoint
as `delta_values`; an unseen validation class raises an error. This dataset has
11 classes: -120, -60, -9, -5, -1, 0, 1, 5, 9, 60, and 120. Charge metrics use
native charge units and exact-integer accuracy, not pixels. Checkpoint config
stores the target and annotation provenance. Older contiguous position-class
checkpoints remain supported.

Both tested seeds achieved zero errors on all 88,590 validation transitions.
The frozen position and charge models were then evaluated together from 256
validation starts, feeding both outputs back for up to 128 steps. Each seed made
zero charge errors across 30,909 scored predictions; the position model made one
transient one-pixel error. Recorded actions and reference segment boundaries are
used, with correct x/charge supplied only at initialization. This evaluates the
paddle subsystem; it does not include learned ball, brick, or terminal dynamics.
The full method and limits are in the [charge results](history.md#2026-09-18-learned-charge-updates-and-removal-of-paddle-velocity).

### Discrete horizontal-velocity probes

The registered `ball_velocity_mlp` predicts only the next horizontal ball
velocity. Its source values are current ball position and velocity, paddle
position and width, controller charge, previous paddle-hit count, fractional
vertical position, brick-contact memory, and the brick layout. The eight
nonterminal output classes are -2, -1.5, -1, -0.5, 0.5, 1, 1.5, and 2. A matched
regression variant predicts an additive velocity change. Paddle velocity is not
an input.

The 2026-09-18 diagnostic campaign keeps reconstructed collision memory in RAM;
it does not modify datasets or state caches. Reconstruction starts with the known
integer ball position at episode reset. The fractional vertical position persists
across life loss, even though training examples and model history still stop at
each life boundary. Replay uses prior/current state, never the target to choose
the current memory. Validation successors check the calculation and score the
model. Both formerly ambiguous validation examples resolve to a 5/8-pixel
fraction. This establishes input sufficiency on the checked train/validation
transitions, not an accuracy guarantee for other environment configurations.

The local experiment driver is
`logs/ball-vx-discrete-20260918/train.py`; `prepare.py` in the same directory
reconstructs source inputs without persisting feature arrays. The campaign uses
the existing 128 training and 32 validation episodes, full training sampling,
two hidden layers of 128 SiLU neurons, and 100 epochs per fit. It compares
collision-memory masking, categorical versus regression targets, and scalar
versus scalar-plus-bit inputs. Runs live under `runs/ball-vx-discrete-20260918/`.
Best checkpoints minimize validation native-velocity MSE; the reserved test is
not used. Do not run this driver over an existing output directory.

Two follow-up drivers, `train_brick_augmentation.py` and `train_relative.py`, use
the same splits, seeds, epoch budget, and checkpoint selection. The first replaces
brick-layout inputs with randomly chosen training layouts only when source RAM y
is above 100, too far below the bricks for a contact in two native frames. The
native diagnostic verifies target invariance on both checked splits. This
augmentation happens only in training RAM; validation uses the original inputs,
and no dataset records are added or changed. Its separate random generator keeps
the control's training order intact. The second adds an optional ball-to-paddle
horizontal offset encoding to that same augmented fit. Hybrid offset models have
195 encoded inputs and 42,632 parameters, compared with 182 and 40,968 without it.

These are recorded-source, one-step probes. They do not predict the other game
variables or establish recursive simulation accuracy. Reconstruction code is
diagnostic preparation, not part of neural playback. Dataset additions still
require the user's approval.

### Focused paddle-region velocity probes

The local drivers in `logs/ball-vx-paddle-focus-20260918/` select eligible
nonterminal sources with RAM ball y between 160 and 183 inclusive and positive
vertical velocity. Selection never uses whether the successor contains a hit.
This retains all 6,336 training and 1,519 validation paddle hits, plus approaches
without a hit: 30,204 training and 7,164 validation examples in total. The existing
episode split and life boundaries remain unchanged; the reserved test is unused.

Two model contracts use the same selected examples: the previous 118-value input
with training-only brick-layout augmentation, and `paddle_only=True` with nine
values (ball x/y/vx/vy, paddle x/width, charge, prior hit count, fractional y).
Both use hybrid encoding, derived horizontal offset, two 128-unit SiLU layers,
and eight velocity classes. Compact encoding has 85 inputs and 28,552 parameters;
the full input has 195 and 42,632. Native diagnostics verify that removing bricks
and contact memory cannot change the checked targets in this region. Native
rules remain outside neural inference.

`train.py` screens both contracts with seeds 91 and 2026 for 100 epochs. This
provides only 3,000 optimizer updates, compared with the global probe's 35,100.
`train_stepmatched.py` trains fresh models for 1,170 epochs to match that update
count, stretching the cosine schedule accordingly. Matching updates does not
match unique-example coverage or validation-selection opportunities. AdamW,
learning rate 0.001, weight decay 0.0001, batch size at most 1,024, gradient clip
5, and minimum-validation-MSE checkpoint selection are otherwise retained.

Reports separate wrong direction, wrong absolute speed, and their intersection;
these are diagnostics, not separate output heads. They also distinguish native
frame timing and approaches that pass the paddle without a hit. Each saved
checkpoint is reloaded and must reproduce its validation metrics. Source-memory
reconstruction, augmentation, and feature removal happen in RAM without dataset
or cache writes. Results are in the
[research history](history.md#2026-09-18-focused-paddle-region-velocity-learning).
These are one-step velocity probes restricted to the selected region, not complete
state models or evidence of recursive accuracy.

### Isolating horizontal direction

`logs/ball-direction-20260918/train.py` compares the compact eight-class velocity
model with `objective=direction`, a two-class left/right predictor. Both receive
the same nine source values and 85 encoded inputs, two 128-unit SiLU layers,
the same 30,204 training sources, seeds 91 and 2026, minibatch order, optimizer,
and 35,100-update schedule. The direction model has 27,778 parameters versus
28,552 for the velocity model. The output predicts the sign of next vx only.

Both fresh fits select the checkpoint with the fewest direction errors over all
7,164 selected validation sources, keeping the earliest tie. This removes the
previous velocity-MSE selection difference. An additional eight-class decoding
sums probabilities over the four leftward and four rightward velocities, with
its own best checkpoint under the same criterion. The driver stores this decoding
choice; ordinary model `predict()` still selects the most likely velocity class.
The eight-class loss continues to distinguish speeds, whereas the binary loss
does not. Native event labels are used only to report accuracy by event.

The experiment also scores previous MSE-selected checkpoints and the baseline
that preserves incoming direction. It checks source-index identity against the
focused experiment, retains final-epoch metrics as well as selected checkpoints,
and verifies identical metrics after checkpoint reload. Artifacts live under
`logs/ball-direction-20260918/` and `runs/ball-direction-20260918/`. Dataset and
cache files remain unchanged, and no reserved-test or recursive claim follows
from validation direction accuracy.

### Auditing direction-error coverage

`logs/ball-direction-coverage-20260918/audit.py` evaluates both frozen direction
checkpoints on the same selected validation sources. It compares exact source
matches, distances to 1/5/25 training neighbors, and coarse source-geometry
counts. The three distance representations are standardized nine-value state,
standardized relative geometry with combined integer/fractional y, and the
model's hybrid input encoding. All scaling uses training inputs only. Density
thresholds come from 2,048 seeded training paddle-hit anchors, excluding their
entire source episode when finding neighbors. Validation neighbors come only
from the training split. Distances use float64 and check against direct norms.

The audit separately uses the pinned native transition rules to perturb source
ball x in eighth-pixel increments, up to two pixels in each direction. A change
in next-vx sign identifies a local direction boundary. These are diagnostic
counterfactual states, not new training examples or evidence that each perturbed
state is reachable. The scan includes wall interactions and hit/miss changes;
it is not limited to the paddle's direction-selection branch. Model inference
still uses the original source inputs and no native transition rules.

`unseen.py` repeats the density and boundary comparisons after excluding exact
training-state duplicates. The reports include error counts, episode-bootstrap
descriptive intervals, and source/nearest-neighbor witnesses. The same validation
episodes selected the checkpoints, so these associations are exploratory, not
causal proof or an independent test score. Both scripts write only diagnostic
artifacts; dataset and cache files remain unchanged.

### Direction learning with more training episodes

`logs/ball-direction-scale-20260918/train.py` compares nested sets of 128, 256,
and 512 original training episodes with two seeds each. The split seed and
ordering come from the existing state-cache manifest. The first 128 training
episodes and all 32 validation episodes remain identical; expansion uses only
unused episodes from the original dataset's training partition. The held-out
partition and reserved test records are not read.

`prepare.py` reads the existing controller-annotated dataset and applies the
same state adapter and life boundaries in RAM. Source charge comes from the
existing `source_paddle_charge` column. Source hit count, fractional y, and
collision memory use the prior diagnostic reconstruction. Native replay checks
the successors; it does not select current memory using the target. Preparation
requires exact equality of the old 128 episodes' focused inputs, targets, events,
episode IDs, and steps. No dataset or feature-cache files are created or modified.

Each fresh fit uses the compact 85→128→128→2 direction model, seeds 91 and 2026,
35,100 updates, and batches of exactly 1,024 examples. Repeated shuffled passes
span batch boundaries without dropping examples. Every fit therefore presents
35,942,400 examples in total, with more unique examples and fewer repetitions
at larger episode counts. AdamW settings and gradient clipping remain unchanged.
The cosine schedule and validation checks advance every 30 updates for 1,170
blocks. This replaces the previous approximately 1,007-example batches, so the
new 128-episode controls are the primary comparison.

Checkpoint selection minimizes direction errors across the unchanged 7,164
validation sources. Reports include all 1,519 paddle hits and the fixed sparse,
boundary, and previously unseen groups from the earlier coverage audit. Those
groups are evaluation diagnostics and do not affect sampling or the loss.
Checkpoints and reports live under `runs/ball-direction-scale-20260918/` and
`logs/ball-direction-scale-20260918/`. Checkpoint reload must reproduce metrics.
This experiment concerns recorded-source direction prediction; it does not
evaluate speed or recursive simulation.

### Inferring controller state from history

`controller_history_mlp` is an isolated classifier for current charge or current
measurement. Inputs contain H observed paddle positions, native-frame velocities,
widths, validity flags, and the H preceding requested actions. The current action
is excluded because the target is the current internal state. Startup observations
remain available, but history never crosses a life boundary. Hidden values are
supervision only. `gymemu/state_controller_history.py` owns this input construction.

The model combines scaled scalars and binary digits of the same observed integers,
then applies two 128-unit SiLU layers and a categorical output. Output values come
only from the training split. Unseen validation values remain errors, not clipped
labels. The experiment used 8 and 32 observations, 50 epochs, batch size 2048,
AdamW at 0.001 with no weight decay, cosine decay to 0.00005, and seed 2026.
Checkpoint selection minimizes validation exact-integer errors, then native MAE.

The bounded experiment drivers and reports are retained locally under
`logs/controller-history-20260918`: `audit.py`, `train.py`, `downstream.py`,
`audit.json`, `training.json`, and `downstream.json`. Run directories are
`runs/controller-history-{charge,measurement}-h{8,32}-20260918`. Driver output
directories are intentionally new-only. Checkpoints use
`gymemu-controller-history-probe-v1`, tensor-only loading, and the explicit model
registry. No joint-player loader was added for these diagnostic checkpoints.

The [history-inference experiment](history.md#2026-09-18-controller-state-from-history)
separates input ambiguity, finite-budget learning accuracy, and downstream paddle
error. A lack of duplicate-input contradictions does not prove a history sufficient.
Complete executed-action history from a known native reset can reconstruct the
controller, but this is a different information contract from an arbitrary finite
paddle-history window. Life-loss boundaries clear model context while the native
controller retains its state. These probes do not establish autonomous rollouts.


### Full-data direction geometry experiments

`logs/ball-direction-highaccuracy-20260918/` contains the continuation toward
approximately 99.99% direction accuracy. `session.py` and `prepare_chunk.py`
construct the full 1,824-training-episode input set in RAM, preserving the fixed
32-episode development validation partition and life boundaries. Native checks
cover 5,271,950 nonterminal transitions. The descending paddle-region selection
contains 440,833 training examples and 93,389 paddle hits. No dataset or feature
cache files are written.

`ball_velocity_mlp(spatial_features=True)` is an optional 93-input baseline.
`ball_direction_geometry` instead freezes a learned intermediate paddle model
and trains a direction MLP on relative coordinates and motion proposals. The
intermediate paddle auxiliary target comes from recorded controller values and
the audited native rule. Only its labels use that rule; neural inference does
not. The final geometry encoding includes absolute-position bits and has 84
inputs. Its direction network has three 256-unit ReLU layers and two logits.
The complete checkpoint includes the frozen 25→128→128→13 SiLU paddle model.
See [model contracts](approaches.md) for the three geometry encodings.

`transfer.py` streams compressed arrays directly from local RAM to remote RAM.
`remote_train.py`, `remote_followup.py`, `remote_hard.py`, and
`remote_absolute.py` preserve configs, metrics, checkpoints, and error witnesses.
The GPU environment is recorded in `gpu-artifacts/frozen-selection.json` under
`runs/ball-direction-highaccuracy-20260918/`. This is an isolated diagnostic
workflow, not a new playable approach or a change to the shared runner.

`freeze.py` chooses by development validation errors before `prepare_test.py`
reads the reserved 64 test episodes. `remote_test.py` evaluates that choice once.
`remote_test_audit.py` attributes remaining errors without fitting; `verify_local.py`
checks the selected checkpoint through the production registry on CPU. Training
and test arrays remain in RAM throughout. The selected seed-91 model has zero
validation errors but six independent test direction errors, two on paddle hits.
Paddle test accuracy is 99.9380%; all selected test approaches score 99.9602%.
The inspected test episodes must not silently become a fresh independent test
for subsequent development. Full results and limitations are in the
[experiment history](history.md#2026-09-18-near-perfect-direction-development-accuracy-and-reserved-test).

### Isolated horizontal speed and frozen direction

`logs/ball-speed-20260918/` trains `ball_horizontal_velocity` on the same full
training episode set and development validation approaches as the direction
experiment. It retains the nine-source-variable contract and encodes geometry
with the previously selected direction model. All direction and intermediate
paddle parameters are frozen. Training caches those encoded features in GPU
memory and fits only the speed network with cross-entropy over the four observed
magnitudes, 0.5, 1, 1.5, and 2 native pixels per native frame.

The initial speed network is 84→128→128→4 with ReLU, trained with seeds 91 and
2026 for up to 35,100 updates. Wider three-layer 256-unit fits are conditional
on remaining validation errors. AdamW uses learning rate 0.001, weight decay
0.0001, batches of 1,024, gradient clipping at 5, and cosine decay to 0.00005.
Every 30 updates, exact speed errors on all validation approaches select the
checkpoint, retaining the earliest tie. Zero validation errors stop a fit.

Evaluation separates speed, direction, and their combined signed velocity. It
reports paddle hits, no-hit approaches, actual speed changes, unchanged speed,
fast paddle hits, first/second native-frame hits, and approaches passing below
the paddle. Keeping the incoming speed with the same frozen direction predictor
is the baseline. Aggregate accuracy alone is not sufficient evidence of learning
speed changes.

The existing 64 test episodes have already been inspected. `plan_holdout.py`
reserves 64 different episodes from the remaining 400 original held-out episodes
using only metadata and seed 190918. `freeze.py` selects the model before
`prepare_fresh_test.py` reads their transitions. Derived arrays remain in RAM;
`transfer.py` and `send_test.py` stream them over SSH without writing a feature
cache. Checkpoints include every inference weight and are reloaded on both CUDA
and CPU. The experiment does not alter dataset files or enable recursive play.


The selected speed head is the three-layer, 256-unit seed-91 fit. Combined
velocity has one error / 7,164 validation approaches and ten / 15,346 fresh-test
approaches. On fresh-test paddle hits, speed makes eight errors / 3,263 and the
combined velocity makes nine. Speed-change accuracy is 384 / 388. Checkpoint,
full subgroup results, frozen-weight checks, and a diagnostic attribution audit
are under `runs/ball-speed-20260918/gpu-artifacts/`. See the
[experiment history](history.md#2026-09-18-learned-horizontal-speed-with-frozen-direction)
for the complete comparison and the distinction between near-paddle and
full-game prediction.

### Full-game horizontal velocity with a frozen paddle component

`logs/ball-vx-router-20260918/` combines the selected near-paddle speed/direction
checkpoint with the older full-field eight-class vx model. The registered
`ball_horizontal_router` selects the near-paddle component when current RAM y
is in [160, 183] and current vy is positive. All other sources use the full-field
MLP. Routing uses no successor values or event labels. Inference remains learned;
native rules are used only to reconstruct and audit diagnostic inputs offline.

The full-field network remains 195→128→128→8 with SiLU. Fine-tuning updates only
its 42,632 parameters; the paddle component's 329,747 parameters stay frozen.
The composite contains every inference weight and both nested constructor specs.
It is a diagnostic vx checkpoint, not a complete state-dynamics/player checkpoint.

Two matched controls retain the original 128 training episodes. Two expanded
fits use all 1,824 eligible training episodes from the original training split.
Both start from the same existing full-field checkpoint, exclude the fixed
near-paddle source region from training, and use seeds 91 and 2026. Each fit has
30,000 updates, batches of 2,048 sampled uniformly with replacement, cross-entropy,
AdamW at 0.0003 with weight decay 0.0001, cosine decay to 0.00001, and gradient
clipping at 5. Random training-only brick-layout augmentation retains the earlier
native-audited region RAM y > 100. The validation partition remains 32 episodes.
Every 300 updates, exact vx errors outside the gate select the checkpoint;
the parent is also a candidate and earlier checkpoints win ties.

Four chunks stream derived training arrays from local RAM to remote RAM over
SSH. No feature cache or dataset columns are written. `prepare_chunk.py` checks
reconstructed current fraction, contact memory, and prior hit count against
recorded successors before training. This audit does not choose source phase
from a successor label. `holdout-plan.json` reserves 64 episodes using metadata
and seed 290918, excluding both previously inspected 64-episode test sets.
`final_test.py` copies and hashes the selected checkpoint before reading those
new test transitions. Report ordinary flight, walls, bricks, paddle hits, actual
vx changes, and unchanged vx separately; event categories can overlap.

The expanded seed-91 model was selected at update 20,400. Validation has 16
errors / 88,590 (99.9819%); the fresh test has 50 / 185,517 (99.9730%). The
same fresh test gives 751 errors for the old model, 119 for the routed parents,
and 84 for the best original-data fine-tune. All 21 fresh-test brick errors are
missed speed increases: only 31 / 52 actual brick accelerations are correct.
Keep this rare-event measure alongside aggregate accuracy. The selected model,
CPU/CUDA reload checks, provenance, and remaining-error audit are under
`runs/ball-vx-router-20260918/`. See the
[full-game vx experiment](history.md#2026-09-18-full-game-vx-integration-with-a-frozen-paddle-model)
for category counts and the matched training-data controls.

### Isolated brick-triggered horizontal acceleration

`logs/ball-acceleration-20260918/` freezes the complete selected full-game vx
model and fits a keep/increase-speed head. The source-only domain is RAM y <=
100 and incoming horizontal magnitude < 2. Across all prepared nonterminal
training transitions outside the paddle region, every magnitude change lies in
this domain; its observed outcomes are incoming magnitude or 2. Labels come
from recorded successor vx, not a native-rule label generator.

Input auditing found no conflicting acceleration labels among 178,725 distinct
reduced input vectors across development training/validation. The reduced inputs
are x, RAM y, vx, vy, fractional y, brick-contact memory, and 108 brick cells.
Native-rule checks under three interventions on paddle/controller values left
all 5,117 eligible validation targets unchanged. These checks support the input
contract on this data; they do not establish arbitrary-state observability.

All 1,824 training episodes provide 304,107 eligible examples, including 1,694
accelerations. Preparation still checks the complete 5,271,950 nonterminal
training transitions before filtering. Validation has 5,117 eligible examples,
including 29 accelerations. Flat scalar/binary heads use 149→128→128→2 ReLU;
constant-velocity geometry variants use 179 inputs. Compare uniform sampling
with 25% positive/75% negative batches using seeds 91/2026, batches of 1,024,
and 30,000 updates. AdamW uses 0.001, weight decay 0.0001, cosine decay to
0.00002, and gradient clipping at 5. Validation is evaluated every 100 updates.
Select minimum speed-decision errors, breaking ties by fewer misses and then
earliest checkpoint; zero validation errors stop a fit.

Because sampling changes alone retain substantial errors, a spatial head shares
13→64→64 ReLU layers across all 108 brick cells. It masks absent cells, max
pools the 64 learned features, appends six global scalars, and applies
70→128→2 ReLU layers. Cell-relative current/proposed positions use the fixed
layout geometry; no native collision decisions run during inference. Spatial
fits compare uniform and 25% positive sampling, both seeds, batch 512, and
15,000 updates with the same optimizer settings. A bounded capacity check uses
three shared 128-unit layers and 30,000 updates. Every fit freezes all 372,379
parent parameters. The selected small spatial head adds 14,402 trainable
parameters; the complete model contains 386,781.

The selected seed-91 balanced spatial fit is from update 6,900. It has one
missed acceleration and one false acceleration on validation, giving three
full-field vx errors / 88,590. The wider variants do not improve that count.
Reserve 64 fresh episodes using metadata seed 390918, excluding all three
previously inspected 64-episode test sets, then hash the selected checkpoint
before reading their transitions. Derived arrays remain in RAM and dataset
files are unchanged. The SSH log stream disconnected near the end of training;
recovered checkpoints, completed summaries, and the remote freeze record confirm
that all 14 fits and selection completed before test evaluation.

On 184,160 new test transitions, the parent makes 57 vx errors and the selected
model makes 16 (**99.9913% exact**). Actual upper-field horizontal speed increases
improve from 32 / 48 to **47 / 48**, while false increases fall from 29 to **3**.
The head makes four magnitude-decision errors; one inherited direction error
brings the region's full-vx errors to five. Eleven errors remain outside its
domain. These are one-step results with supplied current collision memory;
recursive full-state prediction remains untested. Checkpoints, all development
comparisons, provenance, and fresh-test results are under
`runs/ball-acceleration-20260918/`.

## Isolated vertical ball velocity

`ball_vertical_velocity` learns eight signed next-vy classes from the same
118-value current-state contract as the horizontal model. Source-state native
replay checks all 5,271,950 nonterminal training transitions from 1,824 episodes
and 88,590 validation transitions from 32 episodes. Paddle measurement is
reconstructed exactly from charge on these sources; this diagnostic finds no
additional input requirement. Fractional y, brick-contact memory, and prior
paddle-hit count are still supplied from audited reconstruction, not predicted
by this model. No dataset columns or persistent feature caches are added.

There are three disjoint source-selected regions: upper field (RAM y <= 100),
descending paddle approaches (160 <= RAM y <= 183 and vy > 0), and the rest.
Training contains 1,782,412 upper-field sources, 440,833 paddle sources, and
3,048,705 remaining sources. The remaining region has no observed vy changes.
Its 1→32→8 ReLU classifier is fitted to unique training velocity pairs and
then frozen. All inference remains neural; native rules only support the audit.

The upper head uses shared 13→64→64 ReLU cell features, occupancy-masked max
pooling, then 135→128→128→8 ReLU layers. Its 71 global features encode current
and proposed coordinates, velocities, fractional y, and contact memory. The
paddle head uses 97→256→256→256→8 ReLU layers: 84 existing geometry features
plus scalar/binary charge. Its hidden layers start from the learned horizontal
speed head, with the extra charge weights initially zero. The upper cell encoder
starts from the horizontal acceleration head. These copies train independently;
the complete 386,781-parameter horizontal model remains frozen. The new model
contains 585,845 parameters, including 199,064 fitted vertical parameters.

Both collision heads train together with summed cross-entropy. Upper batches
contain 512 sources, 25% with changed vy; paddle batches contain 1,024 sources,
50% with changed vy. AdamW uses learning rate 0.001, weight decay 0.0001, cosine
decay to 0.00002, and gradient clipping at 5. Compare seeds 91 and 2026 at 12,000
and 60,000 updates. Every 200 updates, select each independent head by minimum
validation exact errors, keeping the earliest tie. Compose those disjoint best
heads and select the fit with the fewest total validation errors.

The eight observed classes are -3.375, -2, -1.5, -1, 1, 1.5, 2, and 3.375
native pixels per native frame. Evaluation reports actual velocity changes,
ordinary transitions, paddle, brick, ceiling, and side-wall events separately;
event groups can overlap. Reserve 64 previously unused held-out episodes using
metadata seed 490918, excluding the four earlier test sets. Freeze and hash the
checkpoint before reading these test transitions. Artifacts and provenance are
under `runs/ball-vy-20260918/` and `logs/ball-vy-20260918/`.

The selected seed-91 long fit uses upper weights from update 30,800 and paddle
weights from update 17,200. Validation has one error / 88,590 (**99.9989%**),
a missed paddle bounce. The fresh 64-episode test has 13 errors / 182,441
(**99.9929%**): nine on ordinary transitions and four on actual velocity changes.
Of 10,441 actual changes, 10,437 are predicted exactly (**99.9617%**). Paddle
hits have one error / 3,335, brick events three / 4,895, and ceiling events zero
/ 2,211. Side-wall events have one / 4,527, overlapping another event category.
Two of 527 magnitude-changing outcomes are wrong. Reusing current vy would make
10,441 errors overall, illustrating why aggregate accuracy alone is insufficient.

The complete checkpoint reloads with identical predictions on CPU and CUDA.
The horizontal weights and predictions are identical to the frozen parent.
These results establish accurate one-step vy under the audited supplied-state
contract. They do not close the position, memory-update, or terminal models and
do not establish autonomous rollouts. No training follows fresh-test inspection.
