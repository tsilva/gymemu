# Training guide

Install the command with `uv tool install . --editable --exclude-newer "7 days"` from the checkout.
After dependency changes, rerun it with `--reinstall`; `uv sync` updates only the
checkout's `.venv`, not the installed command's uv tool environment.
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

## Browse published research

```bash
uv run gymemu play
uv run gymemu play /absolute/path/to/best.pt
```

The token-protected local browser reads the private R2 catalog. Its path is
Environment → Research Goal → Goal Revision → Goal Variant → Run → Checkpoint.
The Goal page lists checked-in recipes. A Run page shows its authoritative
publication state, comparable held-out RGB MSE, recovery file names, and resolved
Goal and Run YAML. Search, breadcrumbs, Refresh, and browser Back/Forward work at
each level. Opening a Checkpoint starts the Player paused in teacher-forcing mode.

The catalog reads at most 50 Run projections per browser request. **Load more Runs**
continues the R2 listing; search and comparable rank cover the loaded Runs, and
rank is shown only after the last page is loaded. It does not scan local run directories or
all R2 artifacts per request. An unavailable R2 catalog returns an error instead of
an empty history. `resume.pt` is listed in recovery details and cannot be selected
for playback. Inference Checkpoints and their starting scenes are downloaded into
`~/.cache/gymemu/checkpoints/` with SHA-256 and size checks. Explicit local
Checkpoint paths still open unpublished Runs; `--local-only --runs-dir PATH`
retains the older local directory browser for migration work.

Playback options such as `--device cpu`, `--autoregressive`, `--start-state`, and
`--empty-start` apply to each selected checkpoint. In autoregressive mode, missing
starting scenes remain an error unless an explicit alternative is supplied. Headless playback and
`--list-start-states` still require a checkpoint argument. Supplying a checkpoint
directly opens the Player window; its **Diagnostics** button opens the companion window.

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

Online W&B training uploads Run artifacts and a catalog record to the private
`gymemu` R2 bucket. It copies only inference-ready Checkpoints and their playback
scene to the separate public `gymemu-public` bucket at
`https://gymemu-assets.tsilva.eu`. Recovery files, Run metadata, metrics, and
diagnostics stay in the private bucket. The browser reads Run metadata with the
private token, then downloads selected playback files over public HTTPS and checks
their size and SHA-256 before loading them. Disabled or offline W&B mode keeps the Run local, even when
`r2.enabled=true`. Use `gymemu sync RUN_DIRECTORY` after completion to publish
that Run under its original ID. `r2.enabled=false` also keeps the Run local.

Provision the bucket once in the same Cloudflare account you use for Gradlab:

1. In **R2 object storage**, create a bucket named `gymemu`. Keep it private.
2. Create an R2 API token with **Object Read & Write** permission scoped to that bucket.
3. Create a separate `gymemu-public` bucket and connect the public custom domain
   `gymemu-assets.tsilva.eu`. Create an Object Read & Write token scoped only to
   this bucket, with a limited lifetime; rotate it before expiry.
4. Export the private token's account endpoint, access key ID, and secret access key in the training
   process as `GYMEMU_MODELS_R2_ENDPOINT_URL`, `GYMEMU_MODELS_R2_ACCESS_KEY_ID`, and
   `GYMEMU_MODELS_R2_SECRET_ACCESS_KEY`.
5. Export the public token's matching account endpoint and keys as
   `GYMEMU_PUBLIC_R2_ENDPOINT_URL`, `GYMEMU_PUBLIC_R2_ACCESS_KEY_ID`, and
   `GYMEMU_PUBLIC_R2_SECRET_ACCESS_KEY`.

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

The public token can use an analogous `~/.config/gymemu/public-r2.toml` profile.
Use `GYMEMU_PUBLIC_R2_CONFIG` to choose another path. Its `keychain` references
must point to the public token, not the private token. A local installation can
therefore upload both buckets without putting secrets in the repository. The public
token created for this installation expires on 2027-09-23.

```bash
# With W&B authentication and the three R2 variables already configured
uv run gymemu train output=runs/breakout-stored

# Choose another dedicated bucket or object prefix
uv run gymemu train r2.bucket=gymemu r2.prefix=experiments

# Local-only smoke with no W&B or R2 credentials
uv run gymemu train experiment=smoke wandb.mode=disabled r2.enabled=false

# Retry a failed online publication under its saved Run ID
uv run gymemu upload-checkpoints runs/breakout-stored

# Publish a completed offline Run to W&B, R2, and the catalog
uv run gymemu sync runs/offline-run
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
does not reopen or backfill a finished W&B run. `gymemu sync <run-directory>` can
publish a completed local Run to W&B and R2 under its original Run ID. R2 also
stores `resume.pt`, including the optimizer and progress required to continue
training after a restart.

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

`play.py` serves the player on loopback and opens a dedicated Player app window.
The **Diagnostics** button opens or focuses a second app window with its own icon.
Both pages use a compiled Svelte shell and the existing playback controllers.
Installed packages include the compiled assets; source UI changes require
`pnpm build:web`. The native viewer uses a pinned, verified Neutralinojs runtime
downloaded on first use. `--no-browser` skips that download and prints both complete URLs for opening
manually or in Codex's in-app Browser. `--port auto`
selects an unused port; `--port NUMBER` selects a specific port. Each server has one
playback session and a random access token shared by its printed URLs. Both tabs
read the same atomic frame/metric revisions from that session, without duplicate
inference. Switching windows keeps playback running and clears held keys. Closing
the Player page pauses playback; closing both native windows stops the server.
Diagnostics only sends explicit chart selections as seek commands; it sends no
keyboard, heartbeat, or blur commands. Ctrl+C also stops the server.

The UI reuses Gradlab's panel module lifecycle, registry, GridStack layout, fonts,
and theme, including paired views with per-tab panel placement. Original, Prediction,
and difference belong to the player tab with its playbar; input history, error charts,
and custom widgets belong to Diagnostics. Its default arrangement puts
Input history across the full top row, with a full-width Prediction error chart
below. History tiles use minimal spacing, overlaid top-left frame numbers,
and no bottom caption. Older default arrangements migrate; custom placements remain. Header links open or focus
the companion window in desktop mode, or a tab with `--no-browser`. Layout updates use BroadcastChannel with a storage-event fallback
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

## Isolated ball positions and fractional y

`ball_position` classifies displacement for one coordinate, keeping the complete
selected vertical/horizontal velocity checkpoint frozen. Both coordinate fits
reuse the same 5,271,950 nonterminal training sources from 1,824 episodes and
88,590 validation sources from 32 episodes. The source contract remains the
existing 118 values. Native replay reproduces the recorded positions on every
source. Fractional-y targets come from the audited replay, not a newly recorded
column. All preparation arrays remain in RAM; dataset files are unchanged.

Training exposes 22 horizontal displacement classes and 19 vertical classes,
all exact multiples of one eighth pixel. For y, classify displacement of the
combined coordinate `RAM_y + fractional_y_eighths / 8`. Add that displacement,
then split the result into integer RAM y and the remainder in eighth pixels.
This prevents independent y/fraction heads from disagreeing about a pixel carry.
The coordinate heads consume current state only, not true successor velocities.

The source-only upper/paddle/remaining regions match the vertical-velocity model.
Each axis has its own shared 13→64→64 ReLU cell encoder, occupancy-masked max
pool, and 135→128→128→K upper head. Its paddle head is
97→256→256→256→K ReLU. K is 22 for x and 19 for y. These hidden layers start
from the successful vy model; their copies train while all parent weights stay
frozen. Outside those regions, x uses 31→128→128→22 ReLU layers over
scalar/binary current/proposed x and current vx. All 2,316 distinct training
(x, vx) pairs in that region have unambiguous displacement targets. The y flight
head uses 1→32→19 ReLU layers, fitted on unique training (vy, displacement)
pairs; all audited flight targets equal twice current vy.

Region-head losses are summed cross-entropies. Upper batches have 512 samples,
25% whose displacement differs from twice incoming velocity; paddle batches
have 1,024 samples, 50% with that difference. The x flight batches have 512
samples, 25% differing. AdamW uses learning rate 0.001, weight decay 0.0001,
cosine decay to 0.00002, and gradient clipping at 5. Evaluate every 200 updates
and retain each disjoint head's earliest minimum-error validation checkpoint.
Compare seeds 91 and 2026 with 30,000 updates for x, then freeze the selected
x model before fitting y with 60,000 updates per seed.

Simple integration is insufficient even with oracle velocity: on validation,
adding twice true successor vx gives 1,669 x errors, and twice true successor
vy gives 3,330 combined-y errors. Twice current velocity gives 3,164 and 5,253
errors respectively. These diagnostic baselines do not supply model inputs.

Reserve one shared set of 64 previously unused held-out episodes using metadata
seed 590918, excluding all five earlier test sets. Freeze and hash both selected
coordinate checkpoints before reading any test transition. Report exact x,
combined y, integer y, fractional y, joint position, collision groups, and
transitions differing from constant-velocity motion. Verify saved-model CPU/CUDA
agreement and unchanged vx/vy predictions. Artifacts and complete provenance are
under `runs/ball-position-20260918/` and `logs/ball-position-20260918/`.

Select x seed 2026: three validation errors / 88,590 (**99.9966%**), using
upper/paddle/flight weights from updates 22,800 / 12,800 / 1,000. Both y seeds
achieve zero validation errors; the predefined name tie-break selects seed 2026,
with upper/paddle weights from updates 29,200 / 12,400. X contains 813,431 total
parameters, including 227,586 fitted position parameters; y contains 789,518,
including 203,673 fitted position parameters. Each stores the same frozen
585,845-parameter velocity parent; those duplicated parent counts are not
independent learned capacity.

The fresh test contains 187,251 transitions. X has 21 errors (**99.9888%**,
0.000283 px MAE), with 12 errors on 3,251 paddle events, three on 5,122 brick
events, and six on ordinary no-event transitions. Y including its fraction has
seven errors (**99.9963%**, 0.000156 px MAE): two on paddle events, three on brick
events, one on a side-wall event, and one ordinary transition. Neither coordinate
errs on any of 2,624 ceiling events. Groups can overlap. Integer y has seven
errors and its fractional component has three (**99.9984% exact**).

Across transitions whose displacement differs from twice incoming velocity,
x has 15 errors / 6,652 and y has five / 10,997. Joint position has 28 errors
(**99.9850% exact**). The unchanged vx/vy models make 17 and 18 errors on this
new test; all four ball quantities jointly have 56 errors (**99.9701% exact**).
Saved predictions agree on CPU and CUDA and both velocity models remain
identical to their parent. No test error is used for subsequent fitting.

Current collision memory, hit count, layout, width, and paddle state are still
supplied. The next fractional-y value is now learned, but feedback across whole
lives and the remaining state updates are separate work. Perfect validation
here does not imply perfect fresh-test or recursive simulation performance.

## Isolated next brick layout

`brick_layout` predicts a categorical layout update: no change, or removal of
one of the 108 occupied cells. The recorded successor layout supplies the labels;
native replay is used only to audit source-state sufficiency. The full training
set contains 5,271,950 nonterminal transitions from 1,824 episodes, including
142,307 removals. Every changed layout removes exactly one brick. There are no
additions, empty source layouts, or wall clears; the minimum source layout has
three bricks. Validation has 88,590 sources from 32 episodes and 2,432 removals.

The game code supports wall refills, but these audited records contain no refill
examples. This model's output contract does not support additions, multiple
removals, or refill timing. Do not generalize removal accuracy to wall completion.
The dataset and its columns remain unchanged, and derived arrays stay in RAM.

The head accepts the established 118-value source contract but uses only ball
x, RAM y, vx, vy, fractional y, brick-contact memory, and the current brick cells.
Paddle/controller fields do not affect its output. The complete selected y
position model, including both velocity predictors, stays frozen. Its cell
encoder initializes a separate trainable 13→64→64 ReLU network. Occupancy-masked
max pooling supplies 64 global features, concatenated with 71 current-state
features. A shared 199→64→64→1 ReLU scorer combines that context with each
cell's 64 local features. A 135→128→128→1 ReLU head scores no change. Absent
cells are masked from removal logits. The argmax selects one coherent update;
no native collision rules or successor/event inputs run during inference.

Compare seeds 91 and 2026 with 30,000 updates, batch size 512, and 25% actual
removal samples. Cross-entropy trains all 109 outcomes together. AdamW uses
learning rate 0.001, weight decay 0.0001, cosine decay to 0.00002, and gradient
clipping at 5. Evaluate every 500 updates and keep the earliest checkpoint with
minimum exact-layout validation errors. Because sampling changes the class
prior, also compare each saved model with an analytical correction to its
no-change bias, derived solely from training removal frequency and the 25%
sampled frequency. No validation threshold sweep or extra fit is used.

Reserve 16 previously unused held-out episodes with metadata seed 690918,
excluding all six earlier test sets and preserving 64 untouched episodes for
later full-state evaluation. Freeze and hash the selected checkpoint before
reading these new transitions. Report exact-layout and changed-layout accuracy,
false changes, missed changes, wrong-cell removals, and removal-cell precision,
recall, and F1. A wrong-cell choice counts as both a false removal and a missed
true removal. Whole-layout accuracy is stricter than per-cell accuracy.

Checkpoints, development comparisons, audit receipts, and evaluation provenance
are under `runs/brick-layout-20260918/` and `logs/brick-layout-20260918/`.

Seed 91's best checkpoint has one false removal at update 18,000. Seed 2026
reaches zero validation errors at update 20,000. The analytical no-change bias
correction is +2.48619236: it replaces seed 91's false removal with one missed
removal and leaves seed 2026 exact. Select the uncorrected seed-2026 checkpoint
using minimum error then name; no correction is needed. The complete model has
845,648 parameters: 789,518 frozen parent parameters and 56,130 fitted parameters.
Every cell has between 877 and 1,800 removal examples in training.

On the fresh 16-episode test, the model predicts all 45,245 complete layouts
exactly. It removes the correct cell on all 1,184 changed layouts and makes no
false removals on the 44,061 unchanged layouts: removal precision, recall, and
F1 are all 1.0. An always-unchanged baseline scores 97.3831% aggregate exactness
but misses every removal. No wall clears or refills occur in this test either;
its smallest source layout contains six bricks.

Reloaded CPU and CUDA predictions agree exactly, and the frozen y/velocity
parent's weights and predictions remain unchanged. The separate x checkpoint
is also unchanged. These are one-step results with current contact memory
supplied; learning its next value and validating recursive full-state feedback
remain separate work. No fitting follows fresh-test inspection.

## Isolated next brick-contact memory

`brick_contact` learns the next value of the contact flag that suppresses repeat
brick hits. Reconstruct source and successor contact in RAM by replaying the
pinned native rules from known resets. Check every replayed nonterminal successor
against recorded positions, velocities, and brick cells. Successor contact is
derived supervision, not a directly recorded dataset field. Keep life boundaries
and preserve all eligible transitions from later lives. No dataset columns or
persistent feature caches are written.

Freeze the selected brick-layout checkpoint, including its y and velocity
parents. Its source-state encoding supplies 71 ball features and two predicted
probabilities for no removal versus any removal. Train a 73→128→128→2 ReLU
classifier with cross-entropy. Compare seeds 91 and 2026 for 20,000 AdamW
updates, batch size 512, with 128 examples from each current/next contact pair.
Use learning rate 0.001, cosine decay to 0.00002, weight decay 0.0001, and gradient
clipping at 5. Select the earliest minimum-error validation checkpoint, checking
every 250 updates. Compare runs by validation errors, then name.

Reuse the 16 held-out episodes previously inspected for brick layout, excluding
them from contact training and selection. This is a reused test, not a new fresh
test. Preserve the remaining 64 untouched episodes for later full-state testing.
Freeze and hash the selected checkpoint before reading the reused-test rows.
Report activation, clearing, and both unchanged cases separately.

Also feed predicted contact into the following transition while supplying every
other state field from the recorded or reconstructed source. Reset contact at
life, episode, and step gaps. Check contact, brick layout, x, combined integer
and fractional y, vx, and vy together, including consecutive windows around
brick collisions. This isolates contact feedback and is not a full-state rollout.

The initial probability-only heads finish with 104 and 102 validation errors.
These are mostly false activations near the top and bottom of the brick area.
An exact float32 input-grouping audit finds no contradictory validation labels,
so this result does not establish that the representation is non-identifiable.

Test `collision_geometry=True` next. It appends 13 cell-geometry features,
weighted by the frozen brick model's removal probabilities, to form an
86→128→128→2 ReLU head. This exposes where the predicted collision occurs
without using a recorded removal label. The head has 27,906 trainable parameters;
the complete model has 873,554, including the unchanged 845,648-parameter parent.
Train both seeds with the same schedule and sampling. All 5,271,950 reconstructed
source rows match the earlier audited ball features exactly. CPU batch-size
changes alter removal probabilities by at most 1.2e-7 during preparation.

Both geometry seeds reach zero validation errors. Seed 91 first does so at
update 14,500; seed 2026 at 18,000. The name tie-break selects seed 2026. Freeze
its complete checkpoint before reused-test evaluation. It predicts all 45,245
next-contact labels correctly, including 996 activations, 993 clearings, 1,357
active retentions, and 41,899 inactive retentions.

Contact-only feedback remains exact across all 52 reused-test life segments.
Brick layout also stays exact. The frozen ball models have 8 x errors, 3 combined-y
errors, 5 vx errors, and 5 vy errors, with 15 distinct wrong transitions. These
counts are identical with supplied and predicted contact. The joint accuracy of
contact, layout, and all four ball quantities is 99.9668%. Of 1,181 complete
seven-step collision windows, 1,178 are jointly exact. Contact feedback adds no
errors; it does not resolve existing ball-model mistakes or establish a complete
recursive emulator. Validation feedback is also exact for contact and layout,
with five existing joint ball errors across 88,590 transitions.

CPU checkpoint reload reproduces the selected model's validation predictions.
All parent weights and the separate x checkpoint remain unchanged. Both GPU
addresses were unreachable, so this experiment trained and evaluated on CPU;
there is no CPU/CUDA comparison for this checkpoint. Artifacts and audit receipts
are under `runs/brick-contact-20260919/` and `logs/brick-contact-20260919/`.
The selected complete checkpoint is `geometry-s2026/best.pt` in the run folder.

## Isolated next paddle-hit count

`paddle_hit_count` learns hit detection and decodes a count that either stays
unchanged or increases by one, capped at 12. Life resets belong to segment
initialization. Reconstruct current counts and native hit labels in RAM, auditing
all recorded nonterminal successors against native replay. The dataset stays
unchanged. All 5,271,950 training sources from 1,824 episodes pass the audit.
There are 440,833 sources in the established paddle region and 93,389 hits.

Use current ball x, RAM y, vx, vy, fractional y, paddle x and width, and charge.
The frozen vertical-velocity model supplies its paddle encoding, including the
learned intermediate paddle position. Drop the prior-count scalar from its
97 features, leaving 96. The event head has three 256-unit ReLU hidden layers
and two output logits. Its 156,930 fitted parameters join 585,845 frozen parent
parameters. Initialize hidden layers from the vertical paddle head, dropping the
count input column. Initialize the no-hit and hit output weights by averaging
the parent's positive-vy and negative-vy class weights respectively.

Train seeds 91 and 2026 for 20,000 updates each with binary cross-entropy, batch
512, and equal hit/non-hit samples from the paddle region. AdamW uses learning
rate 0.0005, cosine decay to 0.00001, weight decay 0.0001, and gradient clipping
at 5. Evaluate every 250 updates. Select minimum validation hit errors first,
then count errors, retaining the earliest update; compare seeds by those metrics
and name. This prevents count saturation from hiding missed hits. Validation
has 1,519 hits, of which 699 increment the count and 820 occur at count 12.

Keep the existing 16 reused brick/contact-test episodes outside count fitting
and selection, preserving 64 untouched episodes for the eventual full-state
evaluation. This set has previously inspected ball errors and is not a new
fresh test. Freeze and hash the selected checkpoint before reading its raw
count-test rows. Report hit precision/recall, missed saturated hits, actual and
false count increments, and exact count accuracy.

Feed predicted count and contact forward from known life starts while supplying
all other state fields. Check whether count errors persist and whether they
change the frozen ball models' predictions. Reset only at reference life,
episode, or step boundaries. This is partial-state feedback, not learned
termination or a complete emulator rollout.

Both seeds finish with zero validation count errors and one hit-event error.
Seed 91 misses a hit at saturated count 12; its best update is 2,250. Seed 2026
adds a false hit at count 12; its best update is 3,750. The predeclared name
tie-break selects seed 2026. Its event precision is 99.9342% and recall is 100%
on validation. Neither event mistake changes the capped validation counter.

On the reused test, the selected checkpoint detects all 801 real hits, including
all 377 count increments and 424 saturated hits. One false hit causes one count
error in 45,245 transitions, or 99.9978% exact count accuracy. Hit precision is
99.8753%; recall is 100%. These conditional event metrics must not be confused
with aggregate count accuracy.

With count and contact fed back across 52 life segments, the false increment
persists for four scored transitions, giving 99.9912% count accuracy. Contact
and brick layout remain exact, and all frozen ball predictions remain identical
to their supplied-count predictions. Joint state errors rise from 15 to 18,
all three additions due to count. Validation feedback has zero count/contact
errors across 79 life segments and retains five existing joint ball errors.

The remaining false hit occurs at episode 1570, step 2228. It predicts count 5
instead of 4. The frozen intermediate-paddle model correctly predicts 61 px,
matching the offline controller rule. The source also caused earlier ball
velocity errors. The reference segment ends four transitions after this source;
the experiment does not demonstrate autonomous recovery or learned termination.
No test-directed refitting is performed.

The selected CPU checkpoint is
`runs/paddle-hit-count-20260919/count-s2026/best.pt`. Complete configuration,
weights, selection, native-audit receipts, feedback results, and the remaining
error audit are under that run folder and `logs/paddle-hit-count-20260919/`.
CPU reload preserves the selected predictions and every parent weight; the
separate x and contact checkpoints retain their hashes. Verification passes
381 tests with two skipped, Ruff, frozen dependency sync, and whitespace checks.

## Isolated next paddle width

`paddle_width` uses recorded next-width labels, independently audited against the
pinned native rules. In the scoped within-life data, ceiling collisions narrow
the paddle from 16 to 12 pixels. Serve and life-loss width resets belong to the
excluded lifecycle transitions. The full audit covers 5,271,950 nonterminal
training sources from 1,824 episodes, including 2,159 narrowings, 4,093,205 wide
retentions, and 1,176,586 narrow retentions. No within-life widening is observed.
The recorded widths match the offline rule on every audited source. Derived
arrays stay in RAM, and the dataset remains unchanged.

The model uses current width, RAM ball y, fractional y, and vy. It combines
integer and fractional y, then encodes three normalized scalars, 11 coordinate
bits, six signed-velocity bits, and a current-narrow indicator. The resulting
21→64→64→2 ReLU classifier has 5,698 parameters. Its two classes decode to
12-pixel and 16-pixel width. It predicts width directly without a native ceiling
rule, successor-state input, or hard-coded retain/narrow update. Existing models
remain unchanged.

Train seeds 91 and 2026 for 10,000 AdamW updates each, with cross-entropy and
batch size 384. Sample 128 examples from each observed transition type: wide
retention, narrowing, and narrow retention. Use learning rate 0.001, cosine decay
to 0.00002, weight decay 0.0001, and gradient clipping at 5. Select minimum exact
width validation errors, then changed-width errors, keeping the earliest
checkpoint at 250-update intervals. Compare seeds by those metrics and name.

Validation contains 36 narrowings, 67,038 wide retentions, and 21,516 narrow
retentions. A width-persistence baseline scores 99.9594% overall but misses all
36 changes. Report actual-change recall, false changes, and false widenings
alongside aggregate accuracy.

Freeze the selected checkpoint before reading the same 16 reused test episodes
used for brick, contact, and count diagnostics. Keep them out of width fitting
and selection, preserving 64 untouched episodes for eventual full-state testing.
Feed width, count, and contact together from known life starts; all remaining
state fields stay supplied. Re-evaluate the frozen companion models whenever a
fed-back input differs from its recorded or reconstructed source. This checks
whether width feedback adds errors but does not establish full-state rollouts
or learned termination.

The initial heads detect all 36 validation changes but produce two and one
false narrowings for seeds 91 and 2026. Add the constant-velocity proposal
`y + vy`, normalized and encoded with 11 bits, to make transition timing more
explicit. This adds no source field or collision rule. The resulting
33→64→64→2 network has 6,466 parameters. Reconstruct this feature from the
RAM-only cached inputs and verify exact agreement with direct encoding on all
359,086 original training sources.

| Candidate | Best update | Validation width errors / 88,590 | Change errors |
| --- | ---: | ---: | ---: |
| Original encoding, seed 91 | 6,750 | 2 | 0 |
| Original encoding, seed 2026 | 7,250 | 1 | 0 |
| Motion proposal, seed 91 | 4,750 | 2 | 0 |
| Motion proposal, seed 2026 | 7,500 | **0** | **0** |

Freeze the selected proposal/seed-2026 checkpoint. It predicts all 45,245
reused-test widths exactly: 18 narrowings, 36,007 wide retentions, and 9,220
narrow retentions. Change precision and recall are both 100%, with no false
widening. These finite tests cover only 18 test width changes and do not prove
universal correctness.

Width stays exact when width, count, and contact are fed back across all 52 test
life segments. The outputs of every older model match the prior count/contact
feedback evaluation exactly, including four count errors and 18 joint state
errors. Validation feedback has zero width errors across 79 segments and retains
five existing joint ball-state errors. This is not full-state feedback; ball
coordinates/velocities, paddle position/charge, and brick inputs remain supplied.
Reference life boundaries still stop the runs.

The selected CPU checkpoint is
`runs/paddle-width-20260919/proposal-s2026/best.pt`. Audit receipts, comparisons,
selection, and feedback results are under that run folder and
`logs/paddle-width-20260919/`. CPU reload reproduces the selected raw-input
predictions; all frozen companion checkpoint hashes remain unchanged.

Verification for the width model passes 385 tests with two skipped, Ruff,
frozen dependency sync, and whitespace checks.


## Isolated life-loss termination

`life_termination` predicts whether the current two-native-frame transition ends
a life. Unlike the ball-state probes, include terminal source rows in its
training view. Use the existing recorded boundary labels (lives decrease or
successor ball y equals zero); do not label truncation or invalid-data censoring
as death. Retain all later lives as separate segments. Nothing is added to the
dataset.

Reconstruct the terminal source's fractional y from the preceding nonterminal
source's causal native replay output, asserting that both belong to the same
life. The native game checks for loss before each native movement. An offline
audit agrees with every included recorded boundary using current vertical state
at the fixed frame skip of two. Native rules supply auditing and fractional-state
reconstruction only; the trained network executes no loss threshold.

The head reads current RAM ball y, fractional y, and vy from the shared 118-value
contract. Combine the y parts into eighth-pixel coordinates. Two normalized
scalars, 11 coordinate bits, six signed-velocity bits, and normalized `y + vy`
with 11 bits yield 31 inputs. Use 31→64→64→2 ReLU layers and cross-entropy,
6,338 parameters in total. No action history or lives counter enters the model.
This establishes a sufficient tested input set, not that every one of its
features is necessary.

Prepare all 1,824 training episodes in RAM: 5,275,072 transition sources,
including 3,122 deaths. Sample batches of 384 with 128 deaths, 128 surviving
sources with RAM y at least 198, and 128 other survivors. There are 6,715
near-boundary surviving sources and 5,265,235 other survivors. Use AdamW at
0.001, cosine decay to 0.00002, weight decay 0.0001, gradient clipping at 5,
and 8,000 updates per seed. Encoded features stay in RAM.

Select minimum validation errors, then missed deaths, retaining the earliest
checkpoint at 250-update checks and breaking run ties by name. Both seeds have
zero errors on 88,638 validation transitions, including 48 deaths. Seed 91 first
reaches this at update 250; seed 2026 at update 500 and wins the declared run-name
tie break. Training each seed takes about five seconds on the local CPU after
data preparation. Validation first-stop checks cover 79 life segments: all 48
death-ended segments stop on the correct step and 31 censored segments have no
predicted stop.

Freeze the selected checkpoint before reading the same 16 previously inspected
test episodes. Preserve the remaining 64 untouched episodes. The selected model
makes zero mistakes across 45,283 reused-test transitions: 38 deaths and 45,245
survivors. Death precision and recall are both 100%. Every death stops at its
exact transition, with no premature stops across 52 segments; all 14 censored
segments have no predicted stop. These checks supply recorded current ball state
and reconstructed fractional y at each step, so they are not full-state rollouts.
The 38 test deaths are limited event coverage, not proof of universal accuracy.

The selected CPU checkpoint is
`runs/life-termination-20260919/stop-s2026/best.pt`, SHA-256
`ff2bad6f68b5701699684537186ba225b171ae271f65ffbf1587a1a68493f37d`.
Preparation scripts, audit receipts and checkpoint selection are under
`logs/life-termination-20260919/`; evaluation is in
`runs/life-termination-20260919/result.json`. CPU reload exactly reproduces
predictions, and all checked companion checkpoints retain their hashes. The
shared runner and player are unchanged. An integrated simulator must honor the
predicted stop before consuming unsupported terminal successor state outputs.

Verification passes 387 tests with two skipped, Ruff, frozen dependency sync,
and whitespace checks.


## Paired vertical position and velocity

Start with the existing `y-s2026` position checkpoint and its exact frozen
`long-s91` vy parent. Register `vertical_ball_pair` to predict both from the
same current source and update y, fractional y, and vy together. Do not feed a
new position into the old velocity predictor within the same transition.

Use every nonterminal source as a rollout start, requiring the whole requested
horizon to remain inside one life. Compare reference inputs, y-only feedback,
vy-only feedback, and both together at horizons 1, 8, 32, and 128. Hold all other
fields at their recorded or reconstructed values and use reference boundaries.
These are partial-state diagnostics. The windows overlap and are not independent
samples. Terminal successor states and learned stopping remain outside this test.

The baseline has one incorrect validation vy prediction at episode 1918, step
2461. The y head predicts the upward paddle bounce correctly, while vy remains
downward. Y-only feedback does not add errors. Velocity-only feedback affects
18 endpoints at horizon 128; joint feedback affects 128, with maximum vertical
error 938.25 pixels. This is the measured behavior of an unbounded diagnostic
rollout after a missed collision, not evidence of a playable simulator.

Test a smaller intervention before changing the two original heads. Freeze them
and fit a correction using current vy, predicted y displacement, and eight
original vy probabilities. Normalize the scalars by 3.375 and 6.75. The
10→64→64→8 ReLU classifier adds 5,384 parameters and outputs next vy. It uses no
true successor or collision labels as inputs, and executes no native game rules.
All inputs derive from the current state and frozen predictions. Y stays exact
relative to its original checkpoint.

Use the original 128 training episodes for correction fitting: 359,086 sources.
The frozen parents retain their previous larger training provenance. Sampling
uses 128 original-vy mistakes, 128 correctly predicted velocity changes, and
128 other correct predictions per batch, drawn with replacement. These groups
contain 3, 20,628, and 338,455 sources. Keep encoded features in RAM. Fit seeds
91 and 2026 for 6,000 AdamW updates at 0.001, cosine decay to 0.00002, weight
decay 0.0001, and gradient clipping at 5. Use categorical cross-entropy.

Within each seed, shortlist the earliest minimum one-step validation-error
checkpoint at 250-update checks. Compare shortlisted candidates by joint
128-step endpoint errors, then one-step errors, then name; retain the original
pair unless improved. Both seeds reach zero validation errors at update 250 and
remain exact through all 78,806 eligible 128-step windows. Select seed 2026 by
the run-name tie break. No rollout fine-tuning is needed to obtain this result.

Freeze before reading the same 16 previously inspected test episodes. Preserve
64 untouched episodes for later integration. The original y/vy weights remain
byte-identical. Reused-test one-step errors change as follows:

| Output | Original pair | Corrected pair | Sources |
| --- | ---: | ---: | ---: |
| Combined y | 3 | 3 | 45,245 |
| vy | 5 | 1 | 45,245 |
| Either output | 7 | 3 | 45,245 |

With both vertical outputs fed back, the horizon results are:

| Horizon | Eligible windows | Original incorrect endpoints | Corrected incorrect endpoints |
| --- | ---: | ---: | ---: |
| 1 | 45,245 | 7 | 3 |
| 8 | 44,881 | 45 | 24 |
| 32 | 43,640 | 165 | 96 |
| 128 | 39,203 | 422 | 224 |

At horizon 128, fully exact windows improve from 38,692 to 38,979, or 98.6965%
to 99.4286%. This differs from endpoint accuracy because an original rollout can
recover after an earlier error. Mean endpoint y error falls from 1.9200 to
1.5996 pixels, but worst error rises from 782.25 to 897.75 pixels. Mean y error
also rises slightly at horizons 8 and 32 despite fewer incorrect endpoints.
The improvement is fewer failed trajectories, not uniformly smaller drift.

All three remaining first errors originate in the unchanged y predictor, two
at paddle interactions and one at a brick interaction. One also has wrong vy.
Do not fit these held-out witnesses. Short rollout training or a better y head
can be investigated using training/validation cases in a subsequent experiment.
Other supplied fields can conflict with a diverged vertical trajectory, so this
diagnostic does not determine how a fully autonomous state would evolve.

The selected standalone checkpoint is
`runs/vertical-pair-20260919/coupling-s2026/best.pt`, SHA-256
`db047b3a50ba075aba9238240b86109b9c1a209fe19b71234daae3ccdc60e748`.
It contains both original predictors and the correction. Plan, preparation,
fitting and evaluation scripts are under `logs/vertical-pair-20260919/`;
`runs/vertical-pair-20260919/result.json` records all four feedback modes.
No dataset columns or persistent feature caches were written. Shared training
infrastructure and the player are unchanged.

Verification passes 390 tests with two skipped, Ruff, frozen dependency sync,
and whitespace checks. The three focused pair/evaluator tests also pass after
the inference optimization.


## Vertical collision displacement refinement

This experiment is **rejected for use as the current pair**. Retain
`runs/vertical-pair-20260919/coupling-s2026/best.pt`. The candidate and scripts
are preserved to reproduce the result, not as a claimed accuracy improvement.

The original 32-episode validation set is already exact for the current pair.
Before fitting, reserve 256 additional development episodes from the training
split, excluding the original 128 correction-fitting episodes. Select them with
NumPy seed 202619. The remaining 1,568 episodes provide 4,529,910 fitting sources;
the development set provides 742,040 sources. All are nonterminal within-life
transitions. Frozen parents have already trained on this additional development
pool, so it is independent of the new heads' fitting only. It is not a fresh
held-out test. Original validation remains a separate zero-regression gate.

Audit all 5,271,950 training-split sources using the existing causal fractional-y
reconstruction. The fitting subset has 23 y errors and 33 joint y/vy errors;
development has five y errors and six joint errors. No fitting y errors occur in
the flight region. Mistakes include missed collisions, wrong collision timing,
and false movement changes near collision regions. The audit does not establish
missing inputs as their cause.

Freeze the entire current pair. Add residual logits to y in the upper-field and
paddle regions, retaining flight logits. The two ReLU heads are
162→64→64→19 and 124→64→64→19, with 29,222 new parameters total. Features reuse
frozen spatial/paddle representations and append the original y probabilities
and coupled vy probabilities. No true successor or event enters the inputs.
Feed the corrected predicted displacement into the unchanged velocity coupling.
Zero-initialize each final layer to start with the exact parent predictions.

Fit seeds 91 and 2026 for 6,000 updates. For each region and update, sample
128 original y errors, 128 correct nonlinear displacements, and 128 correct
linear displacements with replacement. Upper-field groups have 8, 182,638, and
1,351,618 sources; paddle groups have 15, 80,185, and 298,396. Use AdamW at
0.0005, cosine decay to 0.00001, weight decay 0.0001, gradient clipping at 5,
cross-entropy, and a 0.00001 penalty on squared residual logits. Features remain
in RAM; no dataset columns or feature caches are written.

The unscaled correction is too aggressive. Neither seed has an eligible
zero-error original-validation checkpoint at 500-update checks. Both end with
four original-validation errors. Development joint errors at the final update
are 13 for seed 91 and 12 for seed 2026, versus six for the parent.

Before reading test data, calibrate the final seed-2026 correction with scales
0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, and 1. Require zero
original-validation errors and fewer development joint errors, choosing the
smallest scale on a tie. Scales 0.3 and 0.4 reach three development joint errors;
select 0.3. Fold the scale into the heads' final weights and biases. It scales
class-score corrections, not physical displacement. Y still lies on the same
eighth-pixel grid.

The calibrated model repairs 22 of the 23 fitting y errors without introducing
new fitting y errors. Original validation stays exact, including all 78,806
128-step windows. Development failed 128-step windows fall from 357 to 89 out of
659,845. Mean endpoint y error falls from 0.2143 to 0.0240 pixels and worst error
from 911.25 to 459 pixels. The candidate passes the development promotion checks
and is frozen before the reused held-out evaluation.

| Reused-test check | Current pair | Development-selected candidate |
| --- | ---: | ---: |
| One-step y errors / 45,245 | 3 | 4 |
| One-step vy errors / 45,245 | 1 | 1 |
| Joint one-step errors / 45,245 | 3 | 4 |
| Failed complete 128-step windows / 39,203 | 224 | 352 |
| Fully exact 128-step windows | 99.4286% | 99.1021% |
| Mean 128-step endpoint y error | 1.5996 px | 2.3451 px |
| Worst 128-step endpoint y error | 897.75 px | 897.75 px |

The candidate leaves all three old test errors and introduces one additional
paddle displacement error, confusing bounce timing while still predicting the
correct next velocity. It is therefore not promoted. No fitting follows this
test result. The 64 untouched episodes remain reserved. Overlapping windows are
not independent collision failures, and all nonvertical state remains supplied.

The rejected research checkpoint is
`runs/vertical-refinement-20260919/calibrated-s2026/best.pt`, SHA-256
`e1ff4dc450d3df2992673d4246289c163f531ed2b34c658fea149c9266b9f882`.
Its metadata records development acceptance, not deployment approval. The final
rejection and retained checkpoint are explicit in
`logs/vertical-refinement-20260919/decision.json` and the `disposition` field of
`runs/vertical-refinement-20260919/result.json`. Preparation, training,
calibration, and evaluation scripts and receipts are in that log directory.
Original nested weights remain exact, checkpoint reload reproduces predictions,
and the dataset and shared runner/player are unchanged.

Verification passes 391 tests with two skipped, Ruff, frozen dependency sync,
and whitespace checks. Focused coverage checks exact zero-initialization,
frozen parents, head training, unchanged flight, and checkpoint reload.

## Explicit vertical collision timing

Test whether predicting the velocity used in each of the two native movements
resolves the remaining bounce-timing errors. The `vertical_collision_timing`
model uses the same 118 source values as the existing pair. It classifies one
of 64 ordered velocity pairs, then computes next combined y as current combined
y plus both velocities, and next vy as the second velocity. No native collision
rules execute during inference. Frozen parent features feed upper-field
135→128→128→64 and paddle 97→256→256→256→64 ReLU heads, initialized from the
parent hidden layers. Ordinary flight uses the existing pair.

An instrumented offline native replay audits 5,271,950 nonterminal training-pool
sources without mismatches. The timing labels count 4,966,741 unchanged
movements, 105,682 first-frame changes, and 199,527 second-frame changes; none
change velocity on both frames. The first velocity is also derivable as next
combined y minus current combined y minus next vy. This target decomposition
adds no source information. All labels and features remain in RAM; the dataset
is unchanged.

Reuse the preceding refinement split: 1,568 fitting episodes / 4,529,910 sources,
256 development episodes / 742,040 sources, and the original 32 validation
episodes / 88,590 sources. The frozen parents trained on the development
episodes, although the new heads do not. These development results therefore
cannot demonstrate generalization to episodes unseen by the whole model.

Train seeds 91 and 2026 for 10,000 updates each, using AdamW with hidden-layer
learning rate 0.0001, output-layer learning rate 0.001, cosine decay to 0.00001,
weight decay 0.0001, and gradient clipping at 5. Each region samples 192 unchanged,
96 first-frame, and 96 second-frame cases per update with replacement. The loss
is joint velocity-pair cross-entropy plus 0.25 times the negative log likelihood
of the timing category, marginalized from those joint probabilities.

Standalone timing heads do not pass the validation/development selection gate.
Before reading test data, calibrate a frozen-position prior: add a weighted log
probability of each pair's summed displacement to its learned joint score.
Unsupported displacements receive log-prior −30. Search weights 0, 0.025, 0.05,
0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, and 8 on the final checkpoints;
require zero original-validation errors and no worse development performance.
Choose the smallest weight on a tie. This chooses one coherent outcome, without
averaging physical coordinates.

Seed 2026 with weight 2 on both regional heads reaches zero validation and
development joint errors. All 78,806 validation and 659,845 development complete
128-step windows are exact. Freeze and hash this candidate before evaluating the
same 16 reused held-out episodes; keep the 64 untouched episodes reserved.

| Reused-test check | Existing pair | Timing candidate |
| --- | ---: | ---: |
| One-step y errors / 45,245 | 3 | 4 |
| One-step vy errors / 45,245 | 1 | 2 |
| Joint one-step errors / 45,245 | 3 | 4 |
| Failed complete 128-step windows / 39,203 | 224 | 352 |
| Fully exact 128-step windows | 99.4286% | 99.1021% |
| Mean 128-step endpoint y error | 1.5996 px | 2.7291 px |
| Worst 128-step endpoint y error | 897.75 px | 897.75 px |

Reject this candidate. It retains the three existing joint errors and adds a
paddle timing error at episode 1020, step 2301: predicted y is 173 rather than
179.75, despite correct next vy of −3.375. One old y error also gains an incorrect
vy prediction. No fitting follows the test. Windows overlap and other state
fields and segment boundaries remain supplied; these are partial-state rollouts.

Retain `runs/vertical-pair-20260919/coupling-s2026/best.pt`. The rejected research
checkpoint is `runs/vertical-timing-20260919/prior-both-s2026/best.pt`, SHA-256
`66a8af7e46c4e64700a67602fed9c69dce79f87eafe3b7a0587fa2154037e70d`.
Scripts, plans, label audit, pretest freeze, and rejection receipt are in
`logs/vertical-timing-20260919`; the final disposition is also in
`runs/vertical-timing-20260919/result.json`. Training labels and features are not
persisted. Verification passes 394 tests with two skipped, Ruff, and frozen
dependency sync. Focused tests cover timing-dependent displacement, inactive
region fallback, frozen parents, gradients, empty batches, and checkpoint reload
with and without the prior.

The next experiment should establish development episodes unseen by every
trained component before trying another architecture. This result does not
establish missing state or prove why the generalization gap occurs.

## Vertical development split with untrained episodes

The 2026-09-19 follow-up establishes this split without changing dataset files or
training any model. It applies to future vertical-pair experiments; earlier
experiment scripts retain their historical split definitions.

| Role | Episodes | Permitted use |
| --- | ---: | --- |
| Fit | 1,824 | Gradient fitting, augmentation donors, fitted preprocessing |
| Legacy validation | 32 | Existing regression checks; already used for selection |
| Development | 400 | Diagnosis, calibration, model selection; no gradient fitting |
| Final test | 64 | Reserved until the candidate and evaluation procedure are frozen |

The development set is the union of all previously inspected held-out episode
allocations, including the 16 most recently reused episodes. No component of the
current pair trained on these 400 episodes. They have informed earlier research
decisions, so this is a development baseline, not a new unbiased test result.
The remaining 64 held-out episodes stay untouched. The former 256 development
episodes are part of the fitting pool because frozen parents trained on them.
No life changes split: every life inherits its original episode's role.

Audit the complete learned ancestry, rather than only the last correction head.
Compare the current nested weights exactly against eight checkpoint records:
the pair, y model, vy model, acceleration model, horizontal router, paddle speed,
paddle direction, and intermediate paddle model. Also verify the older global
horizontal initialization's checkpoint hash and training-cache identity. Its
weights subsequently changed during training-only fine-tuning. These nine
provenance entries have zero training overlap with development or final test.
Check every saved held-out allocation against the frozen episode partition.
This audit covers the current pair and its dependencies, not arbitrary future
components; a new dependency requires its own ancestry audit.

The immutable episode lists, component hashes, training episode IDs, and exposure
records are in `logs/vertical-development-20260919/split.json`. Recheck them with:

```bash
PYTHONPATH=. uv run python logs/vertical-development-20260919/establish.py
```

The audit refuses to overwrite an existing plan whose partition or provenance
has changed. Use an explicitly versioned plan for a future change. No transition
targets are read to establish this partition; only metadata, historical receipts,
and checkpoints are inspected.

The paired model's one-step baseline is evaluated on all 400 development episodes
using filtered reads, RAM-only source reconstruction, and the existing life
boundaries. The loader checks episode membership before each read and again on
returned rows. Negative checks confirm that fitting, legacy-validation, and
final-test IDs are rejected by the development reader. Native replay audits the
prepared targets without providing collision decisions to model inference.

| One-step development metric | Existing pair |
| --- | ---: |
| Nonterminal transitions | 1,152,531 |
| Incorrect combined y | 78 |
| Incorrect vy | 68 |
| Either output incorrect | 99 |
| Both outputs exactly correct | 99.9914% |

This is a new measurement of unchanged weights, not a model improvement. Inputs
are recorded current states; the score does not measure accumulated rollout
error. No new 128-step score or final-test score is claimed.

Run the same bounded baseline with:

```bash
PYTHONPATH=. uv run python logs/vertical-development-20260919/baseline.py
```

The baseline plan is written before loading development targets. Results and
per-episode error witnesses are saved in that directory, while feature arrays
remain in RAM. Future tuning must keep these development IDs out of every
gradient-fitting stage, augmentation pool, fitted normalizer, and learned target
vocabulary, including those of frozen parents. Select using the full development
set and the planned feedback metrics, retain legacy regression checks, then freeze
the candidate before any final-test read. Do not report later development gains
as fresh-test evidence.

The completed baseline confirms unchanged dataset file sizes/modification times
and checkpoint hash. `verification.json` records rejection of a modified frozen
partition and a deliberately injected training overlap in a frozen parent, plus
the three development-reader rejection checks. These checks modify only in-memory
test records or temporary files, not the real dataset or provenance receipts.

## Development vertical feedback and factorized paddle outcomes

On 2026-09-21, evaluate the unchanged vertical pair on all 400 audited development
episodes. Keep the same 1,152,531 nonterminal sources, reference life boundaries,
and supplied nonvertical state. Score every complete window at 8, 32, and 128
steps, feeding back both predicted combined y and vy. All 64 final-test episodes
remain untouched.

| Horizon | Eligible windows | Windows with any error | Entire window exact | Mean endpoint y error |
| --- | ---: | ---: | ---: | ---: |
| 8 | 1,145,067 | 739 | 99.9355% | 0.0124 px |
| 32 | 1,119,687 | 2,744 | 99.7549% | 0.1237 px |
| 128 | 1,024,358 | 9,027 | 99.1188% | 1.5756 px |

The 128-step failures originate at 97 distinct first-error transitions. Two of
the 99 one-step errors cannot be a first failure in an eligible 128-step window.
Windows overlap and are not independent failure events. With only y fed back,
mean endpoint y error is 0.0257 pixels; with only vy fed back it is 0.0097 pixels.
Joint feedback reaches 1.5756 pixels on average and 1,012.5 pixels at worst.
These are partial-state diagnostic trajectories, not complete playable rollouts.

Group first failures by the recorded event, assigning simultaneous side-wall
contacts to their paddle or brick event:

| First-error event | Failed 128-step windows |
| --- | ---: |
| Paddle | 3,914 |
| Brick | 2,534 |
| No recorded collision | 2,299 |
| Side wall only | 280 |
| Ceiling | 0 |

The paddle source region contains 55 of the 99 one-step errors, including false
collision predictions on transitions with no recorded collision. The upper
region contains the other 44. Thus event labels and source regions answer
different questions; a no-collision error need not originate in ordinary flight.

Check target consistency before changing the predictor. Across 440,833 fitting
and 96,005 development paddle sources, full 118-field inputs have 17,268 repeated
input groups. The reduced nine paddle fields and actual 97-feature paddle
encoding each have 57,615 repeated groups. None has conflicting displacement/vy
targets. The development subset alone also has zero conflicts, and all fitting
native movements match the audited labels. This finds no observed missing-input
or exact encoder-aliasing problem; it does not prove full observability for
unseen states or establish that the representation is easy to learn.

The one targeted experiment changes the output decomposition. Keep the parent
frozen and replace only the descending paddle region with
`paddle_vertical_pair`. Its shared 97→256→256→256 ReLU trunk copies the original
y head's hidden weights. Separate outputs predict three timing classes and four
outgoing velocities: −3.375, −2, −1.5, and −1 pixels per native frame, derived
only from fitting labels. The model has 158,471 new trainable parameters.
Decoded timing and velocity determine one consistent displacement/next-vy pair.
No new input fields, collision rules, or target-dependent routing run at inference.

The audited fitting labels contain 347,444 no-bounce cases, 19,615 first-frame
bounces, and 73,774 second-frame bounces. There are no double changes in this
region. Fit seeds 91 and 2026 for 12,000 updates each with AdamW, hidden learning
rate 0.0001, output learning rate 0.001, weight decay 0.0001, cosine decay to
0.00001, and clipping at 5. Each update draws 192/96/96 examples from the three
timing classes and an independent 64 examples per bounce-velocity class. The
loss sums timing cross-entropy and bounce-only velocity cross-entropy.

Check every 500 updates using the full development set. Require zero original
validation joint errors, then minimize failed complete 128-step development
windows, breaking ties by joint one-step errors and earlier checked seed/update.
Window failure counts can be computed exactly from reference-input mistakes:
until the first error, recursive and reference inputs are identical. Verify this
shortcut against the full baseline, then run actual candidate rollouts to measure
drift. Acceptance also requires no increase in mean or maximum endpoint y error.
The existing checkpoint remains an immutable reference throughout.

Neither seed yields a zero-error original-validation checkpoint at the declared
checks. For diagnosis, select the candidate with fewest validation errors, then
fewest development failed128 windows and joint errors. This selects seed 2026
at update 3,500. Freeze it before actual feedback evaluation; do not adjust its
weights afterward.

| Check | Existing pair | Factorized candidate |
| --- | ---: | ---: |
| Development y errors / 1,152,531 | 78 | 71 |
| Development vy errors | 68 | 68 |
| Development joint errors | 99 | 85 |
| Failed 8-step windows / 1,145,067 | 739 | 589 |
| Failed 32-step windows / 1,119,687 | 2,744 | 2,126 |
| Failed 128-step windows / 1,024,358 | 9,027 | 7,055 |
| Entire 128-step window exact | 99.1188% | 99.3113% |
| Mean 128-step endpoint y error | 1.5756 px | 0.9690 px |
| Maximum 128-step endpoint y error | 1,012.5 px | 945 px |
| Original-validation joint errors / 88,590 | 0 | 1 |
| Original-validation failed128 / 78,806 | 0 | 98 |

The development improvement includes 21.8% fewer failed 128-step windows. Paddle
events at the first failure fall from 3,914 to 2,033; brick first failures stay
at 2,534. Of the candidate's 41 remaining development paddle errors, 28 concern
timing and 13 concern outgoing speed. The original-validation regression at
episode 1369, step 2902, predicts a first-frame bounce rather than a second-frame
bounce. Next vy is correctly −3.375, but y is 171.125 instead of 177.875.

Reject the candidate under the predeclared zero-validation-error requirement,
despite its better development averages. Retain
`runs/vertical-pair-20260919/coupling-s2026/best.pt`. The diagnostic candidate is
`runs/vertical-rollout-20260921/candidate.pt`, SHA-256
`db1ec5e5e90649954f91f4218d6f47f99c1ea27334a8fee836eaf7fd2be86043`.
Its weights, model specification, training settings, and selection record are
self-contained; split provenance is pinned to
`4039e3136568a0471aa39f2704f24127f383babdf78d35f11f9e710d04e881a0`.
The final rejection is recorded in `logs/vertical-rollout-20260921/decision.json`
and `runs/vertical-rollout-20260921/result.json`.

Plans, preparation and conflict audits, per-chunk feedback metrics, error
witnesses, and experiment scripts are in `logs/vertical-rollout-20260921`.
Feature arrays remain in RAM. Checkpoint reload reproduces every development
prediction exactly; parent weights, dataset file inventory, and the original
checkpoint hash are unchanged. No final-test targets were read. Verification
passes 395 tests with two skipped, Ruff, and whitespace checks. Focused coverage
checks all three timing decodes, no-bounce independence from outgoing speed,
frozen parents during optimization, unchanged fallback, empty batches, and reload.

## Paddle-edge feature experiment

The 2026-09-21 follow-up changes only the factorized model's input representation.
Set `edge_features=true` and append eight signed distances, four for the current
geometry and four for a one-native-frame estimate. The distances measure
ball-right minus paddle-left, paddle-right minus ball-left, ball-bottom minus
paddle-top, and paddle-bottom minus ball-top. Coordinates are rasterized with
floor, matching the recorded integer pixel geometry. Distances are divided by 16.
RAM-coordinate paddle bounds are 180..183, with source-dependent width. Ball
raster extents are one pixel right and three pixels down. The estimate uses
current ball velocity and the existing frozen intermediate-paddle predictor.
It executes no collision test, bounce, reflection, or native transition.

The original 97 features remain intact. The wider shared trunk is
105→256→256→256 ReLU, followed by the same three timing outputs and four outgoing
velocities, for 160,519 fitted parameters. Copy the original parent hidden
weights and initialize the eight added input columns to zero. Preserve the
control's random stream so output initialization and training draws use the
same seeds. Floating-point summation may differ after widening the matrix;
initial-output tests allow float32 roundoff.

Retain seeds 91 and 2026, 12,000 updates each, the original optimizer and learning
rates, balanced timing and bounce-speed batches, and checks every 500 updates.
Use the same 1,824 fitting episodes, 400 development episodes, and 32 original
validation episodes. Derive features in RAM without adding dataset columns.
Reserve all 64 final-test episodes. Preserve the same selection and acceptance
rules, and compare against both the original pair and the previous factorized
candidate. Training plans record the dataset manifest, fitting episode IDs,
split identity, control checkpoint hash, model implementation hash, geometry,
and resolved architecture in `logs/vertical-edges-20260921`.

Neither seed yields a zero-error original-validation checkpoint. The same
diagnostic selection rule chooses seed 2026 at update 3,500. All 2,048 added
first-layer weights become nonzero, so the extra features participate in the
fitted model. Freeze the checkpoint before actual recursive evaluation.

| Check | Original pair | Factorized control | With edge distances |
| --- | ---: | ---: | ---: |
| Joint one-step development errors / 1,152,531 | 99 | 85 | 85 |
| Failed 128-step windows / 1,024,358 | 9,027 | 7,055 | 7,026 |
| Entire 128-step window exact | 99.1188% | 99.3113% | 99.3141% |
| Mean 128-step endpoint y error | 1.5756 px | 0.9690 px | 0.9173 px |
| Maximum 128-step endpoint y error | 1,012.5 px | 945 px | 918 px |
| Original-validation joint errors / 88,590 | 0 | 1 | 1 |
| Original-validation failed128 / 78,806 | 0 | 98 | 98 |

The extra features reduce failed development windows by only 29 relative to the
factorized control. Its 41 paddle errors become 27 timing errors and 14 speed
errors, compared with 28 and 13. The same original-validation case, episode 1369
step 2902, still predicts a first-frame bounce instead of a second-frame bounce.
Its predicted y remains 171.125 rather than 177.875, with correct next vy −3.375.
Thus the tested geometry representation gives a modest drift improvement but
does not solve the validation regression under the fixed training budget.

Reject the candidate under the unchanged acceptance rule and retain the original
pair. Do not infer that additional history or dataset fields are needed from
this result. The diagnostic checkpoint is
`runs/vertical-edges-20260921/candidate.pt`, SHA-256
`97aeb9fe14227a297598432b21bfb1afe3b91b85b1eec6b1cc336797d70667f3`.
Final rejection and the three-model comparison are in
`logs/vertical-edges-20260921/decision.json` and `comparison.json`; the run folder
also records the decision. No further fitting follows the frozen evaluation.

Verification passes 397 tests with two skipped, Ruff, and whitespace checks.
New coverage checks signed geometry, unchanged source tensors, zero extra-column
initialization, preserved random state, frozen-parent training, and reload with
the optional features enabled or disabled. Every new candidate development
prediction matches checkpoint reload exactly. All 96,005 archived control paddle
predictions are reproduced exactly after the code change, and its nonpaddle
parent weights remain unchanged. Dataset file inventory and both reference
checkpoints are unchanged. No final-test targets were read. No feature caches
were written.

## Checkpoint-trajectory dataset comparison

The September 21 comparison tests
[`tsilva/gradlab-breakout-6127e81d`](https://huggingface.co/datasets/tsilva/gradlab-breakout-6127e81d)
at immutable revision `79827bb74fd7a881af7771f111770504c1565d0d` against the
existing paddle training pool. This is a state-only experiment for the paddle
vertical predictor, not an encoder, image reconstruction, or full emulator comparison.
The new snapshot has 500 episodes and 2,059,758 transitions from ten policy
checkpoints between 10M and 100M training steps. Its 400/50/50 episode allocation
groups environment seeds across checkpoints, with 40/5/5 independent seed groups.
Use its published train and validation allocations; reserve test transition targets.
None of its train/validation environment seeds overlap the old dataset's seeds.

Read the nested split Parquet files through the experiment adapter. The new
snapshot lacks the added controller columns. Reconstruct controller state in RAM
from reset seeds and executed actions, verifying the derived no-op counts against
all 450 train/validation session records. All 1,854,012 recorded paddle movements
match replay. After existing life/startup/quality filtering, native rules reproduce
all 1,845,497 usable ball transitions, including vertical fraction and bounce timing.
No dataset columns or persistent feature caches are written. Native rules supply
offline audit labels only; model inference remains learned.

Keep the edge-feature model, frozen dependencies, initialization, losses, optimizer,
and 12,000-update budget identical. Train three arms for seeds 91 and 2026:
old data, new data, and a 50/50 mixture within each timing/speed class. Every arm
uses 384 timing examples and 256 bounce-speed examples per update, AdamW with
hidden/head learning rates 0.0001/0.001, cosine decay to 0.00001, weight decay
0.0001, and gradient clipping at 5. Evaluate fixed update 12,000 in every run;
intermediate metrics do not select checkpoints. The architecture is
105→256→256→256 ReLU with three timing and four outgoing-velocity logits.
Frozen helpers and copied initial hidden weights were trained on old data, so
this compares fitting data for an existing branch, not entire models trained
from scratch independently on each dataset.

The old fitting pool has 440,833 paddle examples from 1,824 episodes; the new pool
has 134,341 from 400 episodes. Their nine-field source inputs have 319,520 and
127,089 distinct states, with 3,088 shared. The new pool therefore adds 124,001
states absent from old fitting. Neither pool contains repeated source inputs
with conflicting y/vy targets. The frozen intermediate-paddle helper has 20 errors
on new fitting sources and six on new validation sources; it is held constant
across all arms.

Joint y/vy errors below count a source once if either output is wrong. Each cell
lists seeds 91 / 2026, using exactly the same evaluation examples for every arm.

| Fitting data | Old development / 96,005 | Old validation / 7,164 | New validation / 17,074 |
| --- | ---: | ---: | ---: |
| Old | 43 / 39 | 2 / 2 | 79 / 68 |
| New | 11 / 11 | 1 / 1 | 15 / 17 |
| 50/50 mixture | 12 / 10 | 0 / 1 | 19 / 19 |

New-only fitting reduces mean errors from 41 to 11 on old development, and from
73.5 to 16 on new validation: reductions of 73.2% and 78.2%. Mean exact paddle
accuracy becomes 99.9885% and 99.9063%. Both old-only runs fit their old training
pool with zero errors; both new-only runs fit the new pool with zero errors.
Thus these runs expose a generalization gap rather than an inability to fit the
observed training mappings. They do not isolate checkpoint diversity from every
other difference between the collections.

For complete vertical paths, keep other state fields and life boundaries supplied.
Count a window as failed at its first incorrect y/vy prediction. For a deterministic
model this gives the same whole-window exactness as recursive feedback, because
inputs remain identical to the reference until that first mistake. This check
does not measure drift after a mistake. The original 9,027 failed development
windows and both archived old-data control counts reproduce exactly.

| Fitting data | Failed old-development 128-step windows / 1,024,358 | Failed new-validation 128-step windows / 174,841 |
| --- | ---: | ---: |
| Old | 7,261 / 6,887 | 13,216 / 11,948 |
| New | 5,518 / 5,535 | 8,056 / 8,120 |
| 50/50 mixture | 5,580 / 5,283 | 8,565 / 8,441 |

The new data helps this branch on both distributions under the matched budget.
Mixing does not beat new-only on new validation, although mixed seed 91 preserves
zero old-validation errors. New-only fixes the previous errors at episodes 1369
and 2041 but introduces a timing error at episode 834, step 2081. Do not replace
the current reference from this dataset comparison: no candidate has undergone
the existing post-error mean/max drift promotion checks. Upper-region predictors
also remain frozen, accounting for 44 old-development and 62 new-validation
one-step errors outside the trained paddle region.

Plans, adapters, coverage, per-seed metrics, witnesses, hashes, and reproduction
instructions are under `logs/dataset-6127e81d-20260921`; all six self-contained
checkpoints are under `runs/dataset-6127e81d-20260921`. Checkpoint reload reproduces
each evaluated paddle error count, and both old controls reproduce prior one-step
and exact-path scores. All 22 downloaded train/validation files match published
SHA-256 checksums. Dataset file inventory and the original reference hash remain
unchanged. Both datasets' final-test transition targets remain reserved. Experiment
scripts compile; Ruff and whitespace checks pass. Shared training/player code and
the direct CNN baseline are unchanged.

## Upper-screen dataset continuation

The next September 21 experiment freezes the improved paddle branch at
`runs/dataset-6127e81d-20260921/new-s91.pt`, SHA-256
`ffb2bb65cf520b5455e30f9fe310ed4e7f5773e27cc55fb940a93e1a6b98b523`.
This fixed first-seed checkpoint is an experimental starting point, not the
promoted current reference. Its upper heads still have the original weights.
Compare further upper-head training on old data with training on the new
checkpoint-trajectory dataset. Keep the immutable dataset revisions and
previous episode allocations; no final-test transition targets are read.

Train only four existing modules: the y and vy brick-cell encoders and their
upper output networks. Each cell encoder is 13→64→64 ReLU, shared across 108
brick locations, followed by occupied-cell max pooling. Combine the 64 pooled
features with 71 numeric/binary geometry features. The separate heads are
135→128→128→19 displacement logits and 135→128→128→8 velocity logits. These
modules contain 81,435 trainable parameters. Keep the paddle predictor,
horizontal dependencies, flight heads, and learned y/vy correction network
frozen. Architecture, input fields, class vocabularies, and inference routing
are unchanged. Source RAM y <= 100 selects the upper region.

The fitting pools have 1,782,412 old and 619,839 new upper examples. For each
pool, run seeds 91 and 2026 for exactly 12,000 updates. Each batch contains 32
velocity-changing and 96 unchanged examples. Sum displacement and raw velocity
cross-entropies with equal weights. Use AdamW at 0.0001, cosine decay to
0.00001, weight decay 0.0001, and gradient clipping at 5. Every arm starts from
the same checkpoint and evaluates its fixed final update. Training-loss logs
do not select intermediate checkpoints. Controller, fraction, collision-memory,
and native-label reconstruction remain RAM-only, with the same replay audits.

Each error cell below lists seeds 91 / 2026. Joint errors count a source once
when either final y or corrected vy is wrong. The baseline is the frozen
improved-paddle checkpoint before this continuation.

| Upper training | Old development / 390,843 | Old validation / 31,429 | New validation / 75,297 |
| --- | ---: | ---: | ---: |
| No continuation | 44 | 0 | 62 |
| Old data | 32 / 29 | 3 / 4 | 43 / 41 |
| New data | 60 / 54 | 8 / 6 | 19 / 13 |

New-data fitting reduces mean upper errors on new validation from 62 to 16,
but increases old-development errors from 44 to 57. The matched old-data
control improves the two larger sets to 30.5 and 42 mean errors, while also
regressing on legacy validation. This continuation therefore shows a dataset
tradeoff, unlike the previous paddle experiment's improvement on both sets.
The raw heads also participate: new-data fitting yields 46/40 y errors and
64/54 raw vy errors on old development, versus baseline 30 and 56. The frozen
correction still reduces velocity errors in both runs; it is not the sole
source of the regression. Ordinary steps with no recorded collision account
for 29/28 old-development joint errors after new fitting, versus 17 before.

Run actual recursive y/vy feedback at horizons 1, 8, 32, and 128 from every
eligible source in all three evaluation sets. Supply other state fields and
reference life boundaries. Unlike the preceding dataset comparison's exact-path
count, this evaluation explicitly feeds incorrect predictions forward and
measures subsequent drift. Windows overlap and never cross lives or episodes;
these are partial-state diagnostics, not full-game simulation.

The next table averages the two fixed-seed runs. The starting checkpoint is
identical for both seeds and appears once. Full vertical paths include the
unchanged paddle branch and flight predictions.

| Upper training | Old dev failed128 / 1,024,358 | Old dev mean endpoint y error | New val failed128 / 174,841 | New val mean endpoint y error |
| --- | ---: | ---: | ---: | ---: |
| No continuation | 5,518 | 0.8729 px | 8,056 | 8.3158 px |
| Old data | 4,077 | 0.7643 px | 6,170 | 5.9026 px |
| New data | 7,462 | 1.5529 px | 3,270.5 | 3.0530 px |

Maximum old-development error remains 992.25 pixels in every run. Maximum new-
validation error decreases from 909.25 to 870.75 pixels in both new-data runs.
Legacy validation has 128 failed windows and 0.0100-pixel mean error before
continuation, due to the frozen paddle error. Old-data continuation raises this
to 512/640 windows and 1.5515/1.9633 pixels; new-data continuation raises it to
924/844 windows and 1.9619/1.9645 pixels. All continued models introduce upper
errors there. Keep the original promoted reference unchanged. A mixed upper
training pool is a proposed next experiment, not a completed remedy.

The current reference remains
`runs/vertical-pair-20260919/coupling-s2026/best.pt`. Four complete diagnostic
checkpoints are stored under `runs/upper-dataset-20260921`; plans, code hashes,
preparation receipts, per-head metrics, all recursive error witnesses, and the
decision are under `logs/upper-dataset-20260921`. The data loader, model registry,
player, and direct CNN baseline have no implementation changes in this experiment.
Every parameter outside the four intended modules remains bitwise equal to the
starting checkpoint, including after portable reload. Recursive evaluation
reproduces the previous baseline exact-window counts, and full one-step errors
equal measured upper errors plus the unchanged paddle errors on every set.
Dataset file inventory and checkpoint hashes remain unchanged. Experiment
scripts compile; Ruff and whitespace checks pass. No persistent feature caches
or new dataset columns are written, and final-test targets remain reserved.


## Fixed-size mixed upper-screen fitting

On 2026-09-22, test whether mixing old and new upper-screen fitting examples
improves the preceding continuation tradeoff at fixed data size. Keep the same
improved-paddle parent, frozen dependencies and correction, four trainable upper
modules, two optimization seeds, 12,000 updates, batch size 128, optimizer and
learning-rate schedule. Use only fitting trajectories for sampling and training.

Each arm contains 619,839 transitions. A separate NumPy RNG with seed 22092026
samples without replacement. The old-only control uses 619,839 old rows. The
mixture uses 309,920 old rows nested in that control and 309,919 new rows. Each
mixed batch draws 16 changed-velocity and 48 unchanged rows from each dataset.
The old-only control draws 32 changed and 96 unchanged rows. Distinct rows can
still contain repeated states and temporal correlations. Both optimization seeds
use the same sampled pools, so this checks optimization variation, not variation
across dataset subsets.

Reuse the preceding new-only runs, whose full new pool already has 619,839 rows.
Verify their checkpoint hashes and matching parent, trainable modules, model
specification, episode roles and optimization settings. Evaluate each newly
trained model at its fixed final update. Do not select intermediate checkpoints.

Error counts below are joint upper y/vy errors for seeds 91 / 2026. The previous
full-old arm is context and uses more fitting rows than the three matched arms.

| Fitting pool | Fitting rows | Old development / 390,843 | Legacy validation / 31,429 | New validation / 75,297 |
| --- | ---: | ---: | ---: | ---: |
| Starting model | N/A | 44 | 0 | 62 |
| Previous full old pool | 1,782,412 | 32 / 29 | 3 / 4 | 43 / 41 |
| Size-matched old pool | 619,839 | 37 / 35 | 3 / 2 | 47 / 46 |
| New pool, reused control | 619,839 | 60 / 54 | 8 / 6 | 19 / 13 |
| 50/50 mixture | 619,839 | 33 / 33 | 5 / 4 | 23 / 21 |

The mixture reduces mean old-development errors from 57 with new-only training
to 33, while new-validation errors increase from 16 to 22. It improves both
larger sets versus the starting model and the size-matched old-only control.
Relative to that control, mean errors fall from 36 to 33 on old development and
46.5 to 22 on new validation. This supports a benefit from fitting-data
composition at fixed row count and update budget. It does not establish that
more data is unnecessary, or isolate policy-checkpoint diversity from the other
differences between datasets. Legacy validation still regresses from zero upper
errors to 5/4, so neither mixed candidate replaces the current reference.


Actual recursive y/vy feedback uses every eligible window at horizons 1, 8, 32
and 128. Other state fields and reference life boundaries are supplied; windows
overlap. These results measure partial-state dynamics, not a complete simulator.
The following values average both optimization seeds.

| Fitting pool | Old dev failed128 / 1,024,358 | Old dev endpoint y MAE | New val failed128 / 174,841 | New val endpoint y MAE |
| --- | ---: | ---: | ---: | ---: |
| Starting model | 5,518 | 0.8729 px | 8,056 | 8.3158 px |
| Previous full old pool | 4,077.0 | 0.7643 px | 6,170.0 | 5.9026 px |
| Size-matched old pool | 4,794.0 | 0.9738 px | 6,304.5 | 7.0994 px |
| New pool | 7,462.0 | 1.5529 px | 3,270.5 | 3.0530 px |
| 50/50 mixture | 4,494.0 | 0.8977 px | 3,805.5 | 3.8842 px |

Mixing lowers old-data drift relative to new-only fitting, from 1.5529 to
0.8977 pixels, while raising new-data drift from 3.0530 to 3.8842. Against the
starting model, mean old-data drift is slightly worse despite fewer one-step
errors and fewer failed windows. The two mixed runs have old-data MAE 0.8633
and 0.9320 pixels. Maximum old-development error remains 992.25 pixels; new-
validation maxima are 870.75 and 897.75. Legacy-validation failed windows rise
from 128 before continuation to 768/535, with endpoint MAE 2.3329/1.1133 pixels
and maximum 827.25. The mixture improves the broader balance but does not pass
reference-promotion checks.

Keep the reference and paddle parent unchanged. Four new checkpoints are under
`runs/upper-mixed-20260922`. Plans, subset hashes, source scripts, audited data
preparation, per-head metrics, recursive witnesses, control verification and
reproduction instructions are under `logs/upper-mixed-20260922`. All weights
outside the four intended upper modules remain bitwise unchanged after portable
reload, and reloaded metrics agree. Old dataset and immutable new snapshot file
inventories are unchanged. Scripts compile, Ruff and whitespace checks pass.
No production code changed in this experiment, so the full test suite was not
repeated. No dataset columns or persistent feature caches were written, and
both final-test target sets remain reserved.

## Horizontal pair on new checkpoint trajectories

On 2026-09-22, fit and evaluate next x and vx using only the train and validation
partitions of `tsilva/gradlab-breakout-6127e81d`, revision
`79827bb74fd7a881af7771f111770504c1565d0d`. Preparation reconstructs controller,
fraction and contact features in RAM and audits them against recorded native
transitions. There are 1,640,422 fitting and 205,075 validation sources after the
existing startup, suspect-layout and terminal filters. Episode and life boundaries
remain explicit; no final-test targets are read and no dataset columns change.

Initialize from `runs/ball-position-20260918/x-s2026/best.pt`, SHA-256
`a7891b6da4c444d79b4ed5e1ac01c0e06825cf65209ff1108ca390e0528dcd37`.
This imports historical x weights and frozen geometry dependencies. New-only
fitting does not mean the whole model was trained from scratch on the new dataset.
Evaluate the historical x model and its routed vx parent as the old baseline.

Compare independent x/vx predictors with a shared-hidden-layer pair. Both have
identical inputs and initial logits. The independent arm copies the pretrained x
hidden layers into a separate velocity branch; the shared arm uses one set for
both tasks. This tests sharing within a controlled architecture; the new eight-
class velocity head is different from the historical routed vx baseline.

| Region | Hidden architecture | Outputs |
| --- | --- | --- |
| Upper, RAM y <= 100 | Shared per-brick 13→64→64 ReLU, occupied-cell max pool, 135→128→128 ReLU | 22 x-displacement logits and 8 vx logits |
| Near paddle, RAM y 160..183 and vy > 0 | 97→256→256→256 ReLU | Same two heads |
| Remaining field | 31→128→128 ReLU | Same two heads |

The independent arm duplicates cell and region hidden layers for vx. It has
447,962 trainable parameters, versus 231,706 for shared layers. All nested
geometry and intermediate-paddle dependencies stay frozen in evaluation mode.
Both next-state outputs use the same current state. No native rules, successor
fields or event labels run at inference.

For seeds 91 and 2026, warm only the new vx output heads for 2,000 updates with
fixed pretrained features. Give both arms those same heads and initial hidden
weights, then train for 12,000 updates. The batch sampler uses its own seeded RNG,
so both arms receive identical batches. Every batch draws 32 transitions from
each region/velocity-change stratum, for 192 total. Fit strata contain 600,631 /
19,208 unchanged/changed upper transitions, 115,446 / 18,895 paddle transitions,
and 869,096 / 17,146 flight transitions.

Use equal displacement and velocity cross-entropies. AdamW uses learning rate
0.0001 for hidden layers and x heads, 0.001 for new vx heads, weight decay 0.0001,
and cosine decay to 10% of each initial rate. No global gradient clipping couples
the independent losses. Evaluate fixed final checkpoints without intermediate
validation selection. One loader serves both objectives, while the independent
arm's trainable parameters and gradients remain disjoint.


| Model | x errors / 205,075 | vx errors / 205,075 | Joint errors / 205,075 | Joint exact accuracy |
| --- | ---: | ---: | ---: | ---: |
| Historical x and routed vx | 149 | 118 | 231 | 99.8874% |
| Independent, seed 91 | 67 | 63 | 103 | 99.9498% |
| Independent, seed 2026 | 71 | 68 | 112 | 99.9454% |
| Shared, seed 91 | 74 | 65 | 111 | 99.9459% |
| Shared, seed 2026 | 81 | 72 | 118 | 99.9425% |

Mean joint error counts decrease from 231 to 107.5 for independent fitting and
114.5 for shared fitting. The shared arm uses about 48% fewer trainable parameters
but has more one-step errors in both matched seeds. This is an observed
capacity/sharing tradeoff; the experiment does not isolate its mechanism or
establish an optimal loss balance. The historical baseline also differs in its
velocity architecture, so its improvement combines new-data fitting with that
architecture change. The matched arms isolate sharing more closely.

Both arms make zero errors on 112,704 flight-region sources. Near-paddle sources
account for 94/96 independent joint errors and 98/102 shared errors. On 3,121
actual paddle-hit transitions, independent errors are 73/72 and shared errors
77/79. Mean paddle-hit joint accuracy is therefore about 97.68% and 97.50%, much
lower than the overall transition accuracy. Upper-region errors are 9/16 and
13/16. Fixed strided fitting probes of 20,006 examples have 5/3 independent joint
errors and 5/5 shared errors; these are sampled diagnostics, not full fitting-set
accuracy.

Run actual recursive x/vx feedback, including after incorrect predictions, for
all eligible validation windows at horizons 1, 8, 32 and 128. Supply other state
fields and reference life boundaries. The table averages seeds for trained arms.

| Model | Entirely exact128 / 174,841 windows | Failed128 | Mean endpoint x error |
| --- | ---: | ---: | ---: |
| Historical baseline | 88.5330% | 20,049.0 | 2.6104 px |
| Independent | 93.8424% | 10,766.0 | 1.4660 px |
| Shared | 93.6562% | 11,091.5 | 1.4061 px |

Independent failed-window counts are 10,411/11,121, and shared counts are
10,515/11,668, versus 20,049 for the historical baseline. Sharing lowers mean
endpoint MAE slightly, from 1.4660 to 1.4061 pixels, while increasing the number
of imperfect paths. Independent maxima are 141/141 pixels; shared maxima are
142.5/140, versus the baseline's 142. These overlapping partial-state windows
are not independent games or a four-variable/full-state emulator test.

Keep the independent pair as the stronger accuracy baseline for subsequent
experiments; retain the shared pair as a parameter-saving tradeoff. Do not
replace the historical reference or change the vertical checkpoints. Near-paddle
horizontal errors remain the main obstacle before further integration. Four
portable models are in `runs/horizontal-pair-20260922`; plans, source, preparation
receipts, per-event metrics, fitting probes, all recursive error witnesses and
verification are in `logs/horizontal-pair-20260922`.

Checkpoint hashes and reloaded metrics match. Frozen dependency weights and
immutable snapshot inventory remain unchanged. Production tests cover shared
versus independent gradient paths, identical initialization, input immutability,
checkpoint reload and brute-force equivalence of recursive evaluation. The full
suite passes with 401 tests and two skips, including bounded direct and latent
train/checkpoint/play tests. Ruff and whitespace checks pass. The direct baseline,
shared runner and RGB player have no implementation changes.

## Four-variable ball-state merge

On 2026-09-22, combine the new-only horizontal-independent and vertical pair
into one `ball_motion` model. Use fixed first-seed parents
`runs/horizontal-pair-20260922/independent-s91.pt` and
`runs/upper-dataset-20260921/new-s91.pt`. Record both hashes and complete nested
specifications in `logs/ball-motion-20260922/plan.json`. This is a staged merge
of existing models, not a new shared-trunk architecture or from-scratch training.

The model predicts x, combined integer/fractional y, vx and vy atomically from
the same 118-field source. Inference has no native game rules. Before fitting,
verify bitwise prediction equivalence with both parents on every validation
source. The composed baseline has 67 x, 30 y, 63 vx and 26 vy errors. Horizontal
joint errors remain 103, vertical errors remain 34, and their union contains
129 erroneous transitions out of 205,075, or 99.9371% exact ball-state accuracy.
Store this portable composition as `runs/ball-motion-20260922/baseline.pt`.

All fitting and validation targets come from the pinned new dataset revision
used in the preceding horizontal experiment. Reconstruct controller, fractional
state and contact in RAM with the same native replay checks. Training has
1,640,422 sources and validation has 205,075. No final-test targets, persistent
feature caches or new dataset columns are used.

Train two continuations from the identical composition with seeds 91 and 2026.
Each uses 6,000 AdamW updates, batch size 192, learning rate 0.00001 decaying by
cosine to 0.000001, and weight decay 0.0001. Sample 32 transitions per current-
region/any-ball-velocity-change stratum. Unchanged/changed counts are 545,421 /
74,418 upper, 107,907 / 26,434 near paddle, and 869,096 / 17,146 flight. No
intermediate checkpoints are selected and no failure mining is performed.

One forward/loss step sums mean cross-entropies for x displacement, vx, upper y
displacement, upper raw vy, paddle timing and true-bounce paddle speed. The
horizontal independent branches, four vertical upper modules and vertical paddle
trunk/heads contain 689,916 trainable parameters. Vertical flight, correction,
geometry and intermediate-paddle dependencies remain frozen. Region-specific
losses receive only their corresponding training sources. Current non-ball fields
are still supplied; this stage has no learned termination integration.

Evaluate every eligible 1-, 8-, 32- and 128-step window with all four ball outputs
fed back, splitting combined y into RAM integer and fractional eighths. Other
state fields and reference life boundaries remain supplied. This tests cross-axis
feedback for the first time in the new-data merging sequence. Overlapping windows
are not independent games and these are not full-state autonomous rollouts.


| Model | Joint errors / 205,075 | Exact one-step ball state | Entirely exact128 / 174,841 | Mean x endpoint error | Mean y endpoint error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Composed parents | 129 | 99.9371% | 92.6608% | 1.8691 px | 7.2169 px |
| Joint continuation, seed 91 | 123 | 99.9400% | 93.3265% | 1.6682 px | 6.3050 px |
| Joint continuation, seed 2026 | 129 | 99.9371% | 93.0909% | 1.7570 px | 7.2310 px |

Select seed 91 for the next merge. It reduces joint errors from 129 to 123 and
failed 128-step windows from 12,832 to 11,668. Both coordinate mean endpoint
errors improve. Its individual errors are 71 x, 31 combined y, 50 vx and 30 vy;
therefore the joint improvement is not an improvement of every individual head.
The corresponding maxima remain 144 pixels in x and 870.75 pixels in y, so rare
large drift remains. Seed 2026 has 129 joint errors and 12,080 failed windows,
but y MAE increases slightly to 7.2310 pixels and its maximum increases to 884.25.
It fails the predeclared continuation gate and remains diagnostic.

Candidate selection requires no worse joint one-step errors, no fewer exact
128-step windows and no worse mean endpoint error in either coordinate than the
composed baseline. Ties use joint errors then seed. This is selection on reused
validation/development data, not a new reserved-test claim. The selected file is
`runs/ball-motion-20260922/candidate.pt`, copied exactly from `joint-s91.pt`,
SHA-256 `3bfd3a5cfc5806a18a3cdfbf7394966dc0f1692193e355a706eacc8255ffa7d1`.
The original global reference and the two parent files remain unchanged.

The current merge sequence now has one checkpoint and one prediction interface
for all ball-motion fields. Brick layout is the next planned addition; remaining
paddle, memory, count, width and termination fields still come from data in this
evaluation. Preserve the staged integration objective without further failure
mining or exhaustive tuning at this step.

Portable reload reproduces all three models' validation metrics. Every frozen
parameter and buffer equals the composed parent state; parent hashes and dataset
snapshot inventory remain unchanged. The full suite passes with 403 tests and
two skips, including direct and latent train/checkpoint/play smoke tests. Ruff,
script compilation and whitespace checks pass. New tests cover simultaneous
four-output decoding, gradients for intended modules, frozen dependencies,
checkpoint portability, and brute-force equivalence of cross-axis feedback.
All scripts, plans, metrics, recursive witnesses, selection and verification
receipts are in `logs/ball-motion-20260922`.

## Ball and brick-layout merge

On 2026-09-22, add brick layout to the selected ball-motion candidate in a single
registered `ball_bricks` model. Parent checkpoints are
`runs/ball-motion-20260922/candidate.pt`, SHA-256
`3bfd3a5cfc5806a18a3cdfbf7394966dc0f1692193e355a706eacc8255ffa7d1`, and
`runs/brick-layout-20260918/spatial-s2026/best.pt`, SHA-256
`ffc256bcacfcd7bb9cfd587895e71c1c6cf26e66eca86697e038d70c04ae6c09`.
The brick initialization is historical; subsequent fitting uses only the new
checkpoint-trajectory dataset. This is not from-scratch fitting.

The output is x, combined y, vx, vy and all 108 brick cells. Both branches read
the same current state and apply their outputs simultaneously. Before training,
verify prediction equality against both parents on all 205,075 validation sources.
The composition preserves 123 ball-state errors and adds 16 complete-layout
errors, with 136 jointly incorrect transitions. Cell error counts are not used
as a substitute for exact complete-layout accuracy.

The pinned new dataset has 34,184 removals among 1,640,422 fitting transitions and
4,251 among 205,075 validation transitions. Native replay and target audits find
no additions, multiple removals or wall clears. Retain the 109-class output
contract: no change or one occupied-cell removal. This does not add support for
wall refills. Targets, reconstructed hidden fields and all features stay in RAM;
no dataset columns or final-test targets are touched.

Use two fixed-seed continuations from the identical composition. Keep the prior
6,000-update AdamW schedule, batch size 192, learning rate 0.00001 decaying to
0.000001 and weight decay 0.0001. Each batch draws 32 examples per current-region/
any-ball-velocity-change stratum. Add brick cross-entropy to the six motion
objectives, averaging removal and no-change losses equally within that same
batch. The brick branch adds 56,130 trainable parameters, for 746,046 total.
Existing frozen dependencies stay frozen. There are no shared trainable weights
between the brick and motion branches, so joint fitting here does not establish
an effect from representation sharing. No hard mining or intermediate checkpoint
selection is performed.

Run actual joint ball/layout feedback at horizons 1, 8, 32 and 128. Predicted
layout becomes the next input layout alongside predicted ball state. Contact
memory, paddle fields, hit count and reference life boundaries remain supplied.
Windows overlap and these are still partial-state rollouts. Compare continuations
against the atomic composition, not against a ball-only metric with fewer outputs.


| Model | Ball errors / 205,075 | Complete-layout errors / 205,075 | Joint errors / 205,075 | Fully exact128 / 174,841 | Mean x endpoint error | Mean y endpoint error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Atomic composition, retained | 123 | 16 | 136 | 92.5407% | 1.7137 px | 6.6627 px |
| Continuation seed 91 | 121 | 46 | 164 | 91.3642% | 1.9903 px | 8.9587 px |
| Continuation seed 2026 | 122 | 50 | 169 | 91.1085% | 1.8814 px | 8.7854 px |

Retain the atomic composition for the next merge. It has **99.9337%** exact joint
ball/layout accuracy and **99.9922%** exact complete-layout accuracy. Both
continuations slightly improve ball error counts but worsen brick generalization
and coupled feedback. Ordinary-transition layout errors rise from 7 to 36/41.
The models have separate trainable branches; this does not demonstrate gradient
interference from sharing representations. Resampling and continuation are
possible causes, not isolated findings. Defer further tuning and mining until
later in the integration sequence.

The retained model has 13,042 imperfect 128-step windows, versus 15,099/15,546
for continuations. At the endpoint, 6,742 baseline layouts contain wrong cells,
compared with 9,423/9,215; individual wrong-cell totals are 19,027 versus
32,164/32,656. Mean ball error is x 1.7137 / y 6.6627 pixels, with maxima 144 and
870.75 pixels. Rare large drift remains. The prior ball-only rollout score of
93.3265% supplied true layouts and scored fewer outputs; the new 92.5407% score
feeds layouts back too. It is a stricter integration test, not a change to parent
one-step predictions.

The predeclared continuation gate requires no worse joint one-step errors,
exact128 count, either coordinate endpoint MAE, one-step layout errors or endpoint
layout errors than the composition. Neither continuation qualifies. Store both
as diagnostics. Save the exact composed baseline as
`runs/ball-bricks-20260922/candidate.pt`, SHA-256
`c12ec04501a1fe1881f8095bcfc6e8c10f5f5079cfb8f074d5061fafcbc7b9fc`.
The merge itself is complete, and brick-contact memory is the next planned
addition. Existing parent files and the historical reference remain unchanged.

All three portable checkpoints reproduce their saved validation metrics after
reload. Frozen parameters/buffers, parent hashes and raw dataset snapshot inventory
verify unchanged. The full suite passes with 405 tests and two skips, including
bounded direct and latent train/checkpoint/play checks. Ruff, compilation and
whitespace checks pass. New tests cover simultaneous ball/layout prediction,
occupied-cell removal constraints, empty layouts, gradient/freeze boundaries,
checkpoint reload and brute-force equivalence of coupled feedback across lives.
Plans, source, audits, per-event scores, recursive witnesses and verification
are in `logs/ball-bricks-20260922`.


## Ball, bricks and contact merge on migrated data

The next merge adds next brick-contact memory to the existing ball/layout model.
Pin dataset `tsilva/gradlab-breakout-6127e81d` at
`8f9838c532a2d6b622b4fc0bb5090fe6c534210f`, with migrated split
`22284897f9fe400efa60772c7e47fc77c870f08b3869cc6d8202b9bac6b24636`.
Gradlab commits `a9654af4` and `e275acb2` introduce the explicit schema-v1
contract and durable migration publication. The new view coexists with old
snapshot paths, so the adapter selects its split explicitly.

Audit all 22 training/validation Parquet hashes, declared schema identities and
physical Arrow schemas. A bounded decoded-value comparison with the previous
revision confirms unchanged values and episode assignments across 400 training
and 50 validation episodes: 1,647,968 and 206,044 transitions respectively.
The migration adds contract metadata without changing these learning examples.
Reserved test targets and frame asset shards remain unread. Derive state features in
RAM; no dataset columns or feature caches are written. After the existing
life/startup/quality filters, retain 1,640,422 training and 205,075 validation
transitions. Native replay still verifies all retained ball/layout transitions.

`ball_bricks_contact` reads the existing 118-value current-state vector and
emits 113 values: x, combined y, vx, vy, 108 bricks, then contact. Compose the
selected ball/layout checkpoint with `brick-contact-20260919/geometry-s2026/best.pt`
(contact SHA-256 `4f23f5fc4d2dea3159a15c813bd98e9303883079447a661a3cac692c5ee297b0`).
Its 86→128→128→2 ReLU head adds 27,906 trainable parameters for 773,952 total.
Contact retains its own frozen learned layout dependency. All branches read the
same original source. Full-validation parent predictions match exactly.

Both seeds (91, 2026) start at that composition: 6,000 AdamW updates, batch 192,
learning rate 1e-5 decaying to 1e-6, weight decay 1e-4. A shared sampled batch
contains 32 rows from each current-region × velocity-change training stratum.
Sum the six motion cross-entropies, contact cross-entropy and the brick loss
balanced between removal/no-change cases in that batch. Dependencies remain
frozen/eval. No hard mining or intermediate checkpoint selection.

| Model | Ball errors | Layout errors | Contact errors | Joint errors | Exact 128-step windows | x / y MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Atomic composition | 123 | 16 | 23 | 145 | 92.2644% | 1.7578 / 7.2773 px |
| Continuation seed 91 | 121 | 46 | 33 | 186 | 90.5983% | 1.9157 / 8.0000 px |
| Continuation seed 2026 | 122 | 50 | 29 | 187 | 90.4513% | 1.8523 / 8.1014 px |

Retain atomic composition as `runs/ball-bricks-contact-20260922/candidate.pt`. It has **99.9293%** exact joint accuracy and **99.9888%** contact accuracy. Neither fixed continuation passes the predeclared no-regression gate. The model merge is complete; further tuning stays deferred. The portable model retains specialized branches with separate hidden layers. For both seeds, all 354 ball/layout parameter and buffer tensors exactly match the prior ball/layout-only continuation. Adding the contact loss did not cause the existing brick regression.

There are 174,841 eligible overlapping 128-step windows; the selected
model has 13,525 imperfect windows. The evaluator
feeds back ball, bricks and contact simultaneously, and reports each error family
separately. Other state (paddle, width, charge and hit count) and reference life
boundaries remain supplied. This is a stricter partial-state test than the prior
ball/layout run, which supplied true contact. It is not autonomous full-state
emulation. The selected endpoint maxima are x 144.0000 and
y 877.5000 pixels; rare large drift remains.

The predeclared gate requires no worse joint one-step error, exact128 count,
x/y endpoint MAE, or one-step/endpoint layout/contact errors than composition.
Among qualifying runs choose fewer joint errors, then seed. This is development
selection, not a fresh final-test estimate. Parent checkpoints and the historical
reference stay unchanged. Selected SHA-256: `3f23502ae775743f600d59432554146a00e7ea5e142d16ffd4dd0759f1db8d7d`.

Portable reload metrics, frozen parameters/buffers, parent hashes and raw snapshot
inventory all verify unchanged. The full suite passes: 407 tests, two skips,
including bounded direct/latent train/checkpoint/play checks; Ruff and whitespace
checks pass. Focused tests cover contact gradients/frozen dependencies, reload,
atomic output and brute-force feedback equivalence across life boundaries.
Reproduction scripts, plans, audits, per-event errors and rollout witnesses are in
`logs/ball-bricks-contact-20260922/`. Next merge: prior paddle-hit count.


## Paddle-hit count added to the transition container

Keep the container approach and postpone a shared MLP until all state predictors
are integrated. Add the selected `paddle-hit-count-20260919/count-s2026/best.pt`
head to `ball-bricks-contact-20260922/candidate.pt`. The registered
`ball_bricks_contact_count` predicts 114 values from the same 118-value source:
four ball fields, 108 brick cells, contact and capped paddle-hit count.
Every branch reads the original source, and all outputs are applied together.

Unlike the previous joint continuations, freeze the entire established state
predictor for this stage. Train only the new 96→256→256→256→2 ReLU hit classifier,
156,930 parameters. Its geometry dependency also stays frozen/eval. The previous
two merge experiments already showed regressions from continuing the established
branches; repeating that training is unnecessary for testing count integration.
This is still one portable container checkpoint with separate learned branches.

Use the same migrated dataset revision `8f9838c532a2d6b622b4fc0bb5090fe6c534210f`
and explicit split `22284897f9fe400efa60772c7e47fc77c870f08b3869cc6d8202b9bac6b24636`.
All 22 train/validation shard SHA-256 values still match the publisher inventory.
Native replay and life/quality filters yield 1,640,422 training and 205,075
validation transitions. Features and labels remain in RAM; no dataset columns or
persistent feature caches are written. Reserved final-test targets remain unread.

The broad paddle region covers all 24,661 training hits and 3,121 validation hits.
Training has 134,341 paddle-region sources; validation has 17,074. Count increments
number 14,171 and 1,853; the remaining 10,490 and 1,268 hits occur at saturated
count 12. Train actual hit/no-hit cross-entropy, including saturated hits, then
decode `min(12, current_count + predicted_hit)`. Offline checks verify every count
label against native hits and its alignment with the next source within a life.

Before fitting, verify full-validation equivalence with both parent checkpoints.
Initial composition has 34 count errors, 47 raw hit errors and 170 joint errors.
The 47 hit errors are 37 false positives and 10 misses; count saturation hides
13 of them. Ball, layout and contact errors remain 123, 16 and 23.

Compare two fixed seeds, 91 and 2026, each initialized from the same composition.
Use 6,000 AdamW updates, batch 192, learning rate 1e-5 cosine-decayed to 1e-6 and
weight decay 1e-4. Each batch samples 96 actual hits and 96 non-hits from training
paddle-region sources. No hard-example mining or intermediate checkpoint selection.
Train only the count branch on these batches; the existing state branch remains
fixed. All validation targets stay out of fitting.

| Model | Count errors | Hit errors | Joint errors | Entire 128-step window exact | x / y endpoint MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial composition | 34 | 47 | 170 | 91.5089% | 1.8137 / 7.4769 px |
| Count tuning seed 91 | 28 | 38 | 163 | 91.6015% | 1.8158 / 7.4646 px |
| Count tuning seed 2026 | 29 | 39 | 163 | 91.6015% | 1.8158 / 7.4646 px |

Selected **initial composition** as `runs/ball-bricks-contact-count-20260922/candidate.pt`. Neither continuation meets the predeclared gate, so retain the initial composition. Both tuned runs improve one-step and exact-window accuracy but increase x endpoint MAE from 1.813656 to 1.815813 pixels, endpoint ball errors from 12,320 to 12,381, and layout errors from 6,609 to 6,611. This is a small trade-off, not a uniform regression.

The selected checkpoint reaches **99.9171%** exact combined accuracy,
**99.9834%** count accuracy and **99.9771%** raw hit accuracy. Its hit detector
has 37 false positives and 10 misses.
Hit precision is 98.8247% and recall is 99.6796%. Overall hit accuracy includes
many ordinary no-hit transitions; count accuracy also benefits from saturation.
All candidates preserve existing ball/layout/contact predictions exactly.

Rollouts now feed ball, bricks, contact and count back together. At horizon 128,
159,995 of 174,841 overlapping windows remain entirely
exact, **91.5089%**. Endpoint count errors number 4,908;
ball/layout/contact endpoint errors are 12,320, 6,609
and 697. Maximum x/y errors are 144.0000 and
877.5000 pixels, so rare large drift persists. Paddle x, width, charge
and reference life boundaries remain supplied. This is partial-state feedback;
the preceding contact merge supplied true hit count and reached 92.2644% exact128.

The predeclared selection gate requires no worse joint/count/hit one-step errors,
exact128 count, x/y endpoint MAE or endpoint ball/layout/contact/count errors than
composition. Among qualifying continuations choose fewer joint errors, then count
errors, then seed. Selection uses development data; final tests remain reserved.
Selected SHA-256: `c0df5706bf2268e90b94f9a979d89b0977a6d7bc726f03db22603a372dee3493`.

All checkpoints reload with identical validation metrics. Frozen parameters and
buffers, parent hashes and raw snapshot inventory remain unchanged. Full tests:
409 passed, two skipped, including direct and latent train/checkpoint/play smoke
checks. Ruff, compilation and whitespace checks pass. Added tests cover frozen
state branches, count saturation, checkpoint reload and brute-force equivalence
of coupled count feedback across life boundaries.

Artifacts and reproduction instructions are in
`logs/ball-bricks-contact-count-20260922/`. Next integration: paddle width.


## Paddle width added to the transition container

Add the historical `paddle-width-20260919/proposal-s2026/best.pt` predictor to
`ball-bricks-contact-count-20260922/candidate.pt`. The registered
`ball_paddle_width` model emits 115 values from the existing 118-value source:
four ball fields, 108 bricks, contact, capped hit count and paddle width. Every
branch reads the original source; all predictions are applied together. Shared
MLP consolidation remains deferred until the container predicts the whole state.

Keep the established state branch frozen/eval. Only the added width classifier
trains: 33→64→64→2 ReLU, 6,466 parameters. Inputs are combined ball y, vertical
velocity and current width, with scalar/binary encoding and a constant-velocity
proposal feature. Output classes are 12 and 16 pixels. The model executes no
native collision rule and uses no successor inputs, images or action history.

Use the same migrated dataset revision `8f9838c532a2d6b622b4fc0bb5090fe6c534210f`
and split `22284897f9fe400efa60772c7e47fc77c870f08b3869cc6d8202b9bac6b24636`.
All 22 train/validation shard hashes still match the publisher inventory.
Native audits and existing life/quality filters retain 1,640,422 training and
205,075 validation transitions. Features and labels remain in RAM; no dataset
columns or persistent feature caches are written. Final-test targets stay unread.

Recorded width labels align exactly with the following source inside each life.
Training contains 624,367 narrow-stay transitions, 1,015,072 wide-stay transitions
and 983 shrinks from 16 to 12. Validation contains 74,409 narrow-stay, 130,551
wide-stay and 115 shrinks. Neither split has an in-life widening. Reference life
boundaries remain explicit; reset behavior is outside this experiment.

The initial composition reproduces both parents on every validation source.
Width is exact on all 205,075 transitions, including every actual width change.
Other one-step errors remain ball 123, layout 16, contact 23 and count 34;
combined errors stay 170. Zero observed width errors is not a guarantee for
unseen or already-diverged rollout states.

Compare fixed seeds 91 and 2026, both starting at the same composition. Each
receives 6,000 AdamW updates, batch 192, LR 1e-5 cosine-decayed to 1e-6 and weight
decay 1e-4. Sample 64 training examples from each of the three observed width
transition strata per batch. This exposes the rare shrinks without mining
validation mistakes. Cross-entropy uses recorded next width. All existing state
parameters remain fixed. No hard mining or intermediate checkpoint selection.

| Model | Width errors | Width-change errors | Joint errors | Entire 128-step window exact | Endpoint width errors | x / y endpoint MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial composition | 0 | 0 / 115 | 170 | 91.5089% | 960 | 1.8137 / 7.4769 px |
| Width tuning seed 91 | 0 | 0 / 115 | 170 | 91.5089% | 713 | 1.8137 / 7.4769 px |
| Width tuning seed 2026 | 0 | 0 / 115 | 170 | 91.5089% | 713 | 1.8137 / 7.4769 px |

Selected **width tuning seed 91** as `runs/ball-paddle-width-20260922/candidate.pt`. Both continuations reduce endpoint width mismatches from 960 to 713 without changing the other measured errors. Both qualify; the predeclared tie-break chooses seed 91.

The selected checkpoint has **100.0000%** one-step width accuracy and
**99.9171%** exact combined accuracy. Rollouts feed predicted width into
source column 5 alongside ball, bricks, contact and count. Of 174,841
overlapping 128-step windows, 159,995 are completely exact,
**91.5089%**. Endpoint width mismatches number 713.
These mismatches can follow earlier ball divergence; they do not contradict the
teacher-forced width score. Paddle x, charge and reference life boundaries remain
supplied. The previous count merge supplied true width and reached 91.5089% exact128.

Endpoint ball/layout/contact/count errors are 12,320,
6,609, 697 and 4,908.
Maximum x/y errors are 144.0000/877.5000 pixels;
rare large drift remains. This is a partial-state integration test.

The predeclared gate requires no worse joint/width/change-subset one-step errors,
exact128 count, x/y endpoint MAE or endpoint ball/layout/contact/count/width errors
than composition, plus at least one strict improvement. If every measure matches
the composition, retain its original weights. Among qualifying continuations choose fewer joint errors, then width
errors, then seed. This is development selection. Selected SHA-256:
`81c780f2ad5cfc9a17bd763e088c9b7d513c01b57010676f3f7e012d7371e817`. Parent checkpoints and the historical reference stay fixed.

Portable reload metrics, frozen tensors, parent hashes and raw snapshot inventory
all verify unchanged. Full tests: 411 passed, two skipped, including direct and
latent train/checkpoint/play smoke checks. Ruff, compilation and whitespace
checks pass. New tests cover atomic width composition, frozen gradients/modes,
checkpoint reload, and brute-force equivalence of width feedback across lives.
Plans, reproduction scripts, per-event metrics and rollout witnesses are in
`logs/ball-paddle-width-20260922/`. Next integration: paddle charge with action input.


## Paddle charge added to the transition container

Add `paddle-charge-s2026-20260918/best.pt` to the selected
`ball-paddle-width-20260922/candidate.pt`. The registered `ball_paddle_charge`
container accepts the existing 118 state values plus the current provider action
at column 118. It emits 116 values: ball x, combined y, vx, vy, 108 bricks,
contact, capped hit count, width and charge. Each branch reads the original
current state; outputs are applied together. Shared-MLP consolidation remains
postponed until the state container is complete.

The charge adapter supplies current paddle x, charge and requested action to
its historical 25→128→128→11 SiLU classifier. Scalar and binary state encoding
plus action one-hot encoding produce the 25 features. Its classes are charge
changes −120, −60, −9, −5, −1, 0, 1, 5, 9, 60 and 120. Decode the winning class
and add it to current charge without clipping. Only its 21,259 parameters train;
the entire established state branch stays frozen/eval. Inference uses no native
controller rules or successor information.

Use the same migrated HF revision `8f9838c532a2d6b622b4fc0bb5090fe6c534210f`
and split `22284897f9fe400efa60772c7e47fc77c870f08b3869cc6d8202b9bac6b24636`.
All 22 train/validation shards match their publisher hashes. Existing life and
quality filters retain 1,640,422 training and 205,075 validation transitions.
Replay the audited controller from reset metadata and full action history in
RAM to recover current/next charge. Recorded paddle observations match replay;
next charge agrees with the next source on all 1,638,465 training and 204,829
validation contiguous pairs. No dataset columns or feature caches are written.
Final-test targets remain unread.

Requested and effective/provider actions agree on every retained transition.
Internal replay uses emulator codes 1–3, while this model receives provider
codes 0–2; the conversion is explicit. Startup/serve behavior remains outside
this active-play experiment. Training action counts are 542,265 / 565,643 /
532,514; validation counts are 69,731 / 69,269 / 66,075. Nine charge-delta classes
occur in training and six in validation, all covered by the original output
head. Initial composition reproduces both parent predictions on every validation
source. Charge has zero errors, including all 132,141 actual changes. Existing
one-step errors stay ball 123, layout 16, contact 23, count 34 and width zero.

Compare fixed seeds 91 and 2026. Both start from the same parent weights and
receive 6,000 AdamW updates, LR 1e-5 cosine-decayed to 1e-6, weight decay 1e-4,
and batch 192 with 64 randomly sampled training examples per action. Charge-delta
cross-entropy is the only training loss. No validation mining or intermediate
checkpoint selection is used. Rare charge changes remain rare under this
sampling; perfect observed validation accuracy does not establish full coverage.

| Model | Charge errors | Joint errors | Entire 128-step window exact | Endpoint charge errors | x / y endpoint MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial composition | 0 | 170 | 91.5089% | 0 | 1.8137 / 7.4769 px |
| Charge tuning seed 91 | 0 | 170 | 91.5089% | 0 | 1.8137 / 7.4769 px |
| Charge tuning seed 2026 | 0 | 170 | 91.5089% | 0 | 1.8137 / 7.4769 px |

Selected **initial composition** as `runs/ball-paddle-charge-20260922/candidate.pt`. Neither continuation supplies a strict improvement while passing every no-regression check. Charge is integrated into the new checkpoint with its original weights; all previously integrated predictors remain fixed.

Selected one-step charge accuracy is **100.0000%** and exact combined
accuracy **99.9171%**, with 170 incorrect transitions.
Joint rollouts feed predicted charge into source column 6 alongside ball,
bricks, contact, count and width. Each step still receives its recorded current
action; action is never overwritten by state feedback. Across 174,841
overlapping 128-step windows, 159,995 are entirely exact
(**91.5089%**). Endpoint charge errors: 0; charge
MAE: 0.000000; invalid charge predictions: 0.
Paddle x and reference life boundaries remain supplied. Thus these are partial
state rollouts under recorded actions, not closed-loop policy evaluations.

Endpoint ball/layout/contact/count/width errors are 12,320 /
6,609 / 697 / 4,908 /
713. Endpoint x/y MAE is 1.8137/7.4769 pixels;
maximum error is 144.0000/877.5000 pixels.

Selection requires no worse combined/charge/change-subset one-step errors,
invalid charge predictions, exact128 count, x/y/charge MAE or endpoint field
errors than initial composition, plus at least one strict improvement. Ties
retain the parent weights; qualifying continuations sort by joint errors, charge
errors, then seed. This is development selection, with SHA-256:
`987e4134ab8d69a4f5217b87ff9f292cf4b957d6090f9b19e6862d15c453cc97`. Parent checkpoints and the historical reference stay fixed.

Reproduce the fixed experiment from the repository root with
`PYTHONPATH=. uv run python logs/ball-paddle-charge-20260922/reproduce.py`.
This requires the recorded parent checkpoints, raw snapshot and audit helpers.
It repeats integrity checks, RAM preparation, baseline comparison, training,
rollouts and checkpoint selection. Tests and documentation generation are separate.

Full suite: 414 passed, two skipped, including direct and latent
train/checkpoint/play smoke checks. Ruff, compilation and whitespace checks pass.
Portable reload, frozen tensors, parent hashes and raw snapshot inventory verify
unchanged. New tests exercise the action adapter, reject invalid action codes,
check charge feedback against brute force across life boundaries, and confirm
that charge training preserves every established predictor. Reproduction scripts,
plans, per-action/event metrics and rollout witnesses are under
`logs/ball-paddle-charge-20260922/`. Next integration: paddle position, then
life-loss termination before autonomous full-state rollouts.


## Paddle position added to the transition container

Add `paddle-minimal-charge-s2026-20260918/best.pt` to the selected
`ball-paddle-charge-20260922/candidate.pt`. The registered `ball_paddle_position`
container keeps 119 inputs: 118 current-state values and current provider action.
It emits 117 values: ball x, combined y, vx, vy, 108 bricks, contact, capped hit
count, width, charge and paddle x. All branches read the original current state,
then their outputs are applied together. Shared-MLP consolidation remains deferred.

The added 25→128→128→23 SiLU classifier uses current paddle x, charge and action.
Scalar and binary encoding of x/charge plus one-hot action produce 25 features.
It classifies integer paddle displacement from −11 through 11, then adds the
winning class to current paddle x without clipping. Only its 22,807 parameters
train. Every established state branch, including charge, stays frozen/eval.
Neither predicted next charge nor native controller rules enter this predictor.

Use migrated HF revision `8f9838c532a2d6b622b4fc0bb5090fe6c534210f` and split
`22284897f9fe400efa60772c7e47fc77c870f08b3869cc6d8202b9bac6b24636`.
The 22 train/validation shards still match publisher hashes. Existing life/quality
filters retain 1,640,422 training and 205,075 validation transitions. Paddle x
labels come directly from recorded successors, converted to integer pixels.
They match following source positions on all 1,638,465 training and 204,829
validation contiguous pairs. Controller replay still audits hidden source state
and charge labels in RAM. Dataset columns and feature caches are unchanged;
final-test targets stay unread.

Both splits contain all 23 displacement classes. Requested and executed actions
agree on every retained active-play transition. The initial composition reproduces
both parent predictors on every validation source; existing errors stay ball 123,
layout 16, contact 23, count 34, width zero and charge zero.

Compare two fixed continuations, seeds 91 and 2026. Each starts from the same
composition and receives 6,000 AdamW updates, batch 192 with 64 randomly sampled
training rows per action, LR 1e-5 cosine-decayed to 1e-6, and weight decay 1e-4.
Paddle-displacement cross-entropy is the only fitted loss. No validation mining
or intermediate checkpoint selection is used.

| Model | Paddle x errors | Joint errors | Entire 128-step window exact | Endpoint paddle x / charge errors | x / y endpoint MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial composition | 79 | 249 | 88.3889% | 105 / 0 | 1.8137 / 7.4769 px |
| Position tuning seed 91 | 35 | 205 | 90.0675% | 45 / 0 | 1.8137 / 7.4769 px |
| Position tuning seed 2026 | 34 | 204 | 90.0675% | 43 / 0 | 1.8137 / 7.4769 px |

Selected **position tuning seed 2026** as `runs/ball-paddle-position-20260922/candidate.pt`. The continuation passes every predeclared no-regression check and supplies a strict improvement.

Selected paddle-position accuracy is **99.9834%**, with
34 / 205,075 errors, MAE 0.00016579 pixels and
maximum error 1.0000 pixels. Exact combined accuracy is
**99.9005%**, with 204 incorrect transitions.

Rollouts now feed every compact state field back together. Paddle x output 116
updates source column 4, while charge output 115 updates column 6. Current action
is taken from each recorded transition. Across 174,841 overlapping
128-step windows, 157,475 are entirely correct
(**90.0675%**). Endpoint paddle-position errors are
43, MAE 0.000292 pixels, maximum
4.0000 pixels. Endpoint charge errors are
0, charge MAE 0.000000, with
0 out-of-range charge predictions.
The prior charge container reached 91.5089% exact128 while receiving true paddle
position; this experiment removes that remaining state correction.

Endpoint ball/layout/contact/count/width errors are 12,320 /
6,609 / 697 / 4,908 /
713. Mean endpoint ball x/y errors are
1.8137/7.4769 pixels, maximum
144.0000/877.5000 pixels. These are recorded-action,
within-life rollouts. Reference life boundaries still stop windows; terminal
transitions and wall refill behavior are outside this experiment. This is not yet
an autonomous emulator or an RGB player integration.

The predeclared gate requires no worse joint/paddle-x/change-subset one-step
errors, exact128 count, x/y/paddle-x/charge endpoint MAE, invalid charge predictions
or endpoint ball/layout/contact/count/width/charge/paddle-x errors than initial
composition, plus a strict improvement. Ties retain composition; qualifying
continuations sort by joint errors, paddle-x errors, then seed. Selected SHA-256:
`0693e80263ccde9eaa2695ced7d913baf733991a75f9c5f90d8f9dc75cd9c77a`. Selection uses development validation, not reserved test.

Portable checkpoint reload metrics agree; frozen tensors, parent hashes and raw
snapshot inventory remain unchanged. Full suite: 417 passed, two skipped,
including direct and latent train/checkpoint/play smoke checks. Ruff, compilation
and whitespace checks pass. New tests verify current-state composition, frozen
training behavior, checkpoint round trips and coupled paddle position/charge
feedback against brute force across life boundaries.

Reproduce from the repository root with
`PYTHONPATH=. uv run python logs/ball-paddle-position-20260922/reproduce.py`.
This requires the recorded raw snapshot, parent checkpoints and audit helpers.
It runs integrity checks, RAM preparation, baseline equivalence, training,
rollouts and selection. Tests and document generation run separately. Metrics,
plans, per-action/event breakdowns and witnesses are in
`logs/ball-paddle-position-20260922/`. Next integration: life-loss termination.


## Life-loss termination added to the transition container

Add `life-termination-20260919/stop-s2026/best.pt` to
`ball-paddle-position-20260922/candidate.pt`. The registered
`ball_life_termination` model takes the existing 119 current-state/action values
and emits 117 next-state fields plus a stop flag at column 117. The stop head
runs first. Only continuing rows invoke the state predictor; stopped rows have
zero placeholder state fields that the consumer ignores. Passing the previous
boolean stop mask keeps those rows stopped and skips both networks.

The stop classifier is 31→64→64→2 ReLU, with 6,338 parameters. It uses current
integer/fractional ball y and vy, scalar/binary encoding and a constant-velocity
proposal feature. The complete established compact-state branch stays
frozen/eval. The head executes no native termination rule at inference.

The RAM preparation view now restores life-loss transitions previously excluded
from state fitting. Their stop labels come from the existing recorded-life-loss
contract; their successor state is entirely NaN-masked because it can contain a
respawn. Current terminal-source fractional y/contact come from the preceding
causal replay output within the same life, verified against the native source
stop-timing rule. All other state targets retain their previous alignment.
No dataset columns or persistent feature caches are written. Every life remains
available as a separate segment; no trajectory crosses death, a data gap or an
episode boundary.

Dataset revision remains `8f9838c532a2d6b622b4fc0bb5090fe6c534210f`, split
`22284897f9fe400efa60772c7e47fc77c870f08b3869cc6d8202b9bac6b24636`.
All 22 train/validation shard hashes match. Training now has 1,642,324 transitions:
1,640,422 continuing transitions and 1,902 deaths over 1,957 segments. Validation
has 205,314 transitions: 205,075 continuing transitions and 239 deaths over 246
segments. Seven validation segments are censored. Final-test targets stay unread.

The initial composition reproduces its continuing-state parent on all applicable
validation sources. It catches all 239 deaths with no false stops; existing state
errors remain 204. Compare seeds 91 and 2026 with 6,000 AdamW updates each, LR
1e-5 cosine-decayed to 1e-6, weight decay 1e-4 and batch 192. Sample 64 training
rows from each of death, nonterminal y≥190 and nonterminal y<190. Stratum sizes
are 1,902, 7,369 and 1,633,053. Fit only two-class stop cross-entropy; no terminal
successor-state loss, validation mining or intermediate checkpoint selection.

| Model | One-step false / missed stops | Exact 128-step-or-death windows | Full-segment exact death / early / missed | False stops in censored segments | Fully exact segments |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial composition | 0 / 0 | 89.7222% | 169 / 62 / 8 | 5 | 141 / 246 |
| Stop tuning seed 91 | 0 / 0 | 89.7222% | 169 / 62 / 8 | 5 | 141 / 246 |
| Stop tuning seed 2026 | 0 / 0 | 89.7222% | 169 / 62 / 8 | 5 | 141 / 246 |

Selected **stop tuning seed 91** as `runs/ball-life-termination-20260922/candidate.pt`. Both continuations reduce false stops in 128-step survival windows from 3,526 to 3,522, with other stop-timing and exact-window counts unchanged. Both qualify; the predeclared tie-break selects seed 91. Their longer survival exposes 150 more state-transition errors in bounded windows and four more in full-segment evaluation; those raw counts have different numbers of simulated transitions.

Selected stop classification is exact on all **205,314** validation sources:
239 / 239 deaths detected, 0
false stops and 0 missed deaths. Combined one-step state/stop
accuracy is **99.9006%**, with 204 errors.
State accuracy excludes terminal successor fields by design.

The dedicated evaluator feeds all state back under recorded actions and halts on
the first predicted stop. Bounded evaluation starts at every source, includes
shorter windows that end in death, and excludes shorter censored windows. At
128 steps or recorded death, 183,472 /
204,489 windows are entirely exact (**89.7222%**).
Among 29,648 death-ending windows, 28,510
stop on the exact step, 172 stop early and
966 miss the recorded death. There are
3,522 false stops in windows with no
reference death. This denominator differs from previous fixed-length nonterminal
windows; its percentage is not directly comparable to the prior 90.0675% score.

Full-segment evaluation seeds once per segment and never corrects predicted
state. Of 239 death-ending segments, **169
stop on the correct step**, 62 stop early and
8 miss the reference death. Among
7 censored segments,
5 stop falsely. Entire-state trajectories
are exact for **141 / 246 segments
(57.3171%)**. A missed death is scored as a miss,
then evaluation is censored at the recorded life end. Eventual late-stop timing
is unknown; no next-life actions or respawn states are substituted.

An offline audit of 217,487 source rows evaluated during selected full-segment inference finds **zero disagreements** between the neural stop flag and the native bottom-boundary timing rule applied to the same predicted inputs. Recorded-action full-segment timing failures therefore arise from earlier state-trajectory divergence in this evaluation. Native rules only audit the predictions; they never alter model outputs.

The predeclared selection gate permits no increase in teacher false positives or
missed deaths, or in rollout timing errors, early/false stops or misses, and no
decrease in exact-window counts at 1/8/32/128 steps or full-segment starts. It
requires a strict improvement somewhere; ties retain composition. Qualifying
candidates sort by teacher errors, full-segment timing errors, then seed.
Selected SHA-256: `05c5d195c0b280a083ac8c53d5544a9ffe889f199491ab28635c529c793d8916`. The parent checkpoints and historical
reference remain fixed. This is development validation, not a reserved-test score.

Full suite: 422 passed, two skipped, including direct and latent
train/checkpoint/play smoke checks. Ruff, compilation and whitespace checks pass.
New tests cover skipped inference and absorbing stop masks, frozen state training,
portable checkpoints, masked terminal targets, and independent brute-force
accounting for exact, early, missed and censored stops. Snapshot inventory and
frozen tensors remain unchanged.

Reproduce with `PYTHONPATH=. uv run python logs/ball-life-termination-20260922/reproduce.py`
from the repository root.
The recorded snapshot, parent checkpoints and audit helpers are required. Native
stop attribution is an additional read-only evaluation in `audit_stopping.py`.
Plans, checkpoints, per-window counts and audits are under the matching ignored
`logs/ball-life-termination-20260922/` and `runs/ball-life-termination-20260922/`
paths. RGB player/shared-runner behavior remains unchanged. All compact-state
branches and learned termination are now in one container; recursive accuracy
still needs work before shared-MLP consolidation or playable-emulator claims.


## First errors in full-state rollouts

Read-only diagnosis on 2026-09-22 of the selected life-termination container,
SHA-256 `05c5d195c0b280a083ac8c53d5544a9ffe889f199491ab28635c529c793d8916`.
Use the same migrated dataset revision and explicit train/validation split as the
termination experiment above. Recheck all 22 raw shard hashes. Reconstruct inputs
in RAM with the existing native/controller replay audits; do not write dataset
columns or persistent feature caches. No fitting, checkpoint selection or
final-test reads occur.

Replay all 246 validation segments with recorded actions. The baseline exactly
reproduces 141 entirely exact segments, 169 exact deaths, 62 early deaths, eight
missed deaths, five false stops in censored segments, 12,174 state-error steps and
165,026 simulated transitions. A missed death is still censored at the reference
boundary. Trace the first erroneous prediction in each of the 105 divergent
segments, verifying that its incoming state is exactly the recorded state: these
are initiating errors, not mistakes caused by earlier erroneous feedback.

| First erroneous transition involves | Segments |
| --- | ---: |
| Ball position or velocity | 67: 52 paddle region, 15 upper region |
| Paddle-hit count | 18 |
| Paddle position | 14 |
| Bricks or contact | 12 |
| Width, charge or stopping | 0 |

Six transitions involve both ball and another category; counts therefore overlap.
There are 61 ball-only, 38 other-only and six mixed first failures. Median first
error is transition 332 after initialization (range 18–2,731). Of the 105 first
errors, 44 occur on actual paddle collisions, 16 on brick collisions, four on
side-wall-only collisions and 41 on transitions with no recorded collision.
The source-defined paddle region includes near misses; it is broader than the
actual collision event.

### Correct the first error once to measure its consequences

These **oracle interventions use validation truth only for diagnosis**. At the
original first-error transition, replace only the erroneous fields in the named
category, once per segment. Continue learned feedback without further corrections.
Do not update weights or treat the resulting scores as deployable accuracy.

| Diagnostic intervention | Corrected transitions | Entire segments exact / 246 | Exact deaths / 239 | Early / missed deaths | False stops / 7 censored segments |
| --- | ---: | ---: | ---: | ---: | ---: |
| None | 0 | 141 | 169 | 62 / 8 | 5 |
| First ball error only | 67 | 179 | 201 | 34 / 4 | 2 |
| First non-ball error only | 44 | 159 | 172 | 59 / 8 | 5 |
| All erroneous fields at the first error | 105 | 199 | 205 | 30 / 4 | 2 |

The ball intervention restores 38 completely exact segments. Correcting all
fields restores 58. Independently verify the latter count against segments with
at most one erroneous teacher-forced transition. Raw state-error steps fall to
8,018/184,502 simulated transitions for ball correction and 5,441/187,883 for all
fields; the denominators change because corrected trajectories survive longer.
The experiment shows that initial errors cause substantial later drift, while
additional independent errors remain.

### Coverage, fitting and generalization

Search every nonterminal training row for exact matches to the current inputs
used by each affected head, respecting its source-defined routing. Integer hashes
only shortlist matches; verify equality of all selected input values. Compare
head labels, using displacement for positions and the native hit event for the
count classifier. No complete 119-value first-error state/action has an exact
training match. The relevant ball, brick/contact and hit-detector input tuples
also have no exact matches for these witnesses. This is not evidence of missing
state: high-dimensional combinations can be novel despite adequate local coverage.
No conflicting labels are found among the matches, but the absence of matches
means this test cannot exclude ambiguity for those other heads.

Paddle position has exact training matches for **13/14** first-error cases,
covering **417** matching rows. All labels agree with validation. This is direct
evidence that its residual errors include fitting failures on represented inputs.
Its key is current paddle x, charge and requested action.

Scan all **134,341 training paddle-region transitions** through the unchanged
container: 306 have some incorrect output, including 58 ball errors, 255 count
errors and 19 paddle-position errors (overlapping). Horizontal x/vx have 37/30
errors; vertical y/vy, bricks, contact, width, charge and stopping have zero in
this region. These are training measurements, not additional validation results.
Existing data therefore supplies a concrete error-mining pool. In contrast,
vertical paddle prediction already fits this training region exactly, so merely
replaying its existing mispredictions provides no examples to fix its validation
gap. Training has 24,661 actual paddle collisions among 1,640,422 nonterminal
transitions (~1.50%); collision rarity alone does not establish that previously
used training samplers underexposed them.

Audit the frozen intermediate-paddle geometry dependency against the pinned native
one-frame calculation. Each of the horizontal, vertical and count copies has 20
errors on the training paddle region and six on its 17,074 validation transitions,
but **none coincide with a first-error witness**. It does not explain these
initiating failures. Among 69 first-error witnesses located in the paddle region,
the vertical head has six incorrect bounce-timing classes and three incorrect
outgoing speeds on actual bounces. The comparator never replaces learned outputs.

Next experiment: use training-only error mining to tune the horizontal ball and
hit-count heads, retaining ordinary and near-collision examples to guard against
regression. Paddle position also has a small, directly evidenced fitting problem.
For vertical timing/speed, test generalization with training-derived near-boundary
sampling rather than expecting a misprediction-only pool to help. Keep the same
one-step, per-head and full-segment validation gates; preserve the current
checkpoint if no candidate improves without unacceptable regressions. The audit
does not justify adding state variables or changing the dataset yet.

Artifacts and diagnostic scripts are in ignored `logs/state-first-errors-20260922/`.
`reproduce_first_error.py` replays the pinned 105 one-transition witnesses in about
2.5 seconds and intentionally exits with an assertion failure while they remain
wrong; two independent runs reproduced all 105. `reproduce.py` rebuilds the full
RAM-only diagnosis using the recorded snapshot and existing native audit helpers.
Production code, checkpoint weights and shared train/play infrastructure are
unchanged. Ruff, diagnostic compilation and whitespace checks pass; the production
suite was not rerun for this documentation/diagnostic-only change.


## Unified residual MLP benchmark

Train a separate shared model from scratch on the same pinned train/validation
split. Use every training transition once per epoch in shuffled batches, with no
error mining, event oversampling, dataset modifications or final-test reads.
The established container checkpoint stays unchanged.

The model has **3,361,486 parameters**. Current 119-value inputs
become 188 scalar/binary features, then a 512-wide projection, six residual blocks
with two 512-wide linear layers each, final LayerNorm and a single 206-logit
output layer. Eleven classification objectives share every hidden layer.
There are no pretrained helpers, region routers or distillation targets.
Training-derived vocabularies contain 22 dx, 19 dy, eight vx, eight vy, nine
charge-change and 23 paddle-displacement classes; fixed event heads cover bricks,
contact, count increment, width and stopping. Validation labels all lie within
this training-derived output support.

Use seed 2026, 20 epochs, batch 1,024, AdamW with weight decay 1e-4, cosine LR
3e-4 to 1e-5 and gradient norm clipping at 1. The eleven task mean cross-entropies
have equal weight. State losses mask terminal rows, including their NaN targets;
stopping learns from all rows. Count supervision is the capped count increment,
not the original count branch's uncapped hit-event auxiliary target. Choose the
lowest CPU validation joint error count at epoch end; earliest epoch breaks ties.

Training processes **32,846,480 row presentations** in
**21.79 minutes** on Apple MPS, including epoch-end
CPU validation. The selected checkpoint is epoch **20**,
`runs/unified-state-mlp-20260922/best.pt`, SHA-256
`74239a438214c7f07f4f83680e9238a3030162cecd5fb3963030e0c0af56f0da`. All 1,642,324 training transitions are reused
exactly 20 times; 205,314 validation transitions only measure/select checkpoints.
The final-test partition remains unread.

| Measure | Existing container | Shared MLP |
| --- | ---: | ---: |
| Exact joint one-step accuracy | 99.9006% | 98.1852% |
| Incorrect transitions | 204 | 3,726 |
| False / missed one-step stops | 0 / 0 | 7 / 8 |
| Entire life/data segment exact | 141/246 | 13/246 |
| Exact death timing | 169/239 | 16/239 |
| Early / missed deaths | 62 / 8 | 186 / 37 |
| False stops among seven censored segments | 5 | 6 |

The per-field table evaluates decoded state heads on every one of the 205,075
nonterminal sources, even if the stop head would prematurely halt. Joint accuracy
uses actual stopping behavior. Brick accuracy requires the complete layout to
match; y includes its fraction.

| State field | Container accuracy | Shared MLP accuracy | Shared MLP errors |
| --- | ---: | ---: | ---: |
| ball_x | 99.9654% | 99.7006% | 614 |
| ball_y_with_fraction | 99.9849% | 98.8648% | 2328 |
| ball_vx | 99.9756% | 99.7703% | 471 |
| ball_vy | 99.9854% | 99.0711% | 1905 |
| bricks | 99.9922% | 99.2520% | 1534 |
| contact | 99.9888% | 99.2227% | 1594 |
| hit_count | 99.9834% | 99.9308% | 142 |
| paddle_width | 100.0000% | 100.0000% | 0 |
| charge | 100.0000% | 99.9980% | 4 |
| paddle_x | 99.9834% | 99.8318% | 345 |

Full-segment evaluation initializes once, feeds back every predicted field and
stops on predicted life loss. Actions remain recorded. Missed deaths are censored
at the original life boundary; no next-life actions or respawn targets are used.
The unchanged comparator reproduces its previous 204 one-step errors, 141 exact
segments and 169 exact death timings on the same freshly reconstructed inputs.
CPU checkpoint reload reproduces the selected epoch's validation results exactly.

This is one shared-network baseline with one seed and a fixed training budget.
The container's earlier predictors used specialized encodings, intermediate
supervision, samplers and training schedules. Equal raw input information and
validation targets therefore do not make this an architecture-only ablation.
A worse result does not establish that shared MLPs cannot reach the container's
accuracy. A fixed uniform 65,536-row training probe is diagnostic only; its full
per-field results are recorded alongside validation. The probe has two errors
on 65,536 training rows, or 99.9969% exact joint accuracy, versus 98.1852%
validation accuracy. This is evidence of a large generalization gap in this run;
it does not establish that insufficient capacity or more training epochs alone
explain the shortfall.

The separate checkpoint does not replace the container or change the RGB runner
or player. Artifacts, plan, history, source audits and reproduction scripts are
under `logs/unified-state-mlp-20260922/`, with weights under the matching `runs/`
directory. Reproduce from the repo root with
`PYTHONPATH=. uv run python logs/unified-state-mlp-20260922/reproduce.py`.
The local pinned snapshot and earlier native audit helpers are required.

Validation: 426 tests passed, two skipped. The full suite includes the existing
direct/multistage train/play smoke tests; new tests cover shared-layer gradients,
small-batch fitting, discrete target decoding, terminal masking, unsupported
labels, absorbing stops and portable checkpoint loading. Ruff, compilation and
whitespace checks pass. The snapshot inventory and original container hash remain
unchanged.

## Frameskip-1 split preparation

Before retraining the unified MLP, freeze a new grouped 80/10/10 split for
`tsilva/gradlab-breakout-c6d579da` at revision
`b2069e01a5a5a3db84f230caf7416bff86494132`. Reusing the frameskip-2 seed assignment
does not establish balance for these new trajectories.

The local split uses the previous split's documented method: search 300,000
seed-group allocations, then improve with cross-split swaps. Keep complete
recorded episodes and all checkpoint recordings of an environment seed together.
The objective emphasizes normalized brick progress, includes return and episode
length means and second moments, and matches pooled brick-progress quantile bins.
Each holdout receives its rounded share of successful episodes. This uses only
episode summaries, with no model-performance selection.

| Split | Episodes | Raw transitions | Mean normalized bricks destroyed | Mean return | Successful episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 800 | 7,665,223 | 0.43738426 | 100.4795 | 257 |
| Validation | 100 | 946,274 | 0.43754630 | 100.4890 | 32 |
| Test | 100 | 962,313 | 0.43782407 | 100.5520 | 32 |

Every one of the 20 checkpoints contributes 40/5/5 episodes. Environment and
policy seeds are disjoint across splits. All previous balance thresholds pass:
mean deviations below 0.1 pooled standard deviations, relative standard-deviation
differences below 10%, and brick-bin proportion differences below four percentage
points. Normalized brick progress retains the publisher's denominator of 216.
These figures describe complete episodes before life-loss/quality filtering.

The frozen consumer manifest is `logs/unified-state-fs1-20260922/split.json`;
`prepare.py` selects train/validation episode IDs from it before decoding
transitions. Episode Parquet views are under the adjacent `balanced-split/`
directory. Reproduce with `uv run python
logs/unified-state-fs1-20260922/balance_split.py`. The download script preserves
this assignment. The preparation step leaves source Hub files unchanged and does not train a
model. The subsequent user-requested publication adds these split views to the
same dataset; see the publication receipt below. Keep test membership frozen and
exclude test transitions from fitting and model selection. This split measures
held-out seeds within one training run, with checkpoint policies shared across
splits.

Publication receipt: the balanced split views were uploaded and verified at
[5f6e0ca8c28e2fc27aeda3ead04851a1f8a45a77](https://huggingface.co/datasets/tsilva/gradlab-breakout-c6d579da/commit/5f6e0ca8c28e2fc27aeda3ead04851a1f8a45a77).
All 46 published file hashes match the local artifacts. The 9,573,810 transition
rows were checked against their original source values before upload. The existing
`all`, frames, and sessions views remain available. Load the `transitions` or
`episodes` configuration with `split="train"`, `"validation"`, or `"test"`,
pinning this revision. `trajectory-view.json` now links the split manifest.

## Unified MLP on frameskip 1

Retrain the shared 188-input, 512-wide, six-residual-block MLP from scratch on the
published balanced frameskip-1 dataset. Pin dataset revision
`5f6e0ca8c28e2fc27aeda3ead04851a1f8a45a77` and its frozen split manifest. Raw source
tables filtered by that manifest contain the same values as the published split
views, verified during publication. Test transitions are excluded from model
preparation, fitting, and evaluation.

Keep seed 2026, AdamW, batch size 1,024, equal task losses, uniform shuffling,
weight decay 1e-4, gradient clipping at 1, and cosine learning rate 3e-4 to 1e-5.
Match the previous run's 32,080 updates and 20 validation checkpoints. The larger
dataset receives fewer complete epochs. Derive numerical output vocabularies
from training only, giving 3,347,635 parameters. The fixed state encoding and
hidden architecture are unchanged.

The one-frame native audit exactly matches 8,584,908 continuing train/validation
transitions. Controller replay matches all 8,611,497 raw train/validation rows.
After quality and life-loss filtering, retain 7,644,744 training and 943,949
validation transitions. No dataset columns or feature caches are written.
There are 327 retained transitions where charge differs from the usual saturated
repeat increment; these startup cases remain in training/evaluation.

Training took 26.47 minutes including validation, with 32,848,160 row presentations, or 4.297 epochs. Select update 32,080 by minimum validation joint errors.

| Measure | Previous frameskip 2 | New frameskip 1 |
| --- | ---: | ---: |
| Exact joint one-step accuracy | 98.1852% | 99.6056% |
| Joint errors / transitions | 3,726 / 205,314 | 3,723 / 943,949 |
| False / missed life-loss stops | 7 / 8 | 0 / 0 |

One-step durations differ. For feedback rollouts, sample 4,096 validation sources
without replacement in each dataset with seed 9127, feed every predicted state
variable back, and retain recorded actions. Score the entire window as exact
only if every state and stop prediction matches. Stop at predicted death; censor
missed deaths at the recorded life boundary. Exclude short censored windows at
each horizon. These overlapping windows are diagnostics, not independent trials.

| Native frames | Previous exact windows | New exact windows |
| --- | ---: | ---: |
| 2 | 4008/4096 (97.85%) | 4065/4096 (99.24%) |
| 16 | 3580/4094 (87.45%) | 3869/4096 (94.46%) |
| 64 | 2415/4092 (59.02%) | 3309/4088 (80.94%) |
| 256 | 832/4082 (20.38%) | 2110/4071 (51.83%) |

| Predicted field | New validation errors | Accuracy |
| --- | ---: | ---: |
| ball_x | 312 / 943,528 | 99.96693% |
| ball_y_with_fraction | 2,850 / 943,528 | 99.69794% |
| ball_vx | 252 / 943,528 | 99.97329% |
| ball_vy | 2,829 / 943,528 | 99.70017% |
| bricks | 1,346 / 943,528 | 99.85734% |
| contact | 2,705 / 943,528 | 99.71331% |
| hit_count | 53 / 943,528 | 99.99438% |
| paddle_width | 0 / 943,528 | 100.00000% |
| charge | 18 / 943,528 | 99.99809% |
| paddle_x | 44 / 943,528 | 99.99534% |

The fixed 65,536-row uniform training probe has 170 joint errors, or 99.7406% accuracy. It is diagnostic only and is never used for checkpoint selection.

The recordings, checkpoint policies, dataset sizes, and balanced seed assignments
differ, so this is not an isolated causal test of frameskip. Improvements apply
to these validation datasets and this recipe. Neither the original container nor
the previous unified checkpoint is replaced.

Checkpoint: `runs/unified-state-fs1-20260922/best.pt`. SHA256: `b188d8115b2e0fa219fd751a8eb1065226d163f3f05a155b156525ad7a25db3f`.

Plan, audits, history, metrics, and scripts are under
`logs/unified-state-fs1-20260922/`. To reproduce, execute `prepare.py`,
`train_fs1.py`, and `evaluate_fs1.py` in the same Python process with repository
imports enabled; run `compare_fs2.py` separately before evaluation. Preparation
keeps features in RAM and requires the pinned raw snapshots and earlier native
audit helpers. Reloading the selected checkpoint exactly reproduces validation.
The previous model also reproduces its 3,726 validation errors.

Published the frameskip-1 unified model as [tsilva/gymemu-breakout-unified-dynamics-fs1](https://huggingface.co/tsilva/gymemu-breakout-unified-dynamics-fs1), pinned at [0de847b1a306b50669debded82794fd491c6bb1a](https://huggingface.co/tsilva/gymemu-breakout-unified-dynamics-fs1/commit/0de847b1a306b50669debded82794fd491c6bb1a). The repository includes standalone PyTorch weights/code, the unchanged original checkpoint, input/output contract, model card, training history, evaluation metrics, and artifact hashes. All 19 downloaded file hashes match; the downloaded verification script and exact model-card quick start pass. This publication does not add test evaluation or change model weights.

## Recorded-state decoder

The 2026-09-22 experiment trains `state_renderer` on recorded visual states and
matching RGB frames from `tsilva/gradlab-breakout-c6d579da`, revision
`5f6e0ca8c28e2fc27aeda3ead04851a1f8a45a77`. It uses the existing episode/seed-grouped
train/validation membership; test remains reserved. No dataset columns change.
Select 32,768 train and 2,048 validation frames uniformly without replacement
(seed 20260922), from successors with ball y > 0, trustworthy brick grids and no
initial-wall flag. Join successor labels to `successor_frame_id` by ID, never
array offset. Training and validation selected frame IDs do not overlap.

The visual state is ball x/integer RAM y, paddle x/width and 108 brick bits.
Fixed spatial features feed a 10,249-parameter 20→64→64→64→9 SiLU pixel MLP.
Training uses palette cross-entropy with four equally weighted regions:
128 uniform playfield, 64 ball-neighborhood, 32 paddle-neighborhood and 32
brick-wall pixels per frame. Each update samples 32 training frames. The extra
object weighting prevents background pixels from dominating the objective.
Run 12,000 AdamW updates, seed 2026, learning rate 0.003 decaying to 0.00003,
weight decay 0.00001 and gradient clipping 5. Select the checkpoint with lowest
float32 RGB MSE over every playfield pixel in all fixed validation frames,
evaluating every 1,000 updates. HUD rows 0–16 are excluded; full RGB MSE is saved
separately and includes the deliberately black HUD.

Scripts, sample identity, provenance, history and reports live under
`logs/state-decoder-fs1-20260922/`. Run `prepare.py`, `train.py` and `report.py` in
that order in one Python process with repository imports enabled. Preparation
uses the local pinned dataset snapshots and keeps sampled states/images in RAM.
Checkpoint: `runs/state-decoder-fs1-20260922/best.pt`. This is a structured neural
renderer benchmark, not evidence about an unconstrained autoencoder. It has not
yet been evaluated on predicted-state rollouts or integrated into the player.

The selected update 12,000 checkpoint reached **99.999377% exact RGB pixels**,
**1,969/2,048 (96.1426%) pixel-perfect playfields**, and masked float32 RGB MSE
**1.64581e-6**, after 523 seconds including validation. There are 394 wrong pixels
across 79 frames (maximum 71 in one frame). The ball neighborhood has zero RGB
error across the sample; an independent isolated-sprite detector confirms exact
ball placement in all 1,769 scorable frames. The other 279 frames are not scored
by that detector because their ground-truth ball is not uniquely isolated.
Full RGB MSE is 0.00301980, dominated by the omitted HUD, so masked scores must
not be compared directly with existing full next-frame RGB baselines.

Visual inspection of the four worst reconstructions shows artifacts along the
lower edge of the top wall. Checkpoint reload reproduces all validation metrics;
CPU and MPS renders match on a 16-frame probe. The original dynamics checkpoint
hash is unchanged. Validation selected the checkpoint; this is not a test result
or a guarantee about unseen/predicted states. The reconstruction comparison and
`summary.json` are in the experiment log directory.

Published the unchanged decoder to
[tsilva/gymemu-breakout-state-decoder-fs1](https://huggingface.co/tsilva/gymemu-breakout-state-decoder-fs1),
revision `1aee45f2e741d499985cb666017ae0c15a345e83`. The 22 published artifact hashes
match downloaded files. Standalone verification and the exact model-card quick
start pass from the Hub cache. The package includes the original checkpoint,
inference weights/code, input contract, provenance, measurements and comparison
image. Publication adds no training or test evaluation.

## Longer training for the frameskip-1 unified MLP

Continue the published model with another 32,080 uniformly sampled updates,
keeping the architecture, numerical class vocabularies, split, batch size, losses,
weight decay, and gradient clipping unchanged. Use a constant learning rate of
1e-5, the original run's final rate. AdamW moments restart because the published
checkpoint did not save optimizer state. Recreate the original sampler position
from its seed and update count, continuing the partially consumed fifth epoch.
This is a low-rate weight continuation, not an exact optimizer resume.

Reproduce the parent's full validation metrics, fixed training probe, and all
sampled rollout results before fitting. Choose the minimum joint validation
error count over the parent and 20 continuation checkpoints, earliest tie.
Rollouts use the same 4,096 roots and denominators and do not select checkpoints.

The additional run takes 38.99 minutes of elapsed run time including validation, a memory optimization pause, and replayed updates, and brings total exposure to 8.594 epochs. Select added update 32,080, total update 64,160. All 32,080 planned updates complete.

To reduce memory pressure, pack the binary brick inputs in RAM and verify exact
round-trip equality for every affected value. Resume from the last saved optimizer
and sampler checkpoint, discarding and replaying uncheckpointed updates. The
retained update count, sampling, features, and objective remain unchanged.

| Measure | Published parent | Continued model |
| --- | ---: | ---: |
| Exact joint one-step validation | 99.60559% | 99.65846% |
| Joint errors / 943,949 | 3,723 | 3,224 |
| Fixed training-probe accuracy | 99.74060% | 99.87946% |
| False / missed life-loss stops | 0 / 0 | 0 / 0 |
| Entire 2-native-frame window exact | 4065/4096 (99.24%) | 4075/4096 (99.49%) |
| Entire 16-native-frame window exact | 3869/4096 (94.46%) | 3908/4096 (95.41%) |
| Entire 64-native-frame window exact | 3309/4088 (80.94%) | 3422/4088 (83.71%) |
| Entire 256-native-frame window exact | 2110/4071 (51.83%) | 2259/4071 (55.49%) |

| State field | Parent errors | Continuation errors |
| --- | ---: | ---: |
| ball_x | 312 | 241 |
| ball_y_with_fraction | 2,850 | 2,440 |
| ball_vx | 252 | 204 |
| ball_vy | 2,829 | 2,439 |
| bricks | 1,346 | 1,185 |
| contact | 2,705 | 2,306 |
| hit_count | 53 | 53 |
| paddle_width | 0 | 2 |
| charge | 18 | 20 |
| paddle_x | 44 | 116 |

State-field denominators are 943,528 nonterminal transitions. Rollouts feed back
all predicted state while retaining recorded actions, stop on predicted death,
and censor missed deaths at reference boundaries. Test remains reserved; the
dataset and published parent weights remain unchanged.

Selected checkpoint: `runs/unified-state-fs1-continue-20260922/best.pt`. SHA256: `411c4624695578aa1a72b304fb0f530f178df202d7c0cf85a28080db6af50023`.

Artifacts and scripts are under `logs/unified-state-fs1-continue-20260922/`.
Run `prepare.py`, `train_continue.py`, then `evaluate_continue.py` in the same
Python process with repository imports enabled. This requires the pinned raw
snapshot and the original run's audit/evaluation helpers. Features remain in RAM.
`resume-last.pt` saves optimizer moments and sampler state for future exact
continuation. Verify the selected CPU reload, 256 CPU/MPS predictions, original
checkpoint hash, and the saved sampler/optimizer round trip.

Longer training reduces joint validation errors by 13.4% and increases exact
256-frame windows by 3.66 percentage points on the fixed validation sample.
The training probe improves from 170 to 79 errors, a larger relative reduction
than validation (3,723 to 3,224), so generalization remains a concern. Vertical
ball motion and contact still dominate errors; paddle-position errors increase
from 44 to 116 despite the overall gain. The final checkpoint is best, so this
run does not establish that further training has reached a plateau.

## Interactive state dynamics and decoder

Run `uv run gymemu play-state` to combine the published frameskip-1 unified
state dynamics and state decoder in the browser player. It opens paused in
teacher-forcing mode using the first episode in the pinned validation split.
The mode selector and direct `--autoregressive` launch both start from the
selected episode's first eligible recorded state. Direct autoregressive startup
loads state metadata without fetching recorded RGB assets. `--start-source`
explicitly selects a custom complete state; there is no synthetic fallback. Tab toggles continuous play and R resets and pauses.

In teacher forcing, Space advances one recorded transition. Every prediction
receives the recorded source state and executed action, including controller
charge, fractional y, hit count, and contact memory. The loader reconstructs these
fields from preceding observations using the preparation contract from training.
It verifies controller positions against recordings. No predicted states feed
back, even after a false terminal prediction. Inactive or unreliable rows are
skipped; the timeline indexes eligible transitions. Use Playback settings to see
the original recorded step, exact next-state agreement, and predicted/recorded
life-loss flags.

Original displays the recorded successor image, joined by frame ID. Prediction
shows the decoded predicted state. Diff and RGB MSE compare full float32 RGB,
including the decoder's black HUD. Terminal targets do not score state agreement.
If either side is terminal, RGB MSE is unavailable; predicted terminal placeholders
are never decoded. Replay continues to the end of the selected recorded episode.
This is inspection, not a new aggregate accuracy evaluation.

```bash
uv run gymemu play-state --episode-id 6
uv run gymemu play-state --headless-steps 32 --output logs/state-replay.png
uv run gymemu play-state --autoregressive
uv run gymemu play-state --start-source path/to/source.json
uv run gymemu play-state --headless-actions 1,1,2,0 --output logs/state-play.png
```

`--episode-id` selects another episode in `--split validation`, the default.
`--split train` and `--split test` require explicit selection; no test data is read
by default. One episode is loaded per launch. `--dataset PATH` accepts a complete
local snapshot containing the configured split and original trajectory assets.
The pinned model/dataset revisions, split identity, key bindings, and 60-step/s
target cadence are in `configs/state_playback.yaml`. The synthetic Hub usage
example is not a playable default state.
Model downloads include only JSON and weights; inference uses local registered
implementations and never imports Hub Python files.

In autoregressive mode, each fresh Left, Right, or Space key press executes one
native-frame prediction. Every predicted field feeds the next dynamics input;
decoded pixels are display output only. No recorded frames correct the rollout.
Predicted life loss stops playback and preserves the last valid image. These
models do not implement respawn. Actual continuous speed depends on inference
and browser transport time.

Launching with `--autoregressive` uses the published synthetic full-wall example.
A custom start file contains `{"source": [[...119 values...]]}` in the published
dynamics input order and physical units. Its last value is an action placeholder
replaced by each key press. A screenshot alone cannot provide hidden state.
RGB history editing is disabled for both state modes. The HUD remains black,
and errors can accumulate during autoregression.

## Recorded initialization and controller fine-tuning

Direct `gymemu play-state --autoregressive` now starts from the selected episode's first eligible recorded state. It uses the same reconstruction as teacher forcing, including controller charge and hidden ball context, and does not fetch RGB assets. Explicit `--start-source` remains available. Missing data raises an error rather than falling back to the synthetic Hub usage example. The mode selector and reset preserve this starting state. Default teacher-forcing behavior is unchanged.

The startup regression test fails on the previous implementation, which tries to open the synthetic example. It passes after the fix. An additional test proves state loading works with image assets and publication metadata absent. Real CPU player smoke, native Codex in-app browser stepping/reset, both direct and multi-stage CLI train/play tests, 449 passing tests, two skipped tests, Ruff, and whitespace checks pass.

### Training and selection

Start from the longer-trained local checkpoint, not from the older public weights. Architecture remains the shared 188-input, width-512, six-residual-block MLP. No native rules are inserted into inference.

Generate 262,144 controller variations in RAM from training-only roots and unrelated training-only ball/brick contexts. Replay verified controller rules to label only charge and paddle position. Roots outside charge 1500..2500 have already passed the startup repeat acceleration; hold actions in 16-step blocks. The synthetic labels use repeat=60, and original recorded examples retain their actual startup labels. This is causal augmentation for controller outputs, not a claim that all mixed full-game states are jointly reachable. No augmented ball or brick targets are invented. No dataset files or columns change, and no test transitions are decoded.

First try 4,000 output-head-only updates with a frozen trunk. All four checkpoints and three smaller weight blends regress full validation, so discard that approach. This does not prove frozen features can never work; it rejects this bounded recipe.

Then restart from the parent and fine-tune the unchanged shared network for 4,000 updates at learning rate 3e-6. Each batch contains 768 uniformly sampled original training rows, 128 uniform controller variations, and 128 mined controller failures. Minimize the mean of the original eleven objectives plus 0.2 times the mean of the two controller objectives on variations. Refresh training failures every 1,000 updates. Mined errors decrease from 349 to 12. All normal and augmented fitting examples are training-derived. Native-label oracle use is confined to preparation/training/evaluation.

The original selection rule picks update 2,000 by full one-step validation errors. It has 3,111 joint errors but slightly worse 256-frame rollout exactness, 55.32%. Evaluate the final update separately because it has the better controller-field score. Retain update 4,000 as a separate local candidate: 3,120 joint errors, better controller stress performance and slightly better 256-frame rollouts, with regressions at shorter horizons. This selection extension is adaptive validation tuning, not a fresh test result. Preserve `best.pt` as the 2,000-update one-step selection and `candidate.pt` as the final controller candidate.

### Results

All standard one-step comparisons use the same 943,949 validation transitions, with 943,528 nonterminal field targets. Rollouts use the same 4,096 roots and horizon-specific denominators, predicted-state feedback, and recorded actions. Exact windows require every state and termination prediction to match. Overlapping windows are not independent trials; the small 256-frame gain is not a statistical significance claim.

| Measure | Published model | Longer-trained parent | Controller candidate |
| --- | ---: | ---: | ---: |
| Exact next-state accuracy | 99.60559% | 99.65846% | 99.66947% |
| Paddle-position errors | 44 | 116 | 84 |
| Charge errors | 18 | 20 | 18 |
| Exact 16-frame windows | 94.46% | 95.41% | 95.21% |
| Exact 64-frame windows | 80.94% | 83.71% | 83.41% |
| Exact 256-frame windows | 51.83% | 55.49% | 55.69% |
| Controller stress joint errors / 65,536 | 52 | 64 | 16 |
| Controller stress illegal charge outputs | 32 | 32 | 7 |
| Controller stress illegal position outputs | 6 | 8 | 1 |

The stress set is generated after checkpoint selection using validation-derived roots/contexts and a separate fixed seed. It tests varied controller/game contexts under teacher forcing; it is distinct from the standard dataset and from long free-running play. The one known published charge-overflow fixture is correct in both the longer-trained parent and the new candidate, so its repair cannot be credited to this fine-tune. The inconsistent synthetic-start fixture still fails, which is why the startup code fix is essential.

The scripted free-running probe remains mixed. From the same 15 nonterminal recorded roots and idle/right/left sequences, the parent runs 3,159 steps with 4 charge errors, no position errors, and 14 steps carrying invalid charge. The new candidate runs 3,184 steps with 5 charge errors, one position error, and 15 invalid-charge steps. Predicted death changes the denominators. Better controller stress and one-step metrics do not establish glitch-free interaction. The new candidate still emits seven illegal charges in the broader stress set. No hard range mask or native controller replacement was added.

### Use the local candidate

```sh
uv run gymemu play-state --config logs/controller-repair-20260923/playback.yaml --autoregressive
```

The normal configuration still points to the unchanged published model, now with corrected recorded-state initialization. The explicit configuration above selects the new local candidate and the existing published decoder. No Hugging Face files were uploaded or replaced.

Artifacts are under `logs/controller-repair-20260923/` and `runs/controller-repair-20260923/`. `run.py` reproduces preparation and the head-only attempt; in the same Python process, run `interpolate.py`, `finetune.py`, `evaluate_control.py`, `evaluate_final.py`, and `package.py` for the subsequent experiments and selected export. Full features and hidden activations stay in RAM. JSON records retain settings, provenance, every validation check, controller diagnostics and hashes. Standalone local component loading reproduces all 256 tested CPU predictions exactly, and CPU/MPS outputs agree on those rows.

Candidate SHA256: `09bb5df6f46ced6e0df2cc2ce286cc4d11b15a5afafa6c2bdc7278ed00b66b3f`.
