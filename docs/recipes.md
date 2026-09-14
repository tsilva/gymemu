# Training recipes

Recipes collect the settings for an experiment in YAML. They use the existing Hydra
composition and the same runner for direct and multi-stage approaches.

## Named recipes

| Recipe | Purpose |
| --- | --- |
| `direct` | Original RGB next-frame CNN and uniform MSE |
| `latent` | Frame reconstruction, then prediction with a frozen codec |
| `breakout_cnn` | Settings from the successful ten-epoch CUDA Breakout run |
| `breakout_actions` | Same Breakout experiment with previous executed actions as inputs |
| `breakout_scheduled` | Action-history CNN with progressively sampled prediction feedback |
| `breakout_scheduled_fast` | Same curriculum with optimized feedback execution |

Run a recipe, inspect it, or override settings:

```bash
uv run python train.py recipe=breakout_cnn --cfg job --resolve
uv run python train.py recipe=direct game=custom game.dataset=/path/to/dataset
uv run python train.py recipe=latent experiment=smoke output=runs/latent-smoke
uv run python train.py recipe=direct optimizer=adamw optimizer.weight_decay=0.01
uv run python train.py --multirun recipe=direct,latent seed=47,48
```

Named recipes live in `configs/recipe/`. `experiment` presets apply after recipes;
command-line values apply last. `experiment=smoke` resets loading and compilation
settings as well as bounding the model and training budget.

To derive another recipe, create `configs/recipe/my_cnn.yaml`:

```yaml
# @package _global_
defaults:
  - breakout_cnn
  - _self_
name: breakout-cnn-smaller
model:
  width: 16
trainer:
  learning_rate: 0.0003
  epochs: 5
```

Select it with `recipe=my_cnn`. Dataset IDs, revisions, actions, and recorded starting
scenes belong in `configs/game/`. A recipe selects a game through its defaults:

```yaml
defaults:
  - direct
  - override /game: my_game
  - _self_
```

Model construction stays in its registry, and approach configs define their named
models and ordered stages. Stage epochs and learning rates inherit trainer values
unless overridden individually. `optimizer=adam` preserves the original optimizer;
its betas, epsilon, weight decay, and AMSGrad setting are explicit YAML values.
`optimizer=adamw` selects decoupled weight decay. The learning rate remains a stage
setting so multi-stage recipes can tune it independently.

A recipe can select implemented behavior. Other losses, input corruption, or feedback
curricula need an explicit approach implementation before a recipe can enable them.
The direct recipe remains the reference CNN with its original objective.

## Action-history experiment

`recipe=breakout_actions` inherits `breakout_cnn` and selects `approach=direct_actions`
with `model=action_history_cnn`. The default eight action slots contain seven previous
executed actions followed by the current action. Each slot becomes its own one-hot
spatial planes, concatenated with the eight RGB frames before the first convolution.
The remaining encoder/decoder and uniform next-frame RGB MSE are unchanged.

```bash
uv run python train.py recipe=breakout_actions --cfg job --resolve
uv run python train.py recipe=breakout_actions output=runs/breakout-actions
uv run python play.py runs/breakout-actions/best.pt

# Tune how much action history to supply (includes the current action)
uv run python train.py recipe=breakout_actions model.action_history=4
```

`model.action_history` defaults to `${history}` and accepts 1 through `history`.
With eight RGB frames, four action slots mean three previous actions plus the current
one. Missing actions at episode start use `START`, never a game's NOOP. Bootstrap
examples have zero RGB history and all-`START` action slots. No future action is input.
The scene exported by training includes recorded actions between its frames; playback
then records the actions actually pressed and resets both histories together.

This experiment tests whether action history reduces prediction ambiguity. RGB motion
already provides velocity information; action history may add information about hidden
controller state or make dynamics easier to learn. Neither input choice guarantees
full observability or prevents MSE averaging. Check paddle and ball behavior in generated
rollouts as well as held-out MSE.

Data revision, targets, seed, width, optimizer, and epoch budget match `breakout_cnn`.
The extra input planes increase parameters from 343,875 to 358,211 for Breakout, so
this is not a parameter-matched comparison. `model.action_history=1` provides a
current-action-only architecture control. It matches the original CNN's outputs when
given the same weights; separately initialized runs need not produce identical weights.
No full training result is recorded for this recipe yet.

## Scheduled frame feedback

`recipe=breakout_scheduled` trains the action-history CNN from scratch, using the same
dataset revision, width, optimizer, seed, ten epochs, and supervised targets as
`breakout_actions`. It selects `approach=scheduled_actions` and exposes these settings:

```yaml
approach:
  options:
    rollout_steps: ${history}
    schedule:
      warmup_epochs: 2
      ramp_epochs: 6
      max_probability: 0.8
```

The feedback probability is zero in epochs 1 and 2, then 13.3%, 26.7%, 40%, 53.3%,
66.7%, and 80% in epochs 3 through 8. Epochs 9 and 10 stay at 80%. The probability
changes at epoch boundaries. These values are an initial experiment, not a measured
optimal schedule. A short smoke can finish entirely inside warmup unless overridden.

For each supervised target, the loader supplies enough earlier recorded frames and
executed actions to generate a bounded prefix. Starting from recorded context, the
model predicts each prefix frame in chronological order. An independent coin flip
for each example and timestep chooses that whole prediction or the corresponding
recorded frame. The chosen frame enters the context for the next prediction. These
are fresh predictions from the current weights, including errors from earlier sampled
predictions, rather than a fixed cache of generated frames.

Eight prefix steps can replace all eight frames in the final context. Shorter prefixes
replace only its newest frames. Episode boundaries remain intact: nonexistent frames
stay zero, while a valid episode-initial frame may be replaced by the model's bootstrap
prediction using all-START actions. Actions always come from the aligned recorded
trajectory. The frame being predicted and later frames are never available as inputs
to its generation.

Only the final prediction receives uniform RGB MSE against its recorded target.
Feedback generation runs without gradients, so this experiment trains recovery from
imperfect context without backpropagating through the rollout. The model may still
lose information after severe drift; this curriculum does not guarantee recovery or
correct collision dynamics. It differs from optimizing a multi-step rollout loss.

```bash
# Uses the same verified cache as the previous Breakout recipes
uv run python train.py recipe=breakout_scheduled output=runs/breakout-scheduled

# Test all-generated context after the ramp
uv run python train.py recipe=breakout_scheduled \
  approach.options.schedule.max_probability=1.0

# Shorter feedback horizon to reduce compute
uv run python train.py recipe=breakout_scheduled approach.options.rollout_steps=4

# Exercise feedback in a bounded local smoke, not just warmup
uv run python train.py recipe=breakout_scheduled game=custom game.dataset=/path/to/fixture \
  experiment=smoke approach.options.schedule.warmup_epochs=0 \
  approach.options.schedule.ramp_epochs=1
```

`rollout_steps` accepts 1 through `history`; `max_probability` accepts 0 through 1.
Set `max_probability=0` for a recorded-context control with the same architecture.
Warmup skips the extra model forwards, although the loader still supplies extended
prefixes. Once feedback starts, each batch does `rollout_steps` extra forward passes
without gradients plus one supervised forward/backward pass. The epoch/sample budget
matches the reference; the compute budget does not. Do not extrapolate total runtime
from the first two epochs.

`metrics.jsonl` and checkpoint metadata record each epoch's probability and prefix
length under `curriculum`. Training loss uses mixed context. The comparison metric
and checkpoint selection continue to use float32 RGB MSE on unchanged held-out targets
with recorded histories. Playback needs no curriculum configuration; it uses generated
frames throughout, as before. Separately inspect long rollouts and paddle contacts
before concluding that recovery improved. See [history.md](history.md#scheduled-sampling)
for the research basis.

### Optimized feedback execution

`recipe=breakout_scheduled_fast` inherits the original scheduled recipe and adds:

```yaml
model:
  factor_actions: true
approach:
  options:
    feedback_dtype: autocast
    selective_threshold: 0.2
```

`factor_actions` computes the constant one-hot planes' contribution using the existing
first-convolution weights during autocast training. Float32 evaluation/playback retain
the original convolution path. It includes the zero-padding boundaries and keeps all
weight names and parameter counts. `feedback_dtype=autocast` avoids repeatedly widening generated
frames to float32 during bf16 training; float32 operation stays float32. Targets and
comparison MSE remain float32.

Below `selective_threshold`, independent per-example/per-step Bernoulli masks are sampled
on the CPU. The approach batches each example's first selected frame, then its second,
and so on. Every selected prediction uses the earlier selections in its own context.
Unused forwards are omitted, and small inference batches are padded to multiples of 16
with duplicate rows that are discarded afterward. At higher probabilities, the dense
compiled loop is faster. The threshold controls execution, not the selection probability.
Set it to `0` to force dense execution; `1` selects sparse execution whenever probability
is less than one. The default `0.2` selects it only in epoch 3 of this recipe.

The original recipe defaults to `factor_actions=false`, `feedback_dtype=fp32`, and
`selective_threshold=0`. The fast recipe keeps its dataset, batch size, optimizer,
epochs, rollout length, and curriculum. Changed arithmetic grouping and CPU versus
CUDA RNG mean separately trained weights need not match bit for bit. Replaying the
same saved recipe and environment remains the reproducibility contract. No completed
training/rollout-quality result is claimed for this optimization.

```bash
uv run python train.py recipe=breakout_scheduled_fast \
  trainer.frame_cache=/path/to/verified-cache output=runs/breakout-scheduled-fast
```

See [matched measurements](performance.md#scheduled-feedback-optimization). A benchmark
does not resume the stopped training run or write a replacement checkpoint.

## Reproduce the successful Breakout run

`recipe=breakout_cnn` records the run completed on September 13, 2026:

- Dataset revision `676ff6388f4218d3c3a3ce9f2f33e075fa7314a3`.
- Original CNN, width 32, eight full RGB history frames, seed 47.
- Adam at 0.001, batch size 64, ten epochs, bf16 training and float32 evaluation.
- All 5,387,476 training and 1,343,754 held-out examples per epoch.
- Verified LZ4 cache, direct pinned batches, compilation, and CUDA prefetch.
- Best held-out RGB MSE **0.00003076008331352503**, selected at epoch 8.
- Total training and evaluation time **7,483.9 seconds**, about 2 hours 5 minutes.

These are observed reference results, not guarantees for every rerun. The original
source archive SHA-256 is
`355502918199cf8932004252107f0c94f5c116dd95c2770796b96c16288d076e`.
The local run directory is `runs/beast3-20260913T115132Z`; its original source archive,
resolved configuration, metrics, downloaded checkpoints, and delivery receipt remain
there. The named recipe changes the run name and default cache/output locations.

On a CUDA host, build the lossless cache once:

```bash
uv sync --frozen
uv run python cache_frames.py \
  --dataset tsilva/gradlab-breakout-trajectories \
  --revision 676ff6388f4218d3c3a3ce9f2f33e075fa7314a3 \
  --output data/breakout-676ff638-lz4
uv run python train.py recipe=breakout_cnn output=runs/breakout-cnn
```

To reuse an existing verified cache, override `trainer.frame_cache=/path/to/cache`.
The cache builder requires a new destination and checks every frame's lossless
roundtrip. Training verifies cache identity against the dataset.

## Replay a run's saved YAML

Every new run saves `recipe.yaml` alongside `resolved.yaml`. The recipe contains all
job settings without depending on the current named presets. It retains internal
references such as `${model}` and `${trainer.epochs}` so the same tuning knobs work.
Environment-dependent values are captured at run time. The actual dataset revision
and local resource paths are recorded, and output is reset to null for a fresh run.

```bash
uv run python train.py --recipe runs/breakout-cnn/recipe.yaml --cfg job --resolve
uv run python train.py --recipe runs/breakout-cnn/recipe.yaml output=runs/replay
uv run python train.py --recipe runs/breakout-cnn/recipe.yaml \
  trainer.epochs=5 output=runs/shorter
uv run python train.py --recipe runs/breakout-cnn/recipe.yaml \
  --multirun seed=47,48 hydra.sweep.dir=runs/replay-seeds
```

`--recipe` accepts an existing standalone `.yaml` file, including paths with spaces.
A saved recipe can be copied elsewhere. Relocate `trainer.frame_cache`,
`game.dataset`, or `game.start_scene` when moving between machines. A local dataset
with identical contents retains its fingerprint even when moved. Changed data is
rejected; explicitly set `expected_dataset=null` when starting a different experiment.

Use named recipes for switching configuration groups, such as `approach=latent`.
Standalone recipes have already composed those groups, so override their individual
values, such as `optimizer.kind=adamw`, instead. Explicit per-stage values stay fixed;
only stages that inherit `${trainer.epochs}` change with `trainer.epochs`.

The Python equivalent is `compose_config(overrides, recipe=path)` followed by `train`.
Existing legacy flags and ordinary Hydra commands remain supported.

## Ball-coordinate experiment

`recipe=breakout_ball` inherits `breakout_actions`' pinned dataset, eight-frame
history, executed-action history, optimizer, and ten-epoch training budget. It uses
a new `ball_state_cnn` model and the `ball_state` approach.

```bash
uv run python train.py recipe=breakout_ball output=runs/breakout-ball
uv run python play.py runs/breakout-ball/best.pt
```

This retains the parent recipe's CUDA, bfloat16, compilation, and existing frame-cache
path. For a bounded CPU smoke without that cache:

```bash
uv run python train.py recipe=breakout_ball experiment=smoke wandb.mode=disabled r2.enabled=false
```

Each RGB history frame has an aligned row containing `ball_x_normalized`,
`ball_y_normalized`, and a label-availability flag. Both output heads share the
convolutional encoder; predicted coordinates also condition the RGB decoder.

```text
loss = next_frame_rgb_mse + coordinate_loss_weight * masked_coordinate_mse
```

The coordinate term averages squared x/y error, masks unavailable targets, and then
averages over the full batch. The default weight is `0.01`, an initial tuning choice
without a completed quality comparison. Override it with
`approach.options.coordinate_loss_weight=0.001`, for example. Validation `loss`
includes both terms; validation `mse` and best-checkpoint selection remain float32
RGB MSE on the same held-out targets as the other approaches.

The Gradlab adapter reads scalar normalized labels from transition `record_json`.
They belong to the successor frame. Coordinates follow episode/step order, so
reused image IDs do not merge distinct trajectory states. RGB and coordinate
histories use the same oldest-to-newest window and left zero padding. Initial-frame
labels are absent in this dataset: that row has zero coordinates and availability
zero, and its coordinate loss is masked. It never borrows the first successor's
labels. Missing or invalid successor labels are errors. Values use the provider's
recorded normalization directly, with no division by image width or height.

The saved `start-scene.npz` includes aligned recorded coordinates. Each subsequent
action feeds back predicted RGB and coordinates; reset restores both histories.
`--empty-start` starts both histories at zero; the initial coordinate output has no
direct reset-label supervision. Existing named RGB/action snapshots lack coordinates
and are rejected. Save a new named state from a coordinate-aware run's scene instead.

The intended benefit is better visual ball tracking, but a joint loss does not
establish that result. Compare generated rollouts and ball visibility against
`breakout_actions`, alongside the unchanged RGB metric. Training and evaluation use
recorded histories, so both RGB and coordinate errors can accumulate in playback.

## Code, environment, and limits

A recipe alone cannot recreate code or numerical libraries that have changed.
Each new run therefore also saves:

- `source.tar.gz`, containing root Python scripts, the `gymemu` package, YAML configs,
  curated `start_states` snapshots, `pyproject.toml`, `uv.lock`, and `.python-version`,
  including uncommitted source edits.
- `reproduction.json`, recording source and recipe SHA-256 hashes, Git commit/status,
  dataset identity, package versions, device, CUDA/cuDNN versions, CPU threads, and
  determinism flags. It does not dump environment variables.

To recover the captured implementation, extract the source archive into a new directory,
run `uv sync --frozen` there, and replay the saved recipe with that directory's `train.py`.
The archive does not contain datasets, caches, external scene files, or checkpoints.
Those remain separate artifacts and paths recorded by the run.

Checkpoints and completed summaries include the recipe hash. Preserve the complete run
directory when keeping a result. The existing `compare.py` continues to group runs by
matching held-out targets and data identity.

Replaying starts training from the recorded seed. It does not resume optimizer state.
Matching settings, code, data, and dependencies supports reproducible experiments;
CUDA kernels, compilation, hardware, and numerical nondeterminism can still change
weights and scores. CPU smoke tests verify exact replay for both existing approaches.
Rollout quality remains a separate evaluation from next-frame MSE.
