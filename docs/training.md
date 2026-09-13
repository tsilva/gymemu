# Training guide

See the [README](../README.md) for setup, playback, and comparison commands.

## Hierarchical configuration

[Hydra](https://hydra.cc/docs/intro/) composes YAML defaults and command-line overrides.
`configs/config.yaml` selects the defaults. Named `recipe` files collect complete
experiment settings; `experiment` presets apply afterward for runtime or smoke overrides. `hydra.job.chdir` is false, so relative dataset and scene paths remain relative
to the directory where you launch the command.

| Group | Controls |
| --- | --- |
| `game` | Dataset, revision, splits, bindings, and recorded scene |
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
and each prediction uses the previous actions actually pressed plus the fresh key press.

## Named debug start states

`play.py CHECKPOINT --start-state NAME` loads a named snapshot from
`start_states/<game>/NAME.npz`. `--list-start-states` lists names and descriptions for
the checkpoint's game. `--start-state`, `--start-scene`, and `--empty-start` are mutually
exclusive; omitting all three preserves the checkpoint's normal recorded start.
The repository's library is found regardless of the current working directory.
Use `--state-dir /path/to/library` to select a different library.

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
