# gymemu

Train a neural emulator from recorded trajectories, then play inside its predictions.
This is a fresh, deliberately plain baseline: an RGB frame stack and the current
executed action go through a convolutional encoder/decoder to predict the next RGB frame.

## Install

Python 3.13 and [uv](https://docs.astral.sh/uv/) are used for the locked environment.

```bash
uv sync --frozen
```

## Train

```bash
uv run --frozen python train.py --output runs/breakout-001
```

The default dataset is
[`tsilva/gradlab-breakout-trajectories`](https://huggingface.co/datasets/tsilva/gradlab-breakout-trajectories),
pinned to commit `b8091d248295eb5135011dd9b943c75f4a7d50be`. It has 1,440,267 training
transitions and 359,075 held-out transitions. Downloading the Parquet files directly
does not depend on Hugging Face's Dataset Viewer jobs being healthy.

Defaults: 8 history frames, 32 base channels, batch size 32, Adam at 0.001, 10 epochs,
and CUDA, MPS, or CPU selected in that order. `--device` overrides device selection.
For an RTX 4090 with enough CPU/RAM for six loader workers, start with:

```bash
uv run --frozen python train.py --output runs/cuda-001 --device cuda \
  --precision bf16 --workers 6 --threads 2 --batch-size 64 --epochs 1
```

This runs one full training epoch and the full held-out split. Batch size 64 gives
22,513 optimizer updates over 1,440,781 examples, including episode starts. Learning
rate remains 0.001. Use `--epochs 10` for the original ten-pass budget.

CUDA loading uses pinned memory and nonblocking transfers. Images stay as exact
`uint8` pixels through loading and transfer, then normalize on the GPU. `--workers`
starts persistent image-decoding processes; `--threads` controls the main process's
PyTorch CPU threads, not the worker count or GPU parallelism. Worker processes use
one PyTorch thread each. CUDA convolution autotuning is enabled.

`--precision bf16` uses [PyTorch autocast](https://docs.pytorch.org/docs/2.10/amp.html)
for convolution computation with float32 MSE and float32 model/optimizer weights.
It requires a CUDA GPU supporting bfloat16. It changes numerical precision, not the
model or objective; bitwise agreement with float32 training is not expected.
The default remains float32 for CPU/MPS compatibility. Checkpoints play locally
in float32 without CUDA.
An existing nonempty output directory is rejected so experiments are not overwritten.

The model has three stride-2 convolutions and three transposed convolutions, ReLUs,
and a sigmoid RGB output. The action is one-hot encoded and broadcast as input channels.
The objective is ordinary mean squared error over every RGB pixel in `[0, 1]`.
Training uses real frame histories and one next-frame target. There are no skip/warp
connections, residual frame predictions, auxiliary labels, weighted losses, recurrent
state, diffusion, or multi-step training objectives.

Images retain their original dimensions and colors. This dataset already has its top
17 HUD rows masked. The network pads the bottom/right to a multiple of eight internally
and crops that padding from its output; there is no dataset resizing or binarization.

Outputs:

- `config.json`: model, data revision, training settings, and capture contract.
- `metrics.jsonl`: epoch train/evaluation pixel MSE, sample counts, batch counts, and time.
- `best.pt`: weights from the lowest evaluation-MSE epoch.
- `last.pt`: weights from the last completed training and evaluation epoch.
- `latest.pt`: inference snapshot every `--checkpoint-seconds` seconds, default 60,
  plus the end of each training pass. Its metadata records the current epoch and
  completed training batches/samples; it has not necessarily completed evaluation.

Progress lines show the current epoch, completed/total batches, cumulative MSE,
examples per second, and estimated seconds remaining in that train/evaluation pass.
Checkpoint files are replaced atomically, so a reader sees a complete previous or
new snapshot. An abrupt stop can lose work since the latest snapshot. These files
allow playback but do not resume optimizer progress.

These checkpoints contain weights and plain configuration, loaded with
`torch.load(weights_only=True)`. They are inference checkpoints; optimizer-state resume
is not implemented. They are incompatible with previous gymemu experiments.

For a bounded installation smoke:

```bash
uv run --frozen python train.py --output runs/smoke --device cpu --threads 2 \
  --epochs 1 --batch-size 4 --limit-episodes 2 --train-batches 8 --eval-batches 2
```

The smoke verifies the pipeline. Its tiny training budget does not produce a playable model.

## Play

```bash
uv run --frozen python play.py runs/breakout-001/best.pt
```

The window initially waits without running the model. Press any action key to generate
the initial frame from an empty stack. That first press initializes the scene; subsequent
presses each execute one action and predict one successor. No real environment or dataset
is loaded during play. Generated frames become the next history; there is no periodic
correction from recorded frames.

| Key | Effect |
| --- | --- |
| Left arrow | Action 2: left |
| Right arrow | Action 1: right |
| Space | Action 0: button/serve |
| R | Clear history and wait for an action key to generate a fresh initial frame |
| Escape | Quit |

Holding a key does not advance the model. Key releases, window refreshes, and unrelated
keys do not run inference. A predicted transition follows the dataset's configured
frame-skip-2 cadence; it is not necessarily a single native emulator frame.

For other integer action vocabularies, provide bindings such as
`--key-action left=10 --key-action right=20 --key-action space=30`.

Headless smoke (first value initializes, the following four execute transitions):

```bash
uv run --frozen python play.py runs/smoke/best.pt --device cpu \
  --headless-actions 0,1,1,0,2 --output logs/smoke.png
```

## History and the initial frame

Every episode contributes one bootstrap example: an all-zero history, a reserved
`START` action category meaning **no game action**, and its recorded initial frame as
the target. Subsequent targets use the available preceding frames, oldest to newest,
with missing history padded on the left with zeros. Short episodes and the first
transition are retained. Histories never cross episode boundaries.

The 514 training episodes have 30 distinct initial images. An empty history and `START`
do not identify a reset seed, so this deterministic MSE baseline can average those
initial images. Bootstrap examples also occur only once per episode, with the same
weight as every other example. This is a limitation to observe in the first experiment,
not evidence that the initial scene will already be accurate after training.

**Eight frames is a practical baseline, not a proven sufficient minimum.** Exact native
Breakout state includes hidden controller charge, repeat acceleration, subpixel positions,
collision state, and a clock-dependent serve. Even 64-frame histories in this dataset
occasionally have different successors for the same action. See [the source and data
investigation](docs/history.md). Use `--history 2`, `4`, `8`, etc. for separate runs.

Evaluation MSE uses real histories. A low value does not establish stable interactive
simulation: the ball occupies few pixels and errors can accumulate when predictions feed
back into the model. Inspect generated ball motion, paddle collisions, and brick persistence
when comparing checkpoints.

## Dataset contract

`--dataset` accepts a Hub dataset ID or a local snapshot directory. Other Hub IDs resolve
the requested `--revision` (or `main`) to an immutable commit before download. The present
adapter supports the following layout and scalar integer actions; arbitrary Hub schemas
are not automatically inferred.

| Location | Required columns |
| --- | --- |
| `frames/assets/*.parquet` | `frame_id`, embedded HF `image` (`bytes`, `path`) |
| `transitions/<split>/*.parquet` | `episode_id`, `step`, `source_frame_id`, `successor_frame_id`, `native_action_json` |
| `episodes/<split>/*.parquet` | `episode_id`, `initial_frame_id`, `length` |

Use `--train-split` and `--eval-split` to select episode partitions. IDs must be unique
within their table, and training/evaluation episode IDs must be disjoint. The loader
sorts transitions by episode and step, checks lengths and frame continuity, joins by
frame ID, and rejects unseen evaluation actions. Shared image bytes across splits are
storage deduplication; held-out transitions remain excluded from optimization.

Only encoded images, numerical trajectory arrays, and a bounded decoded-frame cache
are kept in memory. Windows are assembled on demand. No expanded 1.6-million-frame RGB
tensor or prebuilt history-window dataset is created.

## Development

`train.py` contains the complete training path and shared model/data definitions.
`play.py` provides playback. Old models, preprocessing, cloud-training scripts, and
previous architecture implementations were removed from the working tree.

```bash
uv run --frozen pytest -q
uv run --frozen ruff check .
```

[MIT License](LICENSE)
