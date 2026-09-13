# Training guide

See the [README](../README.md) for setup and player controls.

## Training and checkpoints

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

