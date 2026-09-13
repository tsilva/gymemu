# Training performance

## Workload and method

The September 13, 2026 investigation used Beast-3's RTX 4090 and Ryzen 5 7600X,
PyTorch 2.14 with CUDA 13.0, and Breakout dataset revision
`676ff6388f4218d3c3a3ce9f2f33e075fa7314a3`.
Other GPU workloads remained running throughout the measurements.

Each matched test used the original direct CNN with width 32, eight full-resolution
210 × 160 RGB history frames, batch size 64, Adam at 0.001, and bf16 training.
The model seed was 47 and sample-order seed 123. Tests used actual training windows,
including loading, transfer, forward, backward, and optimizer updates. A 100-batch
warmup preceded 3,000 measured batches. Setup, compilation, and cache construction
are separate costs. The benchmark invokes the production `run_epoch` function.

## Matched results

| Production training path | Samples | Seconds | Samples/sec |
| --- | ---: | ---: | ---: |
| Original code at `295012a` | 192,000 | 94.581 | 2,030 |
| Cache, direct batches, compiled loss, CUDA prefetch | 192,000 | 20.707 | 9,272 |

The optimized path was **4.57× faster**. A preceding 32,000-sample probe measured
9,229 samples/sec. These results exceed the requested 3× throughput improvement.
They exclude one-time cache creation and cold compilation costs.

Raw matched results are retained locally under
`artifacts/throughput/results/` and remotely under
`/home/tsilva/.config/gymemu/runs/20260913T111327Z/` as
`baseline-final1.json` and `optimized-final1.json`.

## What limited throughput

The original 300-batch probe measured 2,056 training samples/sec. Its loader alone
managed 2,100 samples/sec, while a resident GPU batch reached 8,399 samples/sec.
This isolated frame decoding and batch construction as the initial bottleneck.

A 5,000-frame codec probe measured an 18.1× reduction in decode time with lossless
LZ4 instead of WebP. Caching alone raised full training throughput to about
3,500 samples/sec. Multiprocessing copies and per-frame Python work then dominated.
Resolving frame IDs in batches and decompressing directly into pinned arrays removed
those costs. GPU transfers overlap computation; compilation includes RGB
normalization and loss calculation. Held-out evaluation now computes the direct
CNN prediction once per batch.

The full cache contains all 5,924,824 frames, takes 7.8 GiB, and took 195.5 seconds
to build. Construction verified every decoded byte against its compressed roundtrip.
Loading verifies source hashes, cache hashes, and frame IDs. It never modifies the
source Parquet files. A changed source requires a new cache directory.

## Reproduce

Build the cache as shown in the README. On a CUDA host, run:

```bash
uv run python benchmark_training.py --batches 3000 --out logs/baseline.json
uv run python benchmark_training.py --batches 3000 --optimized --threaded \
  --workers 2 --cache data/breakout-cache --minimum-sps 6170 \
  --out logs/optimized.json
```

The script reproduces this specific Breakout workload. To reproduce the original
baseline exactly, copy the script into a checkout of commit `295012a` and run its
first command there. Current default training also includes reduced metric
synchronization and the duplicate-evaluation fix.

`experiment=cuda_cached` selects the optimized path. Set `trainer.frame_cache` to
a verified cache directory. Other compatible RGB datasets can use the same cache
builder and loader; dimensions, colors, scalar actions, and episode boundaries
remain unchanged. Memory grows with batch dimensions and the bounded prefetch
queue, not with the number of decoded frames in the dataset.

The optimized preset checks loss finiteness every 100 batches, at epoch end, and
before every checkpoint. A non-finite loss aborts without writing a new checkpoint,
but up to 100 updates may occur before detection. Set `trainer.sync_batches=1`
for immediate checking. Default training retains immediate checks.

Compilation can change floating-point rounding and the resulting optimization
trajectory. Pixel inputs, sample order, architecture, objective, and training budget
are preserved; bit-identical learned weights are not promised. Validation still
uses float32 RGB MSE. Throughput varies with other users of the shared GPU, and a
short throughput test does not establish final model quality.

## Scheduled feedback optimization

The September 13 follow-up measured the stopped scheduled-sampling workload on
Beast-3's RTX 4090. The baseline already used the LZ4 cache, compiled loss, bf16,
CUDA prefetch, and two loader workers. Eight prefix forwards per target had become
the main cost. In a resident-batch trace, encoder/decoder convolutions accounted
for about half the GPU time; the first encoder convolution alone took about 23%.
Repeated casts, padding, concatenation, and layout conversion added substantial cost.

`recipe=breakout_scheduled_fast` makes three execution changes:

- Keep mixed histories in the autocast dtype instead of widening every prediction
  and concatenating float32 histories between bf16 forwards.
- Factor the first convolution into RGB convolution and a small action calculation,
  using the same weights and exact spatial padding geometry during autocast training.
  Float32 evaluation and playback use the original convolution implementation.
- Below 20% selection probability, batch by each example's replacement rank. Its
  first selected frame is generated with other examples' first selections, then
  second selections, etc. Skipped frames remain recorded. At higher probabilities,
  use the dense compiled loop because grouping/indexing overhead outweighs savings.

The matched end-to-end runs use the production `run_epoch`, random training windows,
batch size 64, history 8, action history 8, prefix length 8, width 32, Adam 0.001,
and the same dataset revision and model seed as above. Each comparison uses identical
sample indices (order seed 123; SHA-256
`b20a3b6dd40ef297f498dab8fbf013c5f3853c814e7bf2deef94b66ac25dbc3a`).
A 100-batch warmup precedes three 500-batch trials, totaling 96,000 measured targets.
Both recipes start from the same seeded initialization; `--epoch` selects curriculum
probability, not a resumed trained checkpoint. Training losses are not quality results.
All timings include loading, H2D transfer, loss, backward, and Adam updates, with CUDA
synchronization at measurement boundaries. Setup, compilation/warmup, evaluation,
and checkpoint writes are excluded. GPU jobs ran serially; background services retained
about 2.5 GiB of GPU memory, and the long training run remained stopped.

| Feedback probability | Original samples/sec | Fast samples/sec | Speedup |
| --- | ---: | ---: | ---: |
| 13.3% (epoch 3) | 2,872 | 4,833 | **1.68×** |
| 80% (epochs 8–10) | 2,869 | 3,899 | **1.36×** |

These gains do **not** meet 4×. They also do not imply the same gain for a full run,
which includes recorded-context warmup, evaluation, compilation, and checkpoints.
The older live-run speed of roughly 2,068 samples/sec is not the matched baseline;
host contention and run conditions differed.

In isolated resident-GPU tests, bf16 history alone raised throughput from 2,883 to
3,304 samples/sec. Explicit channels-last formatting added no benefit. Selective
execution with factored inputs reached about 6,367 samples/sec at 13.3%, but real
loading/transfers reduced that gain. Loader trials at the same low probability
measured 3,340 / 4,834 / 4,191 samples/sec with one / two / four workers, respectively.
Forcing selective execution at 80% achieved only 2,463 samples/sec. An isolated
max-autotune trial offered about another 4% for dense execution but was not adopted
as a default. No rollout shortening, smaller model, target subsampling, or extra
supervised losses are counted as throughput improvements.

### Reproduce the feedback measurements

On Beast-3, source is staged at
`/home/tsilva/.config/gymemu/feedback-optimization/source`. From that directory:

```bash
uv sync --frozen
uv run python benchmark_feedback.py --recipe breakout_scheduled --epoch 3 \
  --cache /home/tsilva/.cache/gymemu/676ff638-lz4 --out ../baseline-3.json
uv run python benchmark_feedback.py --recipe breakout_scheduled_fast --epoch 3 \
  --cache /home/tsilva/.cache/gymemu/676ff638-lz4 --out ../fast-3.json
# Repeat both commands with --epoch 8 and separate output paths.
```

`--override trainer.workers=4` measures a loader alternative;
`--override approach.options.selective_threshold=1.0` forces selective execution
below 100% feedback. `--minimum-sps` provides a performance gate (exit code 2 below
the requested median throughput). Reports include full resolved config, data identity,
sample-order hash, runtime, warmup time, individual trial timings, and source hashes.
The benchmark is bounded and does not save training checkpoints or restart a run.

The final comparison uses `baseline-3.json`, `baseline-8.json`,
`final-fast-3.json`, and `final-fast-8.json`. Raw results are retained in
`logs/feedback-optimization/results/` locally and
`/home/tsilva/.config/gymemu/feedback-optimization/` remotely. Prototype ablations and
the profiler trace are retained under the ignored diagnostic directory, outside
production modules.

Tests cover factored outputs and parameter gradients on padded and unpadded canvases;
identical-mask selective/dense feedback; empty and START contexts; detached prefixes;
final-loss gradients; graph reuse as the probability changes; and saved-recipe replay,
checkpoint loading, and playback. Evaluation remains float32 next-frame RGB MSE.
Arithmetic grouping and CPU versus CUDA sampling change floating-point rounding and
RNG streams, so learned weights are not bit-identical across execution recipes.
Final rollout quality still requires a completed training experiment.

A trained action-history checkpoint was also checked on 64 real training windows,
including episode starts. The final float32 inference path differed by at most
1.2e-7 in that CUDA check. Under bf16, factored versus original output difference
MSE was 6.23e-8 (target MSE approximately 4.267e-4 for this particular batch).
For an eight-step identical-mask rollout, regrouping gave float32 context difference
MSE 1.97e-14, with a maximum of 2.8e-4 at individual pixels; a four-example float64
check agreed within 3.1e-15. This separates arithmetic/recurrence rounding from
dependency-order correctness. The numerical check is retained as `verify.py` and
`verify.log` in the remote diagnostic directory, with a local diagnostic copy.
