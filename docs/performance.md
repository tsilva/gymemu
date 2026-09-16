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

## Autoregressive training

The September 16 investigation stopped `gymemu-ar-diag-20260916-105712` on Beast-3
and preserved its source and latest inference checkpoint. The benchmark uses the
same RTX 4090, PyTorch 2.14.0+cu130, full-resolution Breakout data, width-32 CNN,
eight RGB/action history slots, and Adam at 0.001. The speech service remains
running and reserves about 2.3 GiB of GPU memory. GPU benchmark jobs run serially.

The original recipe uses batch size 8, bf16 convolutions, float32 feedback history,
eager loss computation, two cache-loading threads, and overlapping CUDA transfers.
It supervises every valid future frame, retaining gradients through the full
rollout. The measured eight-step phase is materially more expensive than the
two-step phase that was active when the run was stopped.

`recipe=breakout_autoregressive_fast` explicitly selects compilation, bf16 feedback
history and the tuned batch size. The original recipe
remains available. These execution options do not shorten rollouts, detach
predictions, crop images, remove targets, or disable diagnostics. Float32 playback
uses the original action-history CNN arithmetic and tensor names. Mixed-precision
regrouping changes numerical rounding; a larger batch also changes Adam's update
count and optimization trajectory. Throughput alone is not evidence of better
trained rollouts.

The shared health collector reuses parameter gradient norms and transfers scalar
statistics together when reporting. Every-update gradient coverage, maximum spikes,
weight-gradient collapse signals, and sampled activation/update statistics remain.
Sampled activation hooks execute on the eager loss path so their installation does
not trigger compiled-graph variants or record hooks on ordinary updates.

### Numerical checks and rejected variants

On 32 real training windows from the stopped checkpoint, including bootstrap and
episode-tail cases, bf16 feedback alone matched the reference loss and gave gradient
cosine similarity 0.999950, with relative gradient L2 difference 1.01%. Adding
compilation while disabling automatic layout optimization gave cosine 0.999975,
relative gradient difference 0.78%, and loss 0.002807224 versus reference 0.002807802.
This is one diagnostic batch, not a training-quality equivalence result.

Factored action inputs produced cosine approximately -0.35 on that batch despite
only a 0.11% loss difference. The compiler's default automatic layout conversion
also changed the gradient substantially, with cosine 0.61. The fast recipe therefore
keeps the original action-plane implementation and sets
`trainer.compile_layout_optimization=false`. Neither faster pilot variant is counted
as an adopted speedup. Float32 inference differences were at most 1.2e-7, the same
scale observed between repeated reference forwards. A separate 110-update CUDA
check found exactly equal health metrics and weights before/after the diagnostic
collector optimization.

The `max-autotune-no-cudagraphs` pilot was stopped after 7.5 minutes of compilation
without a timed trial. It was not adopted. Compiler children were stopped before
subsequent timing runs. Increasing loader workers from one to four had less than
1% effect in the exploratory action-factoring variant; the final sweep keeps two.

### Final matched results

The eight-step comparisons each use a 2,048-sequence warmup followed by three
32,768-sequence trials, totaling 98,304 timed sequence starts per configuration.
All three comparisons have the same complete sample-order SHA-256:
`c602bb9a3dc7ed485ef00f54a62e01c4c55928c0b98e1121bcdab99ce4353511`.

| Eight-step training path | Batch size | Median sequences/sec | Speedup |
| --- | ---: | ---: | ---: |
| Stopped run's original code and settings | 8 | 461.45 | 1.00× |
| Optimized execution, original batch size | 8 | 798.33 | 1.73× |
| `breakout_autoregressive_fast` | 32 | 1,069.31 | **2.32×** |

The adopted batch-32 recipe's three trials measured 1,069.31 / 1,069.22 / 1,069.35
sequences/sec. Final batch-size pilots without action factoring or automatic
layout conversion measured 1,064 / 1,055 / 971 sequences/sec at batches 32 / 64 /
128, respectively. These pilots used 16,384 timed starts each. Batch 32 was the
best measured setting and also requires fewer activation buffers than the larger
candidates. Keeping batch size 8 retains the original number of optimizer updates
per epoch while still providing the measured 1.73× execution gain.

The two-step phase active at interruption was also checked with one matched
32,768-sequence trial per configuration: 1,452.95 sequences/sec originally and
3,613.00 with the batch-32 fast recipe, a **2.49×** gain. These shorter checks are
separate from the three-trial eight-step result. All final timing runs exceeded
the original 1,000 sequences/sec target when using the tuned batch-32 recipe.

A 25-second GPU sample during the optimized eight-step trials averaged 95.8%
utilization, 364 W, and 74.6°C, including probe-related dips. Total device memory
usage was approximately 5.5 GiB including the speech service and CUDA reservations.
These values describe the measured workload; unused VRAM does not imply that larger
batches will improve throughput. This is the best tested configuration within the
numerical constraints above, not a proof of the hardware's absolute throughput limit.

Validation passed 241 tests and `ruff check .`. Bounded direct and two-stage latent
CPU train/checkpoint/play smokes passed. The final fast recipe also passed a real-data
CUDA smoke with eight differentiable rollout steps, compiled ordinary updates,
diagnostics, float32 validation, checkpoint save/load, teacher-forcing replay, and
autoregressive playback from its saved starting scene. The original training run
remains stopped; these checks did not restart a full experiment.

### Measurement method

`benchmark_autoregressive.py` invokes the production loader, loss, optimizer,
health collector, and `run_epoch`. Seed 47 controls initialization; seed 123 fixes
sample order. A sample means a sequence start, not an individual predicted frame.
The resolved config and sample-order hash accompany every result. Input windows
never cross episodes, and probes only observe fixed held-out trajectories.

Measurements include loading, host-to-device transfer, forward/backward, Adam,
health reporting, and local probe images/JSON. They exclude dataset/cache
verification, warmup/compilation, full-epoch validation, checkpoints, W&B upload,
and R2 upload. Thus these are training-loop measurements, not full-job wall time.
CUDA synchronizes at timing boundaries. The existing health cadence remains every
100 updates, with eight 32-step probes every 1,000 updates and at measurement end.
The first ten updates in each measured segment also retain startup diagnostics;
short segments therefore underestimate sustained throughput, especially for large
batches. Compilation requires more than ten warmup batches to reach the ordinary
compiled path before timing.

Raw results and diagnostic scripts are under `logs/autoregressive-tuning/` locally
and `/home/tsilva/.local/share/gymemu/container-workspace/throughput-20260916/` on
Beast-3. The remote `baseline/` is the stopped run's immutable source copy;
`feedback/` contains the optimized execution code. `stopped-latest.pt` preserves
the pre-stop inference bundle. It does not contain Adam state for exact training
continuation.

### Reproduce

Run on a CUDA host with a verified frame cache. The benchmark does not create a
W&B run, upload artifacts, or save trained checkpoints.

```bash
uv run python benchmark_autoregressive.py \
  --override trainer.frame_cache=data/breakout-676ff638-lz4 \
  --epoch 4 --samples 32768 --warmup-samples 2048 --repeats 3 \
  --out logs/ar-reference.json

uv run python benchmark_autoregressive.py \
  --override recipe=breakout_autoregressive_fast \
  --override trainer.frame_cache=data/breakout-676ff638-lz4 \
  --epoch 4 --samples 32768 --warmup-samples 2048 --repeats 3 \
  --minimum-sps 1000 --out logs/ar-fast.json
```

Use the preserved `baseline/` source to reproduce the original health collector
exactly. A local snapshot can be selected with `--override game.dataset=/path/to/snapshot`.
Use `--epoch 2` to measure the two-step phase, or override
`trainer.batch_size=8` on the fast recipe to retain the original update batch size.

## Detached rollout training

The September 16 detached-feedback benchmark used Beast-3's RTX 4090, the pinned
Breakout dataset and existing verified container runtime. GPU jobs ran serially;
desktop services occupied about 2.3 GiB. Every variant used full-resolution RGB,
history 8, horizon 8, per-step RGB and ball-region losses, and detached generated
history. Model initialization used seed 47 and sample ordering used seed 123.
Health statistics, activation samples, and held-out rollout probes remained enabled.

An initial screen used 2,048 warmup windows and three trials of 4,096 windows each.
Median windows per second were:

| Execution | Batch size | Windows/s |
| --- | ---: | ---: |
| Eager, reference layout | 32 | 843 |
| Compiled, reference layout | 32 | 1,101 |
| Compiled, reference layout | 64 | 1,058 |
| Compiled, reference layout | 128 | 878 |
| Compiled, automatic layout | 32 | 1,377 |
| Compiled, factored action inputs | 32 | 1,303 |
| Compiled, automatic layout and factored action inputs | 32 | 1,415 |
| Compiled, reference layout, four loader workers | 32 | 1,103 |

Short trials penalize larger batches because each segment repeats ten detailed
startup updates and an ending rollout probe. Use the longer measurements below to
choose a batch size. Four loader workers offered no material gain over two.

The longer comparison used 4,096 warmup windows and three trials of 32,768 windows
each. All seven variants used identical sample sequences, SHA-256
`b20a3b6dd40ef297f498dab8fbf013c5f3853c814e7bf2deef94b66ac25dbc3a`.

| Execution | Batch size | Windows/s | Peak allocated GiB |
| --- | ---: | ---: | ---: |
| Compiled, reference layout | 32 | 1,251 | 1.95 |
| Compiled, automatic layout | 32 | 1,657 | 1.95 |
| Automatic layout and factored action inputs | 16 | 1,384 | 0.94 |
| Automatic layout and factored action inputs | 32 | 1,677 | 1.80 |
| Automatic layout and factored action inputs | 64 | **1,795** | 3.53 |
| Automatic layout and factored action inputs | 128 | 1,750 | 6.98 |
| Automatic layout and factored action inputs | 256 | 1,599 | 13.86 |

`recipe=breakout_detached_fast` selects the fastest measured variant, batch 64 with
automatic layout and factored inputs. It was 1.435 times as fast as the compiled
batch-32 reference, with three trial speeds of 1,795, 1,792 and 1,797 windows/s.
This is a measured choice among these settings, not a claim that every possible
implementation has been exhausted. The batch change halves updates per full epoch;
learning rate remains 0.001. Diagnostics still use their configured update cadence,
so larger batches also process more windows between reports and probes.

Reproduce the selected settings on Beast-3 with the verified cache:

```bash
uv run python benchmark_autoregressive.py \
  --override recipe=breakout_detached_fast \
  --override trainer.frame_cache=/workspace/frame-cache/676ff638-lz4 \
  --epoch 4 --warmup-samples 4096 --samples 32768 --repeats 3 \
  --out logs/detached-fast.json
```

The trained-checkpoint check used four fixed real training batches at horizon 8.
Compared with eager detached training, compiled reference-layout loss changed by
at most 0.33%, with minimum gradient cosine 0.984. Automatic layout changed loss
by at most 0.68%, with minimum cosine 0.941. Factored inputs changed loss by at most
1.05%, with minimum cosine 0.963. Combining layout conversion and factoring changed
loss by at most 0.78%, with minimum cosine 0.943. Thus these paths preserve the
mathematical objective but are not numerically identical. Float32 inference uses the
original convolution path even when training factors the action inputs.

A separate continuation check compared eager detached training at batch 32 with the
selected fast recipe at batch 64. Each arm processed the same 32,000 sampled training
windows per seed, giving 1,000 reference updates and 500 fast updates. All arms started
from the completed horizon-4 checkpoint with fresh Adam at 0.001. Evaluation reused
512 held-out one-step targets and 31 fixed starts with up to 32 generated steps.

| Batch-sequence seed | Eager rollout MSE | Fast rollout MSE | Change |
| --- | ---: | ---: | ---: |
| 47 | 0.00199559 | 0.00196663 | -1.5% |
| 48 | 0.00184779 | 0.00189265 | +2.4% |

Peak gradient norms remained below 0.005 in all four arms. Fast one-step MSE improved
in both repeats; ball-region rollout MSE changed by +4.1% and -10.1%. These small,
mixed changes support a bounded stability check, not statistical equivalence or a
guarantee of unchanged full-run quality. Batch size and execution arithmetic both
changed, so this check measures the combined recipe. All repeats share one pretrained
checkpoint. Raw quality results and the compiled CUDA curriculum/playback smoke are
saved beside the throughput measurements.

Benchmark timing includes loading, host-to-device transfer, forward/backward, Adam,
health statistics, and local rollout probes. It excludes dataset/cache verification,
compilation warmup, full validation, checkpoint writes, W&B and R2 uploads. Reports
include resolved configs, sample-order hashes, peak allocated memory, runtime versions,
and source hashes. Raw results are in `runs/detached-throughput-20260916/` locally and
`/workspace/diagnostics/detached-throughput-20260916/` on Beast-3's persistent volume.
