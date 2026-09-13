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
