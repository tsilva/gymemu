"""Identical sample order, optimizer and model through the actual production epoch runner."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gymemu.approaches import build_approach
from gymemu.data import Frames, Windows, read_episodes, resolve_dataset
from gymemu.engine import run_epoch


class Tail:
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.iterator = iter(loader)
        self.length = len(loader)

    def __iter__(self):
        return self.iterator

    def __len__(self):
        return self.length


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cache")
    p.add_argument("--optimized", action="store_true")
    p.add_argument("--threaded", action="store_true")
    p.add_argument("--batches", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--minimum-sps", type=float, default=0)
    a = p.parse_args()
    if a.batches < 1 or a.warmup < 1 or a.workers < 0:
        p.error("batches and warmup must be positive; workers must be nonnegative")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(47)
    torch.backends.cudnn.benchmark = True
    root, _ = resolve_dataset(
        "tsilva/gradlab-breakout-trajectories", "676ff6388f4218d3c3a3ce9f2f33e075fa7314a3"
    )
    kwargs = {"cache": a.cache} if a.cache else {}
    frames = Frames(root, compact=True, **kwargs)
    data = Windows(frames, read_episodes(root, "train"), 8, [0, 1, 2])
    indices = torch.randperm(len(data), generator=torch.Generator().manual_seed(123))[
        : 64 * (a.batches + a.warmup)
    ].tolist()
    opts = dict(batch_size=64, num_workers=a.workers, pin_memory=True, sampler=indices)
    if a.workers:
        opts.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
    if a.threaded:
        from gymemu.batches import CachedBatchLoader

        loader = Tail(CachedBatchLoader(data, 64, a.workers, pin_memory=True, sampler=indices))
    else:
        loader = Tail(DataLoader(data, **opts))
    spec = {"kind": "direct", "models": {"predictor": {"kind": "direct_cnn", "width": 32}}}
    torch.manual_seed(47)
    model = build_approach(spec, 8, 3, (3, 210, 160)).cuda()
    optimizer = torch.optim.Adam(model.prepare_stage("next_frame"), lr=0.001)
    opts = {}
    if a.optimized:
        from gymemu.engine import batch_loss

        opts = dict(loss_function=torch.compile(batch_loss), prefetch=True, sync_batches=100)
    started = time.monotonic()
    run_epoch(
        model,
        loader,
        torch.device("cuda"),
        optimizer,
        a.warmup,
        precision="bf16",
        label="warmup",
        **opts,
    )
    torch.cuda.synchronize()
    warmup = time.monotonic() - started
    started = time.monotonic()
    result = run_epoch(
        model,
        loader,
        torch.device("cuda"),
        optimizer,
        a.batches,
        precision="bf16",
        label="measured",
        **opts,
    )
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    result.update(
        vars(a),
        seconds=seconds,
        warmup_seconds=warmup,
        samples_per_second=result["samples"] / seconds,
    )
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if result["samples_per_second"] < a.minimum_sps:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
