"""Bounded scheduled-feedback throughput test using production loading and training."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from benchmark_training import Tail
from gymemu.approaches import build_approach
from gymemu.batches import CachedBatchLoader
from gymemu.config import compose_config
from gymemu.data import Frames, Windows, read_episodes, resolve_dataset
from gymemu.engine import batch_loss, run_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", default="breakout_scheduled_fast")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epoch", type=int, default=8)
    parser.add_argument("--batches", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--minimum-sps", type=float, default=0)
    parser.add_argument("--override", action="append", default=[], help="Hydra setting override")
    args = parser.parse_args()
    if min(args.epoch, args.batches, args.warmup, args.repeats) < 1:
        parser.error("epoch, batches, warmup and repeats must be positive")
    cfg = compose_config(
        [
            f"recipe={args.recipe}",
            f"trainer.frame_cache={args.cache}",
            *args.override,
        ]
    )
    torch.set_num_threads(cfg.trainer.threads or 2)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    root, provenance = resolve_dataset(cfg.game.dataset, cfg.game.revision)
    frames = Frames(root, compact=True, cache=args.cache)
    episodes = read_episodes(root, cfg.game.train_split)
    action_values = sorted({int(a) for episode in episodes for a in episode.actions})
    spec = OmegaConf.to_container(cfg.approach, resolve=True)
    model = build_approach(spec, cfg.history, len(action_values), frames.shape).cuda()
    model.configure_training(compile=cfg.trainer.compile)
    curriculum = model.begin_epoch(args.epoch)
    data = Windows(
        frames,
        episodes,
        cfg.history,
        action_values,
        action_history=model.action_history,
        rollout_steps=model.training_rollout_steps,
    )
    size = cfg.trainer.batch_size
    count = size * (args.warmup + args.repeats * args.batches)
    if count > len(data):
        parser.error("Requested benchmark exceeds training split size")
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(123))[:count]
    loader = Tail(
        CachedBatchLoader(
            data,
            size,
            cfg.trainer.workers,
            pin_memory=True,
            sampler=order.tolist(),
        )
    )
    # Match model initialization, Adam, targets and order across both recipes.
    from gymemu.optimizers import build_optimizer

    stage = cfg.approach.stages[0]
    optimizer = build_optimizer(
        model.prepare_stage(stage.objective),
        OmegaConf.to_container(cfg.optimizer, resolve=True),
        stage.learning_rate,
    )
    loss_fn = torch.compile(batch_loss) if cfg.trainer.compile else batch_loss

    def measure(batches, label):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = run_epoch(
            model,
            loader,
            torch.device("cuda"),
            optimizer,
            batches,
            precision=cfg.trainer.precision,
            label=label,
            loss_function=loss_fn,
            prefetch=cfg.trainer.prefetch,
            sync_batches=cfg.trainer.sync_batches,
        )
        torch.cuda.synchronize()
        result["seconds"] = time.perf_counter() - started
        result["samples_per_second"] = result["samples"] / result["seconds"]
        return result

    warmup = measure(args.warmup, "warmup")
    trials = [measure(args.batches, f"trial={i + 1}") for i in range(args.repeats)]
    result = {
        "arguments": vars(args),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "dataset": provenance,
        "curriculum": curriculum,
        "order_sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "warmup": warmup,
        "trials": trials,
        "median_samples_per_second": statistics.median(t["samples_per_second"] for t in trials),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "source_sha256": {
            str(path.relative_to(Path(__file__).parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted((Path(__file__).parent / "gymemu").rglob("*.py"))
        },
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "config"}), flush=True)
    if result["median_samples_per_second"] < args.minimum_sps:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
