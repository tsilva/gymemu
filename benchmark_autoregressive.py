"""Bounded real-data autoregressive benchmark, including optional health/probe costs."""

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
from gymemu.diagnostics import Health, RolloutProbe
from gymemu.engine import batch_loss, run_epoch
from gymemu.optimizers import build_optimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", help="Standalone resolved recipe YAML")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--epoch", type=int, default=4)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--warmup-samples", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--minimum-sps", type=float, default=0)
    parser.add_argument("--disable-layout-optimization", action="store_true")
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
    )
    args = parser.parse_args()
    if min(args.epoch, args.samples, args.warmup_samples, args.repeats) < 1:
        parser.error("epoch, samples, warmup-samples and repeats must be positive")
    overrides = [] if args.recipe else ["recipe=breakout_autoregressive_ball_region"]
    cfg = compose_config([*overrides, *args.override], recipe=args.recipe)
    size = cfg.trainer.batch_size
    if args.samples % size or args.warmup_samples % size:
        parser.error("sample counts must be divisible by batch size")
    if (
        cfg.trainer.compile
        and cfg.trainer.get("diagnostics", {}).get("enabled", False)
        and args.warmup_samples // size <= 10
    ):
        parser.error("compiled diagnostic runs need more than 10 warmup batches")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.trainer.threads or 2)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    root, provenance = resolve_dataset(cfg.game.dataset, cfg.game.revision)
    frames = Frames(root, compact=True, cache=cfg.trainer.frame_cache)
    episodes = read_episodes(root, cfg.game.train_split)
    actions = sorted({int(a) for episode in episodes for a in episode.actions})
    spec = OmegaConf.to_container(cfg.approach, resolve=True)
    model = build_approach(spec, cfg.history, len(actions), frames.shape).to(device)
    model.configure_training(compile=cfg.trainer.compile)
    curriculum = model.begin_epoch(args.epoch)
    data = Windows(
        frames,
        episodes,
        cfg.history,
        actions,
        action_history=model.action_history,
        rollout_steps=model.training_rollout_steps,
        future_steps=model.training_future_steps,
    )
    count = args.warmup_samples + args.samples * args.repeats
    if count > len(data):
        parser.error("Requested benchmark exceeds the training split")
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
    stage = cfg.approach.stages[0]
    optimizer = build_optimizer(
        model.prepare_stage(stage.objective),
        OmegaConf.to_container(cfg.optimizer, resolve=True),
        stage.learning_rate,
    )
    diagnostics = cfg.trainer.get("diagnostics", {})
    enabled = diagnostics.get("enabled", False)
    probe = None
    if enabled:
        heldout = Windows(
            frames,
            read_episodes(root, cfg.game.eval_split),
            cfg.history,
            actions,
            action_history=model.action_history,
        )
        probe = RolloutProbe(
            heldout,
            samples=diagnostics["samples"],
            horizon=diagnostics["horizon"],
            sprite=cfg.game.get("ball_sprite"),
        )
    keep_layout = args.disable_layout_optimization or not cfg.trainer.get(
        "compile_layout_optimization", True
    )
    if keep_layout and args.compile_mode != "default":
        parser.error("layout override requires the default compiler mode")
    compile_options = (
        {"options": {"layout_optimization": False}} if keep_layout else {"mode": args.compile_mode}
    )
    loss_function = (
        torch.compile(batch_loss, **compile_options) if cfg.trainer.compile else batch_loss
    )

    def measure(samples, label):
        reports = probes = 0
        probe_seconds = 0.0

        def report(metrics, batches, seen, include_probe):
            nonlocal reports, probes, probe_seconds
            metrics.update(model.interval_metrics())
            if include_probe:
                started = time.perf_counter()
                metrics.update(probe.run(model, device, out.parent / f"{out.stem}-probe.png"))
                probe_seconds += time.perf_counter() - started
                probes += 1
            with out.with_suffix(".diagnostics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            reports += 1

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = run_epoch(
            model,
            loader,
            device,
            optimizer,
            samples // size,
            precision=cfg.trainer.precision,
            label=label,
            loss_function=loss_function,
            diagnostic_loss_function=batch_loss,
            prefetch=cfg.trainer.prefetch,
            sync_batches=cfg.trainer.sync_batches,
            health=Health(model) if enabled else None,
            report=report,
            log_every=diagnostics.get("log_every", 100),
            probe_every=diagnostics.get("probe_every", 1000),
        )
        torch.cuda.synchronize()
        result["seconds"] = time.perf_counter() - started
        result["samples_per_second"] = result["samples"] / result["seconds"]
        result.update(
            reports=reports,
            probes=probes,
            probe_seconds=probe_seconds,
            peak_memory_bytes=torch.cuda.max_memory_allocated(),
        )
        print(json.dumps({"label": label, **result}), flush=True)
        return result

    warmup = measure(args.warmup_samples, "warmup")
    trials = [measure(args.samples, f"trial={i + 1}") for i in range(args.repeats)]
    result = {
        "arguments": vars(args),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "dataset": provenance,
        "frame_cache_manifest_sha256": hashlib.sha256(
            (Path(cfg.trainer.frame_cache) / "manifest.json").read_bytes()
        ).hexdigest(),
        "curriculum": curriculum,
        "order_sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "warmup": warmup,
        "trials": trials,
        "median_samples_per_second": statistics.median(t["samples_per_second"] for t in trials),
        "timing_scope": "Loading, transfer, forward/backward, Adam, health, local probes/media. "
        "Excludes dataset setup, warmup, epoch validation, checkpoints, W&B and R2 uploads.",
        "source_sha256": {
            str(path.relative_to(Path(__file__).parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted((Path(__file__).parent / "gymemu").rglob("*.py"))
        },
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"median_samples_per_second": result["median_samples_per_second"]}))
    if result["median_samples_per_second"] < args.minimum_sps:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
