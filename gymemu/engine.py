"""Shared experiment runner: ordered stages, held-out RGB evaluation, and run artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader

from gymemu.approaches import build_approach
from gymemu.batches import CachedBatchLoader
from gymemu.checkpoints import load_model, save_model
from gymemu.data import Frames, Windows, read_episodes, resolve_dataset
from gymemu.diagnostics import Health, RolloutProbe
from gymemu.optimizers import build_optimizer, validate_optimizer
from gymemu.recipes import save_reproduction
from gymemu.runtime import device_for
from gymemu.scenes import write_scene
from gymemu.storage import CheckpointStore, validate_storage
from gymemu.tracking import track_run, validate_tracking
from gymemu.training_state import (
    EpochSampler,
    TrainingInterrupted,
    cpu_state,
    load_training,
    restore_rng,
    rng_state,
    save_training,
    stop_signals,
    training_contract,
)


def checkpoint_due(started, seconds):
    return time.monotonic() - started >= seconds


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate(config):
    validate_tracking(config)
    validate_storage(config)
    _positive(config["history"], "history")
    validate_optimizer(config["optimizer"])
    trainer = config["trainer"]
    for name in ("batch_size", "epochs", "checkpoint_seconds", "sync_batches"):
        _positive(trainer[name], name)
    for name in ("threads", "limit_episodes", "train_batches", "eval_batches"):
        if trainer[name] is not None:
            _positive(trainer[name], name)
    if type(trainer["workers"]) is not int or trainer["workers"] < 0:
        raise ValueError("workers must be nonnegative")
    if trainer["precision"] not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if type(trainer.get("compile_layout_optimization", True)) is not bool:
        raise ValueError("compile_layout_optimization must be boolean")
    if trainer["device"] not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("device must be auto, cpu, cuda, or mps")
    if config["game"]["train_split"] == config["game"]["eval_split"]:
        raise ValueError("Training and evaluation splits must differ")
    if trainer["loader"] not in ("standard", "cached"):
        raise ValueError("loader must be standard or cached")
    if trainer["loader"] == "cached" and not trainer["frame_cache"]:
        raise ValueError("cached loader requires trainer.frame_cache")
    diagnostics = trainer.get("diagnostics", {})
    if diagnostics.get("enabled", False):
        for name in ("log_every", "probe_every", "samples", "horizon"):
            _positive(diagnostics[name], f"diagnostics.{name}")
    stages = config["approach"]["stages"]
    if not stages or len({s["name"] for s in stages}) != len(stages):
        raise ValueError("Stages must be nonempty with unique names")
    for stage in stages:
        name = stage["name"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
            raise ValueError(
                "Stage names must use lowercase letters, digits, underscores or hyphens"
            )
        _positive(stage["epochs"], f"{name}.epochs")
        rate = stage["learning_rate"]
        if not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("learning_rate must be finite and positive")


def device_batches(loader, count, device, prefetch=False):
    """Overlap pinned host transfers with computation without changing sample order."""
    iterator = iter(islice(loader, count))
    if not prefetch or device.type != "cuda":
        for batch in iterator:
            yield tuple(value.to(device, non_blocking=True) for value in batch)
        return
    stream = torch.cuda.Stream(device=device)

    def copy_next():
        batch = next(iterator, None)
        if batch is None:
            return None
        with torch.cuda.stream(stream):
            return tuple(value.to(device, non_blocking=True) for value in batch)

    pending = copy_next()
    while pending is not None:
        current = torch.cuda.current_stream(device)
        current.wait_stream(stream)
        batch = pending
        for value in batch:
            value.record_stream(current)
        pending = copy_next()
        yield batch


def batch_loss(model, history, action, target, compact, training, *auxiliary):
    history, target = history.float(), target.float()
    if compact:
        history = history / 255
        target = target / 255
    if training:
        return model.loss(history, action, target, *auxiliary), target.new_zeros(())
    loss, prediction = model.evaluate(history, action, target, *auxiliary)
    return loss, F.mse_loss(prediction.float(), target)


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    max_batches=None,
    *,
    precision="fp32",
    label="",
    checkpoint=None,
    checkpoint_seconds=600,
    prefetch=False,
    loss_function=batch_loss,
    diagnostic_loss_function=None,
    sync_batches=1,
    health=None,
    report=None,
    log_every=100,
    probe_every=1000,
    resume_state=None,
    state_checkpoint=None,
    stop_requested=lambda: False,
):
    model.train(optimizer is not None)
    model.reset_epoch_metrics()
    samples = batches = 0
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_mse = torch.zeros_like(total_loss)
    started = last_checkpoint = time.monotonic()
    elapsed = 0.0
    if resume_state:
        samples, batches = resume_state["samples"], resume_state["batches"]
        total_loss.copy_(resume_state["total_loss"])
        total_mse.copy_(resume_state["total_mse"])
        elapsed = resume_state["seconds"]
        for name, buffer in model.named_buffers():
            buffer.copy_(resume_state["buffers"][name])
        if health is not None:
            health.load_state_dict(resume_state["health"], device)
    count = len(loader) + batches
    count = min(count, max_batches) if max_batches else count

    def progress():
        return {
            "samples": samples,
            "batches": batches,
            "total_loss": total_loss,
            "total_mse": total_mse,
            "seconds": elapsed + time.monotonic() - started,
            "buffers": dict(model.named_buffers()),
            "health": health.state_dict() if health is not None else None,
        }

    for history, action, target, *auxiliary in device_batches(
        loader, count - batches, device, prefetch
    ):
        detailed = health is not None and (batches < 10 or (batches + 1) % log_every == 0)
        capture = health.capture(detailed) if health is not None else nullcontext()
        # Sampled activation hooks stay outside the compiled graph. Changing hooks
        # must not cause recompilation or make regular updates collect activations.
        compute_loss = (
            diagnostic_loss_function if detailed and diagnostic_loss_function else loss_function
        )
        with torch.set_grad_enabled(optimizer is not None), capture:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                loss, mse = compute_loss(
                    model,
                    history,
                    action,
                    target,
                    loader.dataset.frames.compact,
                    optimizer is not None,
                    *auxiliary,
                )
            if sync_batches == 1 and not torch.isfinite(loss):
                raise ValueError("Non-finite training or validation loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if health is not None:
                    health.backward(loss, len(target))
                optimizer.step()
                if health is not None:
                    health.updated()
        total_loss += loss.detach().double() * len(target)
        total_mse += mse.detach().double() * len(target)
        samples += len(target)
        batches += 1
        if health is not None and (
            batches <= 10
            or batches % log_every == 0
            or batches % probe_every == 0
            or batches == count
        ):
            report(
                health.metrics(), batches, samples, batches % probe_every == 0 or batches == count
            )
        stopping = stop_requested()
        save = (checkpoint or state_checkpoint) and (
            stopping or checkpoint_due(last_checkpoint, checkpoint_seconds)
        )
        if batches % sync_batches == 0 or batches == count or save:
            # Check before every checkpoint; CUDA runs may defer host synchronization.
            if not torch.isfinite(total_loss) or not torch.isfinite(total_mse):
                raise ValueError("Non-finite training or validation loss / pixel MSE")
        if batches % 100 == 0 or batches == count:
            print(
                f"{label} {'train' if optimizer else 'eval'} batches={batches}/{count} "
                f"loss={total_loss.item() / samples:.6f} "
                f"samples/s={samples / (elapsed + time.monotonic() - started):.0f}",
                flush=True,
            )
        if save:
            if checkpoint:
                checkpoint(batches, samples)
            if state_checkpoint:
                state_checkpoint(progress())
            last_checkpoint = time.monotonic()
        if stopping:
            raise TrainingInterrupted("Training stopped; resume.pt saved after the last update")
    if not samples:
        raise ValueError("An epoch must contain at least one sample")
    result = {
        "loss": total_loss.item() / samples,
        "samples": samples,
        "batches": batches,
        "seconds": elapsed + time.monotonic() - started,
    }
    if optimizer is None:
        result["mse"] = total_mse.item() / samples
    diagnostics = model.epoch_metrics()
    if result.keys() & diagnostics.keys():
        raise ValueError("Approach diagnostics must not replace shared epoch metrics")
    result.update(diagnostics)
    if state_checkpoint:
        state_checkpoint(progress())
    return result


def _dataset_identity(root, provenance):
    if "revision" in provenance and provenance["revision"]:
        return {"dataset": provenance["dataset"], "revision": provenance["revision"]}
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.parquet")):
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    return {"local_content_sha256": digest.hexdigest()}


def _evaluation_contract(identity, episodes, dataset, trainer, metric="next_frame_rgb_mse"):
    count = len(dataset)
    if trainer["eval_batches"] is not None:
        count = min(count, trainer["eval_batches"] * trainer["batch_size"])
    digest = hashlib.sha256()
    remaining = count
    for episode in episodes:
        take = min(remaining, len(episode.frames))
        digest.update(
            json.dumps(
                [
                    int(episode.episode_id),
                    episode.frames[:take].tolist(),
                    episode.actions[: max(0, take - 1)].tolist(),
                ]
            ).encode()
        )
        remaining -= take
        if not remaining:
            break
    return {
        "dataset": identity,
        "targets_sha256": digest.hexdigest(),
        "samples": count,
        "metric": metric,
        "normalization": "uint8/255",
        "bootstrap": "included",
        "shape": list(dataset.frames.shape),
    }


def _output_directory(config):
    if HydraConfig.initialized():
        hydra = HydraConfig.get()
        if hydra.mode == RunMode.MULTIRUN and config["output"] is not None:
            raise ValueError("output must be null during a sweep; set hydra.sweep.dir instead")
        output = Path(config["output"] or hydra.runtime.output_dir)
        hydra_output = Path(hydra.runtime.output_dir).resolve()
    else:
        if config["output"] is None:
            raise ValueError("output must be set when using the Python training interface")
        output = Path(config["output"])
        hydra_output = None
    output = output.expanduser().resolve()
    allowed = {".hydra", "train.log"} if output == hydra_output else set()
    if output.exists() and any(p.name not in allowed for p in output.iterdir()):
        raise ValueError("Output directory must be empty; choose a new run directory")
    return output


def train(cfg):
    with stop_signals() as stop_requested:
        try:
            return _train(cfg, stop_requested)
        except TrainingInterrupted as error:
            print(str(error), flush=True)
            output = cfg.output if cfg.output is not None else HydraConfig.get().runtime.output_dir
            return Path(output).expanduser().resolve()


def _train(cfg, stop_requested):
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    validate(config)
    recovery = load_training(config["resume"]) if config.get("resume") else None
    output = _output_directory(config)
    storage = CheckpointStore(config, output)
    trainer, game, spec = config["trainer"], config["game"], config["approach"]
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if trainer["threads"]:
        torch.set_num_threads(trainer["threads"])
    device = device_for(trainer["device"])
    if recovery and (
        recovery["device_type"] != device.type
        or recovery["torch_version"] != str(torch.__version__)
    ):
        raise ValueError("Resume requires the original device type and PyTorch version")
    if trainer["precision"] == "bf16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("bf16 requires a CUDA GPU with bfloat16 support")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print("Loading episode trajectories.", flush=True)
    root, provenance = resolve_dataset(game["dataset"], game["revision"])
    # Freeze mutable Hub references and local paths in replay artifacts.
    game["revision"] = provenance["revision"]
    if provenance["revision"] is None:
        game["dataset"] = str(root)
    if trainer["frame_cache"]:
        trainer["frame_cache"] = str(Path(trainer["frame_cache"]).expanduser().resolve())
    if game["start_scene"]:
        game["start_scene"] = str(Path(game["start_scene"]).expanduser().resolve())
    frames = Frames(root, compact=True, cache=trainer["frame_cache"])
    # Check all episode IDs before applying smoke limits.
    train_episodes = read_episodes(root, game["train_split"])
    eval_episodes = read_episodes(root, game["eval_split"])
    if {e.episode_id for e in train_episodes} & {e.episode_id for e in eval_episodes}:
        raise ValueError("Training and evaluation episode IDs overlap")
    splits = {game["train_split"]: train_episodes, game["eval_split"]: eval_episodes}
    if trainer["limit_episodes"]:
        train_episodes = train_episodes[: trainer["limit_episodes"]]
        eval_episodes = eval_episodes[: trainer["limit_episodes"]]
    actions = sorted({int(a) for episode in train_episodes for a in episode.actions})
    if not actions:
        raise ValueError("Training requires at least one executed action")
    model = build_approach(spec, config["history"], len(actions), frames.shape).to(device)
    model.validate_stages(spec["stages"])
    model.configure_training(compile=trainer["compile"])
    if model.state_fields:
        # The approach declares its input contract; the data adapter owns label alignment.
        splits = {
            split: read_episodes(root, split, state_fields=model.state_fields) for split in splits
        }
        train_episodes = splits[game["train_split"]][: trainer["limit_episodes"]]
        eval_episodes = splits[game["eval_split"]][: trainer["limit_episodes"]]
    input_options = {"action_history": model.action_history, "state_fields": model.state_fields}
    training = Windows(
        frames,
        train_episodes,
        config["history"],
        actions,
        rollout_steps=model.training_rollout_steps,
        future_steps=model.training_future_steps,
        **input_options,
    )
    evaluation = Windows(frames, eval_episodes, config["history"], actions, **input_options)
    loss_function = (
        torch.compile(
            batch_loss,
            options={"layout_optimization": trainer.get("compile_layout_optimization", True)},
        )
        if trainer["compile"]
        else batch_loss
    )
    identity = _dataset_identity(root, provenance)
    if config.get("expected_dataset") is not None and config["expected_dataset"] != identity:
        raise ValueError(
            "Dataset differs from saved recipe; set expected_dataset=null for a new experiment"
        )
    config["expected_dataset"] = identity
    if recovery and training_contract(config) != training_contract(recovery["training_config"]):
        raise ValueError("Resume cannot change the model, dataset, optimizer, or training recipe")
    evaluation_contract = _evaluation_contract(
        identity, eval_episodes, evaluation, trainer, model.evaluation_metric
    )
    config["output"] = str(output)
    metadata = {
        "format_version": 2,
        "history": config["history"],
        "shape": list(frames.shape),
        "action_values": actions,
        "action_history": model.action_history,
        "state_fields": list(model.state_fields),
        "action_padding": "start_token",
        "action_order": "oldest_to_current",
        "approach": spec,
        "dataset": provenance,
        "game": game,
        "key_actions": game["key_actions"],
        "seed": config["seed"],
        "padding": "left_zero",
        "bootstrap": "empty_history_and_start_action",
        "objective": model.evaluation_metric,
        "evaluation": evaluation_contract,
    }
    if recovery:
        metadata["resumed_from"] = str(Path(config["resume"]).expanduser().resolve())
        model.load_state_dict(recovery["state_dict"], strict=True)
    if (root / "manifest.json").exists():
        metadata["dataset_manifest"] = json.loads((root / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive marker also protects simultaneous jobs choosing the same directory.
    with (output / "resolved.yaml").open("x") as stream:
        stream.write(OmegaConf.to_yaml(OmegaConf.create(config)))
    recipe_sha256 = save_reproduction(output, config, cfg, device, evaluation_contract["dataset"])
    metadata["recipe_sha256"] = recipe_sha256
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_scene(output / "start-scene.npz", game, metadata, frames, splits)
    options = {
        "batch_size": trainer["batch_size"],
        "num_workers": trainer["workers"],
        "pin_memory": device.type == "cuda",
        "generator": torch.Generator().manual_seed(config["seed"]),
    }
    sampler = EpochSampler(training, config["seed"])
    if trainer["workers"]:
        options.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
    if trainer["loader"] == "cached":
        cached_options = dict(
            batch_size=trainer["batch_size"],
            workers=trainer["workers"],
            pin_memory=device.type == "cuda",
        )
        train_loader = CachedBatchLoader(training, sampler=sampler, **cached_options)
        eval_loader = CachedBatchLoader(evaluation, shuffle=False, **cached_options)
    else:
        train_loader = DataLoader(training, sampler=sampler, **options)
        eval_loader = DataLoader(evaluation, shuffle=False, **options)
    diagnostics = trainer.get("diagnostics", {})
    probe = None
    if diagnostics.get("enabled", False) and model.predictive_objectives:
        probe = RolloutProbe(
            evaluation,
            samples=diagnostics["samples"],
            horizon=diagnostics["horizon"],
            sprite=game.get("ball_sprite"),
        )
        (output / "probe.json").write_text(json.dumps(probe.manifest(), indent=2) + "\n")
    parameters = sum(p.numel() for p in model.parameters())
    print(
        json.dumps(
            {
                "approach": spec["kind"],
                "parameters": parameters,
                "training_examples": len(training),
                "evaluation_examples": len(evaluation),
            }
        )
    )
    with track_run(
        config,
        output,
        evaluation=evaluation_contract,
        parameters=parameters,
        training_examples=len(training),
        recipe_sha256=recipe_sha256,
    ) as tracker:
        if recovery:
            save_training(
                output / "resume.pt",
                {
                    **recovery,
                    "training_config": config,
                    "config": {**recovery["config"], "resumed_from": metadata["resumed_from"]},
                },
            )
        storage.sync()
        best_rgb = recovery["best_rgb"] if recovery else None
        total_train_samples = recovery["total_train_samples"] if recovery else 0
        total_updates = recovery["total_updates"] if recovery else 0
        elapsed = recovery["elapsed_seconds"] if recovery else 0.0
        started = time.monotonic()
        for stage_index, stage in enumerate(spec["stages"]):
            if recovery and stage_index < recovery["stage_index"]:
                continue
            continuing = recovery is not None and stage_index == recovery["stage_index"]
            final = stage_index == len(spec["stages"]) - 1
            stage_dir = output / "stages" / stage["name"]
            stage_dir.mkdir(parents=True)
            optimizer = build_optimizer(
                model.prepare_stage(stage["objective"]), config["optimizer"], stage["learning_rate"]
            )
            best = recovery["best"] if continuing else float("inf")
            best_bundle = recovery["best_bundle"] if continuing else None
            if continuing:
                optimizer.load_state_dict(recovery["optimizer"])
                if best_bundle is not None:
                    # Carry forward the stage winner even if no resumed epoch improves it.
                    torch.save(best_bundle, stage_dir / "best.pt")
                    if final:
                        torch.save(best_bundle, output / "best.pt")
                restore_rng(recovery["rng"])
            first_epoch = recovery["epoch"] + int(recovery["epoch_complete"]) if continuing else 1
            for epoch in range(first_epoch, stage["epochs"] + 1):
                partial = (
                    recovery["progress"]
                    if continuing and epoch == recovery["epoch"] and not recovery["epoch_complete"]
                    else None
                )
                sampler.epoch = sum(s["epochs"] for s in spec["stages"][:stage_index]) + epoch
                sampler.offset = partial["samples"] if partial else 0
                curriculum = model.begin_epoch(epoch)
                progress = {
                    **metadata,
                    "stage": stage["name"],
                    "epoch": epoch,
                    "curriculum": curriculum,
                }

                def snapshot(batches, samples):
                    info = {
                        **progress,
                        "epoch_complete": False,
                        "train_batches_completed": batches,
                        "train_samples_completed": samples,
                    }
                    save_model(stage_dir / "latest.pt", model, info)
                    if final:
                        save_model(output / "latest.pt", model, info)

                def save_progress(state, epoch_complete=False):
                    save_training(
                        output / "resume.pt",
                        {
                            "config": progress,
                            "state_dict": model.state_dict(),
                            "training_config": config,
                            "optimizer": optimizer.state_dict(),
                            "rng": rng_state(device),
                            "progress": state,
                            "stage_index": stage_index,
                            "epoch": epoch,
                            "epoch_complete": epoch_complete,
                            "total_updates": total_updates,
                            "total_train_samples": total_train_samples,
                            "best": best,
                            "best_bundle": best_bundle,
                            "best_rgb": best_rgb,
                            "device_type": device.type,
                            "torch_version": str(torch.__version__),
                            "elapsed_seconds": elapsed + time.monotonic() - started,
                        },
                    )
                    storage.sync()

                def report_health(metrics, batches, samples, include_probe):
                    step = total_updates + batches
                    if batches:
                        metrics.update(model.interval_metrics())
                    metrics.update(
                        {
                            "train/step": step,
                            "train/samples": total_train_samples + samples,
                            "train/stage": stage["name"],
                            "train/epoch": epoch,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/horizon": curriculum.get("rollout_steps", 1),
                        }
                    )
                    media_path = None
                    if (
                        probe is not None
                        and include_probe
                        and stage["objective"] in model.predictive_objectives
                    ):
                        media_path = output / "diagnostics" / f"{stage['name']}-{epoch}-{step}.png"
                        metrics.update(probe.run(model, device, media_path))
                    tracker.log_diagnostics(metrics, output, media_path)

                if probe is not None and partial is None:
                    report_health({}, 0, 0, True)
                health = (
                    Health(model, start_step=total_updates)
                    if diagnostics.get("enabled", False)
                    else None
                )
                if partial:
                    # Construction, tracking and startup probes must not advance training RNG.
                    restore_rng(recovery["rng"])
                result = run_epoch(
                    model,
                    train_loader,
                    device,
                    optimizer,
                    trainer["train_batches"],
                    loss_function=loss_function,
                    diagnostic_loss_function=batch_loss,
                    prefetch=trainer["prefetch"],
                    sync_batches=trainer["sync_batches"],
                    precision=trainer["precision"],
                    label=f"{stage['name']} epoch={epoch}",
                    checkpoint=snapshot,
                    checkpoint_seconds=trainer["checkpoint_seconds"],
                    health=health,
                    report=report_health,
                    log_every=diagnostics.get("log_every", 100),
                    probe_every=diagnostics.get("probe_every", 1000),
                    resume_state=partial,
                    state_checkpoint=save_progress,
                    stop_requested=stop_requested,
                )
                snapshot(result["batches"], result["samples"])
                if stop_requested():
                    raise TrainingInterrupted("Training stopped; resume.pt saved before evaluation")
                validation = run_epoch(
                    model,
                    eval_loader,
                    device,
                    max_batches=trainer["eval_batches"],
                    loss_function=loss_function,
                    prefetch=trainer["prefetch"],
                    sync_batches=trainer["sync_batches"],
                    precision="fp32",
                    label=f"{stage['name']} epoch={epoch}",
                    stop_requested=stop_requested,
                )
                total_train_samples += result["samples"]
                total_updates += result["batches"]
                record = {
                    "stage": stage["name"],
                    "objective": stage["objective"],
                    "epoch": epoch,
                    "curriculum": curriculum,
                    "train": result,
                    "validation": validation,
                }
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                tracker.log_epoch(
                    record,
                    optimizer_steps=total_updates,
                    train_samples_seen=total_train_samples,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    final=final,
                )
                complete = {**progress, "epoch_complete": True, "validation": validation}
                save_model(stage_dir / f"epoch-{epoch:04d}.pt", model, complete)
                save_model(stage_dir / "last.pt", model, complete)
                if final:
                    save_model(output / "last.pt", model, complete)
                metric = validation["mse"] if final else validation["loss"]
                if metric < best:
                    best = metric
                    save_model(stage_dir / "best.pt", model, complete)
                    best_bundle = {"config": complete, "state_dict": cpu_state(model.state_dict())}
                    if final:
                        best_rgb = metric
                        save_model(output / "best.pt", model, complete)
                save_progress(None, epoch_complete=True)
                if stop_requested():
                    raise TrainingInterrupted("Training stopped; resume.pt saved after evaluation")
            # The next stage starts from the previous stage's best validated weights.
            restored, _ = load_model(stage_dir / "best.pt", device)
            model.load_state_dict(restored.state_dict())
        summary = {
            **storage.summary(),
            "status": "complete",
            "recipe_sha256": recipe_sha256,
            "optimizer": config["optimizer"],
            "history": config["history"],
            "action_history": model.action_history,
            "seconds": elapsed + time.monotonic() - started,
            "optimizer_steps": total_updates,
            "train_samples_seen": total_train_samples,
            "name": config["name"],
            "approach": spec["kind"],
            "seed": config["seed"],
            "parameters": parameters,
            "best_mse": best_rgb,
            "evaluation": evaluation_contract,
            "training": {
                "stages": spec["stages"],
                "samples": len(training),
                "batch_size": trainer["batch_size"],
                "train_batches": trainer["train_batches"],
            },
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        storage.sync(final=True)
        tracker.complete(summary)
    print(f"Checkpoint: {output / 'best.pt'}", flush=True)
    return output
