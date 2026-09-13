"""Shared experiment runner: ordered stages, held-out RGB evaluation, and run artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
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
from gymemu.optimizers import build_optimizer, validate_optimizer
from gymemu.recipes import save_reproduction
from gymemu.runtime import device_for
from gymemu.scenes import write_scene


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate(config):
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
    if trainer["device"] not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("device must be auto, cpu, cuda, or mps")
    if config["game"]["train_split"] == config["game"]["eval_split"]:
        raise ValueError("Training and evaluation splits must differ")
    if trainer["loader"] not in ("standard", "cached"):
        raise ValueError("loader must be standard or cached")
    if trainer["loader"] == "cached" and not trainer["frame_cache"]:
        raise ValueError("cached loader requires trainer.frame_cache")
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


def batch_loss(model, history, action, target, compact, training):
    history, target = history.float(), target.float()
    if compact:
        history = history / 255
        target = target / 255
    if training:
        return model.loss(history, action, target), target.new_zeros(())
    loss, prediction = model.evaluate(history, action, target)
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
    checkpoint_seconds=60,
    prefetch=False,
    loss_function=batch_loss,
    sync_batches=1,
):
    model.train(optimizer is not None)
    samples = batches = 0
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_mse = torch.zeros_like(total_loss)
    started = last_checkpoint = time.monotonic()
    count = min(len(loader), max_batches) if max_batches else len(loader)
    for history, action, target in device_batches(loader, count, device, prefetch):
        with torch.set_grad_enabled(optimizer is not None):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                loss, mse = loss_function(
                    model,
                    history,
                    action,
                    target,
                    loader.dataset.frames.compact,
                    optimizer is not None,
                )
            if sync_batches == 1 and not torch.isfinite(loss):
                raise ValueError("Non-finite training or validation loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.detach().double() * len(target)
        total_mse += mse.detach().double() * len(target)
        samples += len(target)
        batches += 1
        save = checkpoint and time.monotonic() - last_checkpoint >= checkpoint_seconds
        if batches % sync_batches == 0 or batches == count or save:
            # Check before every checkpoint; CUDA runs may defer host synchronization.
            if not torch.isfinite(total_loss) or not torch.isfinite(total_mse):
                raise ValueError("Non-finite training or validation loss / pixel MSE")
        if batches % 100 == 0 or batches == count:
            print(
                f"{label} {'train' if optimizer else 'eval'} batches={batches}/{count} "
                f"loss={total_loss.item() / samples:.6f} "
                f"samples/s={samples / (time.monotonic() - started):.0f}",
                flush=True,
            )
        if save:
            checkpoint(batches, samples)
            last_checkpoint = time.monotonic()
    if not samples:
        raise ValueError("An epoch must contain at least one sample")
    result = {
        "loss": total_loss.item() / samples,
        "samples": samples,
        "batches": batches,
        "seconds": time.monotonic() - started,
    }
    if optimizer is None:
        result["mse"] = total_mse.item() / samples
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


def _evaluation_contract(identity, episodes, dataset, trainer):
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
        "metric": "next_frame_rgb_mse",
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
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    validate(config)
    output = _output_directory(config)
    trainer, game, spec = config["trainer"], config["game"], config["approach"]
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if trainer["threads"]:
        torch.set_num_threads(trainer["threads"])
    device = device_for(trainer["device"])
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
    input_options = {"action_history": model.action_history}
    training = Windows(
        frames,
        train_episodes,
        config["history"],
        actions,
        rollout_steps=model.training_rollout_steps,
        **input_options,
    )
    evaluation = Windows(frames, eval_episodes, config["history"], actions, **input_options)
    loss_function = torch.compile(batch_loss) if trainer["compile"] else batch_loss
    identity = _dataset_identity(root, provenance)
    if config.get("expected_dataset") is not None and config["expected_dataset"] != identity:
        raise ValueError(
            "Dataset differs from saved recipe; set expected_dataset=null for a new experiment"
        )
    config["expected_dataset"] = identity
    evaluation_contract = _evaluation_contract(identity, eval_episodes, evaluation, trainer)
    config["output"] = str(output)
    metadata = {
        "format_version": 2,
        "history": config["history"],
        "shape": list(frames.shape),
        "action_values": actions,
        "action_history": model.action_history,
        "action_padding": "start_token",
        "action_order": "oldest_to_current",
        "approach": spec,
        "dataset": provenance,
        "game": game,
        "key_actions": game["key_actions"],
        "seed": config["seed"],
        "padding": "left_zero",
        "bootstrap": "empty_history_and_start_action",
        "objective": "next_frame_rgb_mse",
        "evaluation": evaluation_contract,
    }
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
    }
    if trainer["workers"]:
        options.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
    if trainer["loader"] == "cached":
        cached_options = dict(
            batch_size=trainer["batch_size"],
            workers=trainer["workers"],
            pin_memory=device.type == "cuda",
        )
        train_loader = CachedBatchLoader(training, shuffle=True, **cached_options)
        eval_loader = CachedBatchLoader(evaluation, shuffle=False, **cached_options)
    else:
        train_loader = DataLoader(training, shuffle=True, **options)
        eval_loader = DataLoader(evaluation, shuffle=False, **options)
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
    best_rgb = None
    total_train_samples = total_updates = 0
    started = time.monotonic()
    for stage_index, stage in enumerate(spec["stages"]):
        final = stage_index == len(spec["stages"]) - 1
        stage_dir = output / "stages" / stage["name"]
        stage_dir.mkdir(parents=True)
        optimizer = build_optimizer(
            model.prepare_stage(stage["objective"]), config["optimizer"], stage["learning_rate"]
        )
        best = float("inf")
        for epoch in range(1, stage["epochs"] + 1):
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

            result = run_epoch(
                model,
                train_loader,
                device,
                optimizer,
                trainer["train_batches"],
                loss_function=loss_function,
                prefetch=trainer["prefetch"],
                sync_batches=trainer["sync_batches"],
                precision=trainer["precision"],
                label=f"{stage['name']} epoch={epoch}",
                checkpoint=snapshot,
                checkpoint_seconds=trainer["checkpoint_seconds"],
            )
            total_train_samples += result["samples"]
            total_updates += result["batches"]
            snapshot(result["batches"], result["samples"])
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
            )
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
            complete = {**progress, "epoch_complete": True, "validation": validation}
            save_model(stage_dir / "last.pt", model, complete)
            if final:
                save_model(output / "last.pt", model, complete)
            metric = validation["mse"] if final else validation["loss"]
            if metric < best:
                best = metric
                save_model(stage_dir / "best.pt", model, complete)
                if final:
                    best_rgb = metric
                    save_model(output / "best.pt", model, complete)
        # The next stage starts from the previous stage's best validated weights.
        restored, _ = load_model(stage_dir / "best.pt", device)
        model.load_state_dict(restored.state_dict())
    summary = {
        "status": "complete",
        "recipe_sha256": recipe_sha256,
        "optimizer": config["optimizer"],
        "history": config["history"],
        "action_history": model.action_history,
        "seconds": time.monotonic() - started,
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
    print(f"Checkpoint: {output / 'best.pt'}", flush=True)
    return output
