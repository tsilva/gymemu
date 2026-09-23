"""Versioned training recovery, independent of portable inference checkpoints."""

import copy
import os
import random
import signal
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Sampler


class TrainingInterrupted(Exception):
    """A requested stop was handled after a complete optimizer update."""


class EpochSampler(Sampler):
    """Reconstruct an epoch's order without consuming model RNG or prefetched cursors."""

    def __init__(self, dataset, seed):
        self.size, self.seed = len(dataset), seed
        self.epoch = self.offset = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist()[self.offset :])

    def __len__(self):
        return self.size - self.offset


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_state(v) for v in value)
    return copy.deepcopy(value)


def rng_state(device):
    numpy = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": (numpy[0], numpy[1].tolist(), *numpy[2:]),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def save_training(path, state):
    """Flush and atomically replace: an interrupted write retains the prior checkpoint."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(cpu_state({"training_state_version": 1, **state}), stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def load_training(path):
    saved = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if saved.get("training_state_version") != 1:
        raise ValueError(
            "Not a resumable training checkpoint; use resume.pt, not inference weights"
        )
    required = {
        "config",
        "state_dict",
        "training_config",
        "optimizer",
        "rng",
        "progress",
        "stage_index",
        "epoch",
        "epoch_complete",
        "total_updates",
        "total_train_samples",
        "best",
        "best_bundle",
        "best_rgb",
        "device_type",
        "torch_version",
        "elapsed_seconds",
    }
    if not required <= saved.keys():
        raise ValueError("Incomplete training checkpoint")
    return saved


def training_contract(config):
    """Only routing and loader execution settings may change during a continuation."""
    contract = copy.deepcopy(config)
    for key in ("output", "resume", "run_id", "launch_overrides", "wandb", "r2", "hydra"):
        contract.pop(key, None)
    for key in ("frame_cache", "workers", "threads", "checkpoint_seconds"):
        contract["trainer"].pop(key, None)
    return contract


def resume_config(path, overrides=()):
    saved = load_training(path)
    config = OmegaConf.create(saved["training_config"])
    config.resume = str(Path(path).expanduser().resolve())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    config.output = str(Path("runs") / f"resume-{stamp}")
    if any("=" not in arg or arg.startswith("-") for arg in overrides):
        raise ValueError("Resume accepts key=value overrides, not --recipe or Hydra flags")
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    config.resume = str(Path(path).expanduser().resolve())
    if training_contract(OmegaConf.to_container(config, resolve=True)) != training_contract(
        saved["training_config"]
    ):
        raise ValueError("Resume cannot change the model, dataset, optimizer, or training recipe")
    return config


@contextmanager
def stop_signals():
    """Defer SIGINT/SIGTERM to a safe optimizer boundary; leave SIGKILL to the OS."""
    requested = []
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, lambda number, frame: requested.append(number))
    try:
        yield lambda: bool(requested)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
