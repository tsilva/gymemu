"""Compose explicit-state dynamics and an RGB decoder for interactive rollout."""

import json
import math
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from gymemu.models import build_model
from gymemu.player import Player


def load_component(repository, revision, kind, device):
    """Load data-only Hub artifacts into a registered local model implementation."""
    folder = Path(repository).expanduser()
    if not folder.is_dir():
        folder = Path(
            snapshot_download(
                repository,
                revision=revision,
                allow_patterns=["config.json", "pytorch_model.bin", "example_source.json"],
            )
        )
    config = json.loads((folder / "config.json").read_text())
    fields = (
        ("vocabulary", "width", "blocks")
        if kind == "unified_state_mlp"
        else ("palette", "width", "layers", "hud_height")
    )
    if config.get("kind", kind) != kind:
        raise ValueError(f"Expected {kind} artifacts")
    model = build_model({"kind": kind, **{key: config[key] for key in fields}})
    model.load_state_dict(
        torch.load(
            folder / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
    )
    return model.to(device).eval().requires_grad_(False), folder


def next_source(prediction, action):
    """Preserve every generated field, including fractional y and collision memory."""
    if prediction.shape != (1, 118) or not torch.isfinite(prediction).all():
        raise ValueError("Expected one finite 118-value prediction")
    if prediction[0, 117] != 0:
        raise ValueError("Terminal placeholders cannot be fed back")
    source = prediction.new_zeros((1, 119))
    source[:, 0] = prediction[:, 0]
    source[:, 1] = prediction[:, 1].floor()
    source[:, 2:4] = prediction[:, 2:4]
    source[:, 4] = prediction[:, 116]
    source[:, 5] = prediction[:, 114]
    source[:, 6] = prediction[:, 115]
    source[:, 7] = prediction[:, 113]
    source[:, 8] = (prediction[:, 1].remainder(1) * 8).round()
    source[:, 9] = prediction[:, 112]
    source[:, 10:118] = prediction[:, 4:112]
    source[:, 118] = action
    return source


def load_source(path):
    value = json.loads(Path(path).read_text())
    return torch.tensor(value["source"] if isinstance(value, dict) else value, dtype=torch.float32)


class StatePlayback:
    """Browser player protocol with state feedback; RGB is output only."""

    mode = "autoregressive"
    history_editable = False
    context_note = "Generated states feed the next prediction. Decoded RGB is display output only."
    pixels = Player.pixels

    def __init__(self, dynamics, decoder, source, config, device="cpu", name="Starting state"):
        self.model = dynamics.eval().requires_grad_(False)
        self.decoder = decoder.eval().requires_grad_(False)
        self.device = device
        self.config = {"history": 1, "shape": [3, 210, 160], **config}
        self.actions = config["action_values"]
        self.playback_fps = config["playback_fps"]
        if self.actions != [0, 1, 2]:
            raise ValueError("Unified dynamics requires provider actions [0, 1, 2]")
        if not math.isfinite(self.playback_fps) or self.playback_fps <= 0:
            raise ValueError("Playback FPS must be finite and positive")
        self.start_states = [(name, source.clone())]
        self.start_index = 0
        self._set_initial_history(source)
        self.reset()

    @property
    def start_name(self):
        return self.start_states[self.start_index][0]

    def _set_initial_history(self, source):
        source = source.to(device=self.device, dtype=torch.float32).reshape(1, -1)
        if source.shape != (1, 119) or not torch.isfinite(source).all():
            raise ValueError("Starting state must contain 119 finite source values")
        if source[0, 118].item() not in self.actions:
            raise ValueError("Invalid starting action")
        if not torch.all((source[:, 10:118] == 0) | (source[:, 10:118] == 1)):
            raise ValueError("Starting brick occupancy must be binary")
        if source[0, 5].item() not in (12, 16):
            raise ValueError("Starting paddle width must be 12 or 16")
        if source[0, 8] != source[0, 8].round() or not 0 <= source[0, 8] < 8:
            raise ValueError("Starting fractional y must be integer eighths in 0..7")
        if source[0, 1] != source[0, 1].floor() or not 0 < source[0, 1] < 208:
            raise ValueError("Starting state must describe an active ball with integer RAM y")
        self.initial_source = source.clone()

    @torch.inference_mode()
    def reset(self, *, cycle=False):
        self.source = self.initial_source.clone()
        self.finished = False
        self.continuous = False
        self.held_keys = []
        self.steps = 0
        self.last_action = None
        self.has_prediction = False
        visual = torch.cat([self.source[:, :2], self.source[:, 4:6], self.source[:, 10:118]], 1)
        self.frame = self.decoder.render(visual)[0].float().cpu()
        self.input_stack = self.frame[None].clone()

    @torch.inference_mode()
    def advance(self, action):
        if type(action) is not int or action not in self.actions:
            raise ValueError(f"Unknown action {action}; choose from {self.actions}")
        if self.finished:
            self.continuous = False
            return
        source = self.source.clone()
        source[:, 118] = action
        prediction = self.model.predict(source)
        if prediction.shape != (1, 118) or not torch.isfinite(prediction).all():
            raise ValueError("Dynamics returned an invalid state")
        if prediction[0, 117] not in (0, 1):
            raise ValueError("Dynamics returned an invalid terminal flag")
        finished = bool(prediction[0, 117])
        if not finished:
            following = next_source(prediction, action)
            frame = self.decoder.render(self.decoder.from_dynamics(prediction))[0].float().cpu()
        self.input_stack = self.frame[None].clone()
        self.last_action = action
        self.steps += 1
        self.has_prediction = True
        self.finished = finished
        if finished:
            self.continuous = False
            self.held_keys.clear()
        else:
            self.source = following
            self.frame = frame
