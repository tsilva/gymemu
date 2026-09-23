"""UI-independent autoregressive playback and keyboard controls."""

from collections import deque

import numpy as np
import torch

from gymemu.data import frame_stack, pad_actions, validate_states

PLAYBACK_FPS = 30


class Player:
    def __init__(
        self,
        model,
        config,
        device,
        initial_history=None,
        initial_actions=None,
        *,
        start_states=None,
    ):
        self.model, self.config, self.device = model, config, device
        self.actions = config["action_values"]
        self.action_history = config.get("action_history", 1)
        if (
            type(self.action_history) is not int
            or not 1 <= self.action_history <= config["history"]
        ):
            raise ValueError("Action history must be between 1 and the RGB history length")
        self.start_states = list(start_states or [])
        self.start_index = 0
        self.history = deque(maxlen=config["history"])
        self.state_history = deque(maxlen=config["history"])
        self.past_actions = deque(maxlen=self.action_history - 1)
        # Validate all scenes before playback, including auxiliary state and actions.
        for _, history in self.start_states:
            self._set_initial_history(history)
        if self.start_states:
            initial_history = self.start_states[0][1]
            initial_actions = None
        self._set_initial_history(initial_history, initial_actions)
        self.reset(cycle=False)

    @property
    def start_name(self):
        return self.start_states[self.start_index][0] if self.start_states else None

    def _set_initial_history(self, initial_history, initial_actions=None):
        config = self.config
        if initial_actions is None:
            initial_actions = getattr(initial_history, "actions", None)
        self.initial_history = [frame.clone() for frame in (initial_history or [])]
        self.state_fields = tuple(config.get("state_fields", ()))
        self.initial_states = []
        if self.state_fields and self.initial_history:
            states = getattr(initial_history, "states", None)
            validate_states(states, len(self.initial_history), self.state_fields)
            self.initial_states = [state.clone() for state in states]
        required = min(self.action_history - 1, max(0, len(self.initial_history) - 1))
        initial_actions = [] if initial_actions is None else list(initial_actions)
        if not required <= len(initial_actions) <= max(0, len(self.initial_history) - 1):
            raise ValueError("Starting scene lacks aligned recorded action history")
        if any(action not in self.actions for action in initial_actions):
            raise ValueError("Starting scene contains an unknown executed action")
        self.initial_actions = (
            [self.actions.index(a) for a in initial_actions[-(self.action_history - 1) :]]
            if self.action_history > 1
            else []
        )
        if len(self.initial_history) > config["history"]:
            raise ValueError("Starting history is longer than the model history")
        for frame in self.initial_history:
            if tuple(frame.shape) != tuple(config["shape"]):
                raise ValueError("Starting frame dimensions differ from the model")
            if not torch.isfinite(frame).all() or frame.min() < 0 or frame.max() > 1:
                raise ValueError("Starting frames must contain finite pixels in [0, 1]")

    @torch.inference_mode()
    def _predict(self, action_index):
        stack = frame_stack(list(self.history), self.config["history"], self.config["shape"])
        self.input_stack = stack.clone()
        self.has_prediction = True
        self.last_action = self.actions[action_index] if action_index < len(self.actions) else None
        context = pad_actions(
            [*self.past_actions, action_index], self.action_history, len(self.actions)
        )
        action = torch.tensor(context, dtype=torch.long, device=self.device)
        action = action if self.action_history == 1 else action.unsqueeze(0)
        inputs = stack.unsqueeze(0).to(self.device)
        self.input_tokens = action
        self.input_states = None
        if self.state_fields:
            states = (
                frame_stack(
                    list(self.state_history), self.config["history"], (len(self.state_fields) + 1,)
                )
                .unsqueeze(0)
                .to(self.device)
            )
            self.input_states = states
            prediction, state = self.model.predict_step(inputs, action, states)
            result = prediction[0].cpu()
            self.state_history.append(state[0].cpu())
        else:
            result = self.model(inputs, action)[0].cpu()
        self.history.append(result)
        if action_index != len(self.actions):
            self.past_actions.append(action_index)
        return result

    def reset(self, *, cycle=False):
        self.continuous = False
        self.held_keys = []
        if cycle and self.start_states:
            self.start_index = (self.start_index + 1) % len(self.start_states)
            self._set_initial_history(self.start_states[self.start_index][1])
        self.history.clear()
        self.state_history.clear()
        self.state_history.extend(state.clone() for state in self.initial_states)
        self.past_actions.clear()
        self.past_actions.extend(self.initial_actions)
        self.history.extend(frame.clone() for frame in self.initial_history)
        self.input_stack = frame_stack(
            list(self.history), self.config["history"], self.config["shape"]
        )
        self.has_prediction = False
        self.last_action = None
        self.steps = 0
        # Recorded starts are displayed immediately; neither reset path runs inference.
        self.frame = self.history[-1] if self.history else None

    def advance(self, action):
        if action not in self.actions:
            raise ValueError(f"Unknown action {action}; choose from {self.actions}")
        if self.frame is None:
            self.frame = self._predict(len(self.actions))  # Empty history, no game action.
            return
        self.frame = self._predict(self.actions.index(action))
        self.steps += 1

    def pixels(self):
        if self.frame is None:
            channels, height, width = self.config["shape"]
            return np.zeros((height, width, channels), dtype=np.uint8)
        return self.frame.permute(1, 2, 0).mul(255).round().byte().numpy()

    def tick(self, keymap):
        """Execute at most one transition per paced playback tick."""
        if self.continuous:
            action = (
                keymap[self.held_keys[-1]]
                if self.held_keys
                else (0 if 0 in self.actions else self.actions[0])
            )
            self.advance(action)


def default_bindings(config):
    if "key_actions" not in config:
        return ["left=2", "right=1", "space=0"]  # Legacy Breakout checkpoints.
    bindings = config["key_actions"]
    if bindings:
        return [f"{key}={value}" for key, value in bindings.items()]
    keys = "1234567890abcdefghijklmnopqrstuvwxyz"
    keys = [key for key in keys if key not in "rc"]
    if len(config["action_values"]) > len(keys):
        raise ValueError("Too many actions for automatic bindings; supply --key-action")
    return [f"{key}={value}" for key, value in zip(keys, config["action_values"])]


def handle_key(player, key, keymap, *, down=True, repeat=False):
    if not down:
        if key in player.held_keys:
            player.held_keys.remove(key)
        return
    if repeat:
        return
    if key == "escape":
        player.continuous = False
        player.held_keys.clear()
    elif key == "r":
        player.reset()
    elif key == "c":
        player.reset(cycle=True)
    elif key == "tab":
        player.continuous = not player.continuous
        if getattr(player, "finished", False):
            player.continuous = False
    elif key in keymap and key not in player.held_keys:
        player.held_keys.append(key)
        if not player.continuous:
            player.advance(keymap[key])
