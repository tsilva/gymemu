"""Play a learned emulator. One fresh action key press produces one predicted frame."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gymemu.checkpoints import load_model
from gymemu.data import frame_stack, pad_actions, validate_states
from gymemu.runtime import device_for, positive
from gymemu.scenes import (
    START_STATES,
    list_start_states,
    load_named_scene,
    load_scene,
    named_scene_path,
)


class Player:
    def __init__(self, model, config, device, initial_history=None, initial_actions=None):
        self.model, self.config, self.device = model, config, device
        self.actions = config["action_values"]
        self.action_history = config.get("action_history", 1)
        if (
            type(self.action_history) is not int
            or not 1 <= self.action_history <= config["history"]
        ):
            raise ValueError("Action history must be between 1 and the RGB history length")
        if initial_actions is None:
            initial_actions = getattr(initial_history, "actions", None)
        self.initial_history = [frame.clone() for frame in (initial_history or [])]
        self.state_fields = tuple(config.get("state_fields", ()))
        self.initial_states = []
        if self.state_fields and self.initial_history:
            states = getattr(initial_history, "states", None)
            validate_states(states, len(self.initial_history), self.state_fields)
            self.initial_states = [state.clone() for state in states]
        self.state_history = deque(maxlen=config["history"])
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
        self.past_actions = deque(maxlen=self.action_history - 1)
        if len(self.initial_history) > config["history"]:
            raise ValueError("Starting history is longer than the model history")
        for frame in self.initial_history:
            if tuple(frame.shape) != tuple(config["shape"]):
                raise ValueError("Starting frame dimensions differ from the model")
            if not torch.isfinite(frame).all() or frame.min() < 0 or frame.max() > 1:
                raise ValueError("Starting frames must contain finite pixels in [0, 1]")
        self.history = deque(maxlen=config["history"])
        self.steps = 0
        self.reset()

    @torch.inference_mode()
    def _predict(self, action_index):
        stack = frame_stack(list(self.history), self.config["history"], self.config["shape"])
        context = pad_actions(
            [*self.past_actions, action_index], self.action_history, len(self.actions)
        )
        action = torch.tensor(context, dtype=torch.long, device=self.device)
        action = action if self.action_history == 1 else action.unsqueeze(0)
        inputs = stack.unsqueeze(0).to(self.device)
        if self.state_fields:
            states = (
                frame_stack(
                    list(self.state_history), self.config["history"], (len(self.state_fields) + 1,)
                )
                .unsqueeze(0)
                .to(self.device)
            )
            prediction, state = self.model.predict_step(inputs, action, states)
            result = prediction[0].cpu()
            self.state_history.append(state[0].cpu())
        else:
            result = self.model(inputs, action)[0].cpu()
        self.history.append(result)
        if action_index != len(self.actions):
            self.past_actions.append(action_index)
        return result

    def reset(self):
        self.history.clear()
        self.state_history.clear()
        self.state_history.extend(state.clone() for state in self.initial_states)
        self.past_actions.clear()
        self.past_actions.extend(self.initial_actions)
        self.history.extend(frame.clone() for frame in self.initial_history)
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


def handle_event(player, event, keymap):
    import pygame

    if event.type == pygame.QUIT:
        return False
    if event.type != pygame.KEYDOWN or getattr(event, "repeat", False):
        return True
    if event.key == pygame.K_ESCAPE:
        return False
    if event.key == pygame.K_r:
        player.reset()
    elif event.key in keymap:
        player.advance(keymap[event.key])
    return True


def default_bindings(config):
    if "key_actions" not in config:
        return ["left=2", "right=1", "space=0"]  # Legacy Breakout checkpoints.
    bindings = config["key_actions"]
    if bindings:
        return [f"{key}={value}" for key, value in bindings.items()]
    keys = "1234567890abcdefghijklmnopqrstuvwxyz"
    keys = [key for key in keys if key != "r"]
    if len(config["action_values"]) > len(keys):
        raise ValueError("Too many actions for automatic bindings; supply --key-action")
    return [f"{key}={value}" for key, value in zip(keys, config["action_values"])]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--scale", type=positive, default=3)
    startup = parser.add_mutually_exclusive_group()
    startup.add_argument(
        "--start-state", help="Named snapshot for this game; use --list-start-states"
    )
    startup.add_argument(
        "--start-scene",
        type=Path,
        help="Recorded history; defaults to start-scene.npz beside the checkpoint",
    )
    startup.add_argument(
        "--empty-start",
        action="store_true",
        help="Test learned initialization from an empty history instead of a recorded scene",
    )
    parser.add_argument(
        "--list-start-states", action="store_true", help="List named snapshots and exit"
    )
    parser.add_argument(
        "--state-dir", type=Path, default=START_STATES, help="Snapshot library directory"
    )
    parser.add_argument(
        "--key-action",
        action="append",
        metavar="KEY=VALUE",
        help="Override checkpoint key bindings, e.g. --key-action left=2",
    )
    parser.add_argument("--headless-actions", help="Comma-separated action values for a smoke")
    parser.add_argument("--output", type=Path, default=Path("logs/play.png"))
    args = parser.parse_args(argv)
    scene_path = (
        None
        if args.empty_start
        else (args.start_scene or args.checkpoint.parent / "start-scene.npz")
    )
    if (
        not args.start_state
        and not args.list_start_states
        and scene_path is not None
        and not scene_path.is_file()
    ):
        parser.error(
            f"Recorded starting scene not found: {scene_path}. Place start-scene.npz beside "
            "the checkpoint, use --start-scene PATH, or use --empty-start to test initialization."
        )
    device = device_for(args.device)
    model, config = load_model(args.checkpoint, device)
    if args.list_start_states:
        for state in list_start_states(config, args.state_dir):
            print(f"{state['name']}: {state.get('description', '')}")
        return
    if args.start_state:
        try:
            scene_path = named_scene_path(args.start_state, config, args.state_dir)
        except ValueError as error:
            parser.error(str(error))
        if not scene_path.is_file():
            parser.error(f"Unknown start state {args.start_state!r}. Use --list-start-states.")
    initial_history = (
        load_named_scene(args.start_state, config, args.state_dir)
        if args.start_state
        else (load_scene(scene_path, config) if scene_path else None)
    )
    player = Player(model, config, device, initial_history)
    if scene_path:
        print(
            f"Recorded start: {scene_path.resolve()} ({len(initial_history)} frames). "
            "Each action key press now predicts one transition.",
            flush=True,
        )
    if args.headless_actions is not None:
        for value in args.headless_actions.split(","):
            if value.strip():
                player.advance(int(value))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(player.pixels()).save(args.output)
        print(
            f"Initialized={player.frame is not None}, action steps={player.steps}: "
            f"{args.output.resolve()}"
        )
        return

    import pygame

    pygame.init()
    try:
        pygame.key.set_repeat()  # A held key does not create inference steps.
        bindings = args.key_action or default_bindings(config)
        print("Key bindings: " + ", ".join(bindings), flush=True)
        keymap = {}
        for binding in bindings:
            key, value = binding.rsplit("=", 1)
            value = int(value)
            if value not in player.actions:
                parser.error(f"Action {value} absent from this checkpoint; set --key-action")
            code = pygame.key.key_code(key)
            if code in (pygame.K_r, pygame.K_ESCAPE):
                parser.error("R and Escape are reserved for reset and quit")
            keymap[code] = value
        _, height, width = config["shape"]
        size = (width * args.scale, height * args.scale)
        screen = pygame.display.set_mode((size[0], size[1] + 40))
        title = (
            f"gymemu — {args.start_state}"
            if args.start_state
            else (
                "gymemu — recorded scene" if scene_path else "gymemu — action-stepped CNN emulator"
            )
        )
        pygame.display.set_caption(title)
        font = pygame.font.Font(None, 22)
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if not handle_event(player, event, keymap):
                    running = False
                    break
            screen.fill((20, 20, 20))
            surface = pygame.surfarray.make_surface(np.swapaxes(player.pixels(), 0, 1))
            screen.blit(pygame.transform.scale(surface, size), (0, 40))
            label = (
                "Press an action key to generate the initial frame"
                if player.frame is None
                else f"Step {player.steps} | press action key | R reset | Esc quit"
            )
            screen.blit(font.render(label, True, (240, 240, 240)), (8, 12))
            pygame.display.flip()
            clock.tick(60)  # Refresh the window only; no inference outside action events.
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
