"""Play a learned emulator. One fresh action key press produces one predicted frame."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train import device_for, frame_stack, load_model, positive


class Player:
    def __init__(self, model, config, device, initial_history=None):
        self.model, self.config, self.device = model, config, device
        self.actions = config["action_values"]
        self.initial_history = [frame.clone() for frame in (initial_history or [])]
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
        result = self.model(
            stack.unsqueeze(0).to(self.device), torch.tensor([action_index], device=self.device)
        )[0].cpu()
        self.history.append(result)
        return result

    def reset(self):
        self.history.clear()
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


def load_scene(path: Path, config: dict) -> list[torch.Tensor]:
    """Load chronological RGB history from a portable, non-pickled NumPy archive."""
    with np.load(path, allow_pickle=False) as archive:
        frames = archive["frames"]
    if frames.dtype != np.uint8 or frames.ndim != 4:
        raise ValueError("Scene frames must be uint8 [history, channels, height, width]")
    if tuple(frames.shape[1:]) != tuple(config["shape"]):
        raise ValueError("Scene dimensions differ from the model")
    if not 1 <= len(frames) <= config["history"]:
        raise ValueError("Scene must contain between one frame and the model history length")
    return list(torch.from_numpy(frames.copy()).float().div_(255).unbind(0))


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--scale", type=positive, default=3)
    parser.add_argument(
        "--start-scene",
        type=Path,
        help="Start from recorded RGB history in an NPZ scene; R restores it",
    )
    parser.add_argument(
        "--key-action",
        action="append",
        metavar="KEY=VALUE",
        help="Replace Breakout defaults, e.g. --key-action left=2",
    )
    parser.add_argument("--headless-actions", help="Comma-separated action values for a smoke")
    parser.add_argument("--output", type=Path, default=Path("logs/play.png"))
    args = parser.parse_args(argv)
    device = device_for(args.device)
    model, config = load_model(args.checkpoint, device)
    initial_history = load_scene(args.start_scene, config) if args.start_scene else None
    player = Player(model, config, device, initial_history)
    if args.start_scene:
        print(
            f"Recorded start: {args.start_scene.resolve()} ({len(initial_history)} frames). "
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
        bindings = args.key_action or ["left=2", "right=1", "space=0"]
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
            "gymemu — recorded scene"
            if args.start_scene
            else "gymemu — action-stepped CNN emulator"
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
