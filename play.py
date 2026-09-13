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
    def __init__(self, model, config, device):
        self.model, self.config, self.device = model, config, device
        self.actions = config["action_values"]
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
        self.steps = 0
        self.frame = None  # The first action-key press generates the initial frame.

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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--scale", type=positive, default=3)
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
    player = Player(model, config, device)
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
        pygame.display.set_caption("gymemu — action-stepped CNN emulator")
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
