"""Open the Gymemu browser player. Tab toggles single-step and continuous playback."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from gymemu.checkpoints import load_model
from gymemu.player import Player, default_bindings
from gymemu.replay import COMPARISON_LABELS, load_replay
from gymemu.runtime import device_for, positive
from gymemu.scenes import (
    START_STATES,
    list_start_states,
    load_named_scene,
    load_scene,
    named_scene_path,
)


def recorded_player(args, model, config, device, scene_path):
    if args.start_state:
        scene_path = named_scene_path(args.start_state, config, args.state_dir)
        if not scene_path.is_file():
            raise ValueError(f"Unknown start state {args.start_state!r}. Use --list-start-states.")
    initial_history = (
        load_named_scene(args.start_state, config, args.state_dir)
        if args.start_state
        else (load_scene(scene_path, config) if scene_path else None)
    )
    start_states = [
        (
            args.start_state or ("recorded scene" if scene_path else "empty start"),
            initial_history,
        )
    ]
    try:
        available_states = list_start_states(config, args.state_dir)
    except (ValueError, OSError) as error:
        print(f"Named start states unavailable: {error}", flush=True)
        available_states = []
    for state in available_states:
        if state["name"] == args.start_state:
            continue
        path = named_scene_path(state["name"], config, args.state_dir)
        if scene_path and path.resolve() == scene_path.resolve():
            continue
        try:
            history = load_named_scene(state["name"], config, args.state_dir)
        except (ValueError, OSError) as error:
            print(f"Skipping start state {state['name']}: {error}", flush=True)
            continue
        start_states.append((state["name"], history))
    print("Reset cycle: " + " -> ".join(name for name, _ in start_states), flush=True)
    player = Player(model, config, device, initial_history, start_states=start_states)
    if scene_path:
        print(
            f"Recorded start: {scene_path.resolve()} ({len(initial_history)} frames). "
            "Each action key press now predicts one transition.",
            flush=True,
        )
    return player


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--scale",
        type=positive,
        default=3,
        help="Accepted for compatibility; resize dashboard widgets to change frame size",
    )
    parser.add_argument("--port", default="auto", help="Local browser port, or auto")
    parser.add_argument("--no-browser", action="store_true", help="Print URL without opening it")
    startup = parser.add_mutually_exclusive_group()
    startup.add_argument(
        "--teacher-forcing",
        action="store_true",
        help="Replay dataset episodes with recorded inputs and side-by-side targets",
    )
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
    parser.add_argument("--dataset", help="Replay dataset path or Hub ID; defaults to checkpoint")
    parser.add_argument("--revision", help="Replay dataset revision; defaults to saved revision")
    parser.add_argument(
        "--split", help="Replay split; defaults to checkpoint eval split or heldout"
    )
    parser.add_argument("--episode-id", type=int, help="Start replay at this recorded episode ID")
    parser.add_argument(
        "--headless-steps", type=positive, help="Replay N transitions and save a PNG"
    )
    parser.add_argument("--headless-actions", help="Comma-separated action values for a smoke")
    parser.add_argument("--output", type=Path, default=Path("logs/play.png"))
    args = parser.parse_args(argv)
    if not args.teacher_forcing and any(
        value is not None
        for value in (args.dataset, args.revision, args.split, args.episode_id, args.headless_steps)
    ):
        parser.error("Dataset replay options require --teacher-forcing")
    if args.teacher_forcing and (
        args.headless_actions is not None or args.key_action or args.list_start_states
    ):
        parser.error(
            "Teacher forcing uses recorded actions; scene listing and action overrides "
            "are unavailable. Use --headless-steps for a replay smoke."
        )
    scene_path = (
        None
        if args.empty_start or args.teacher_forcing
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
    if args.teacher_forcing:
        try:
            player = load_replay(
                model,
                config,
                device,
                dataset=args.dataset,
                revision=args.revision,
                split=args.split,
                episode_id=args.episode_id,
            )
        except (ValueError, OSError) as error:
            parser.error(str(error))
    else:
        try:
            player = recorded_player(args, model, config, device, scene_path)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    if args.headless_steps is not None:
        for _ in range(args.headless_steps):
            player.advance()
            if player.finished:
                break
        args.output.parent.mkdir(parents=True, exist_ok=True)
        from PIL import ImageDraw

        pixels = Image.fromarray(player.pixels())
        result = Image.new("RGB", (pixels.width, pixels.height + 36), (20, 20, 20))
        result.paste(pixels, (0, 36))
        draw = ImageDraw.Draw(result)
        panel_width = pixels.width // len(COMPARISON_LABELS)
        for index, label in enumerate(COMPARISON_LABELS):
            draw.text((index * panel_width + 2, 2), label, fill="white")
        draw.text((2 * panel_width + 2, 18), "Gray=0; brighter +; darker -", fill="white")
        result.save(args.output)
        print(f"{player.start_name} | {player.status()} | {player.comparison_label()}")
        print(f"Saved comparison: {args.output.resolve()}")
        return
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

    from gymemu.web_player import serve

    bindings = args.key_action or default_bindings(config)
    keymap = {}
    for binding in bindings:
        key, value = binding.rsplit("=", 1)
        key = key.lower()
        if key in ("r", "c", "tab", "escape"):
            parser.error("R, C, Tab, and Escape are reserved player controls")
        if int(value) not in config["action_values"]:
            parser.error(f"Action {value} absent from this checkpoint")
        keymap[key] = int(value)
    try:
        port = 0 if args.port == "auto" else int(args.port)
        if not 0 <= port <= 65535:
            raise ValueError("Port must be between 0 and 65535, or auto")
    except ValueError as error:
        parser.error(str(error))

    def mode_factory(mode):
        if mode == "teacher-forcing":
            return load_replay(
                model,
                config,
                device,
                dataset=args.dataset,
                revision=args.revision,
                split=args.split,
                episode_id=args.episode_id,
            )
        path = (
            None
            if args.empty_start
            else (args.start_scene or args.checkpoint.parent / "start-scene.npz")
        )
        return recorded_player(args, model, config, device, path)

    serve(
        player,
        checkpoint=args.checkpoint,
        keymap=keymap,
        port=port,
        open_browser=not args.no_browser,
        mode_factory=mode_factory,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
