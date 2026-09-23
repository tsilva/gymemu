"""Play the unified state dynamics through its learned RGB decoder."""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

from gymemu.config import CONFIG_DIR
from gymemu.runtime import device_for, positive
from gymemu.state_playback import StatePlayback, load_component, load_source
from gymemu.state_replay import load_state_episode, load_state_replay
from gymemu.web_player import serve


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "state_playback.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--teacher-forcing",
        dest="teacher_forcing",
        action="store_true",
        default=None,
        help="Use recorded state and actions on every step (default)",
    )
    mode.add_argument("--autoregressive", dest="teacher_forcing", action="store_false")
    parser.add_argument("--dataset", help="Hub dataset ID or local complete snapshot")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--episode-id", type=int, help="Episode in the selected split")
    parser.add_argument("--headless-steps", type=positive, help="Replay N recorded transitions")
    parser.add_argument("--start-source", type=Path, help="JSON with one complete 119-value source")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--port", default="auto")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--headless-actions", help="Comma-separated provider actions")
    parser.add_argument("--output", type=Path, default=Path("logs/state-play.png"))
    args = parser.parse_args(argv)
    if args.teacher_forcing is None:
        args.teacher_forcing = args.headless_actions is None and args.start_source is None
    if args.teacher_forcing and (args.headless_actions is not None or args.start_source):
        parser.error("Action overrides and --start-source require --autoregressive")
    if not args.teacher_forcing and args.headless_steps is not None:
        parser.error("--headless-steps requires --teacher-forcing")
    try:
        port = 0 if args.port == "auto" else int(args.port)
        if not 0 <= port <= 65535:
            raise ValueError("Port must be between 0 and 65535, or auto")
        torch.set_num_threads(2)
        config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        device = device_for(args.device)
        dynamics, _ = load_component(**config["dynamics"], kind="unified_state_mlp", device=device)
        decoder, _ = load_component(**config["decoder"], kind="state_renderer", device=device)
        replay = None
        recorded_episode = None

        def mode_factory(mode):
            nonlocal replay, recorded_episode
            if mode == "teacher-forcing":
                if replay is None:
                    replay = load_state_replay(
                        dynamics,
                        decoder,
                        config,
                        device,
                        dataset=args.dataset,
                        split=args.split,
                        episode_id=args.episode_id,
                    )
                return replay
            if mode != "autoregressive":
                raise ValueError("Unknown playback mode")
            if args.start_source is not None:
                source = load_source(args.start_source)
                name = str(args.start_source)
            else:
                if replay is not None:
                    recorded_episode = replay.episode
                elif recorded_episode is None:
                    recorded_episode = load_state_episode(
                        config, dataset=args.dataset, split=args.split, episode_id=args.episode_id
                    )
                source = recorded_episode.sources[:1]
                name = f"Recorded start | episode {recorded_episode.episode_id}"
            return StatePlayback(dynamics, decoder, source, config, device, name=name)

        player = mode_factory("teacher-forcing" if args.teacher_forcing else "autoregressive")
        print(
            f"Start: {player.start_name}. Space steps; Tab toggles continuous play; R resets.",
            flush=True,
        )
        if args.headless_actions is not None or args.headless_steps is not None:
            actions = (
                range(args.headless_steps)
                if args.headless_steps is not None
                else [int(a) for a in args.headless_actions.split(",")]
            )
            for action in actions:
                player.advance(None if args.teacher_forcing else action)
                if player.finished:
                    break
            args.output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(player.pixels()).save(args.output)
            print(f"Steps={player.steps}, finished={player.finished}: {args.output.resolve()}")
            if args.teacher_forcing:
                print(player.context_note)
            return
        serve(
            player,
            checkpoint=config["dynamics"]["repository"],
            keymap=config["key_actions"],
            port=port,
            open_browser=not args.no_browser,
            mode_factory=mode_factory,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
