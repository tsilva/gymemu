"""Save a named recorded frame/action snapshot for repeatable playback debugging."""

import argparse
import copy
import tempfile
from pathlib import Path

import torch

from gymemu.checkpoints import load_model
from gymemu.data import Frames, read_episodes, resolve_dataset
from gymemu.scenes import START_STATES, save_start_state, write_scene


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--state-dir", type=Path, default=START_STATES)
    parser.add_argument("--scene", type=Path, help="Import an existing frame/action NPZ snapshot")
    parser.add_argument("--episode-id", type=int, help="Recorded episode to export")
    parser.add_argument(
        "--frame-position", type=int, help="Final frame position in the exported stack"
    )
    parser.add_argument("--split", help="Defaults to the checkpoint's held-out split")
    parser.add_argument(
        "--dataset", help="Local snapshot path or Hub ID; defaults to checkpoint dataset"
    )
    parser.add_argument(
        "--frame-cache", type=Path, help="Optional verified cache for dataset export"
    )
    args = parser.parse_args(argv)
    dataset_options = (
        args.episode_id,
        args.frame_position,
        args.split,
        args.dataset,
        args.frame_cache,
    )
    if args.scene and any(value is not None for value in dataset_options):
        parser.error("Use either --scene or dataset selection options")
    if not args.scene and (args.episode_id is None or args.frame_position is None):
        parser.error("Provide --scene or both --episode-id and --frame-position")
    _, config = load_model(args.checkpoint, torch.device("cpu"))
    with tempfile.TemporaryDirectory(prefix="gymemu-state-") as temporary:
        source = args.scene
        if source is None:
            game = copy.deepcopy(config["game"])
            root, provenance = resolve_dataset(args.dataset or game["dataset"], game["revision"])
            frames = Frames(root, compact=True, cache=args.frame_cache)
            if list(frames.shape) != list(config["shape"]):
                parser.error("Dataset RGB geometry differs from checkpoint")
            split = args.split or game["eval_split"]
            game["start_scene"] = None
            game["start"] = {
                "split": split,
                "episode_id": args.episode_id,
                "frame_position": args.frame_position,
            }
            source = Path(temporary) / "scene.npz"
            write_scene(
                source,
                game,
                {**config, "dataset": provenance},
                frames,
                {split: read_episodes(root, split)},
            )
        path = save_start_state(source, args.name, args.description, config, args.state_dir)
    print(f"Saved {path.resolve()}")
    print(f"Play with --start-state {args.name}")


if __name__ == "__main__":
    main()
