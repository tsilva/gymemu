"""Prepare, train, compare, evaluate, and replay single-ball state dynamics."""

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gymemu.config import CONFIG_DIR
from gymemu.state_data import prepare
from gymemu.state_training import evaluate_checkpoint, playback, search, train


def compose_state_config(path=None, overrides=()):
    path = Path(path or CONFIG_DIR / "state_dynamics.yaml").resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(path.parent)):
        cfg = compose(config_name=path.stem, overrides=list(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


def validate(config):
    model, tc, ec = config["model"], config["trainer"], config["evaluation"]
    if model["kind"] not in ("state_mlp", "state_gru"):
        raise ValueError("State dynamics requires state_mlp or state_gru")
    if model["history"] < 1 or model["past_actions"] < 0:
        raise ValueError("Invalid history lengths")
    for key in ("batch_size", "threads", "eval_every", "rollout_starts"):
        if tc[key] < 1:
            raise ValueError(f"trainer.{key} must be positive")
    if not ec["horizons"] or min(ec["horizons"]) < 1:
        raise ValueError("Evaluation horizons must be positive")
    if ec["starts"] < 1 or tc["epochs"] < 0 or tc["rollout_epochs"] < 0:
        raise ValueError("Invalid evaluation/training budget")
    if len(config["loss"]["motion_tolerances"]) != 6:
        raise ValueError("Six native-unit motion tolerances are required")
    if min(config["loss"]["motion_tolerances"]) <= 0:
        raise ValueError("Motion tolerances must be positive")
    if ec["split"] not in ("validation", "test"):
        raise ValueError("Evaluate only validation or test")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "search", "evaluate", "play"])
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "state_dynamics.yaml")
    parser.add_argument(
        "overrides", nargs="*", help="Strict dotted YAML overrides, e.g. model.history=4"
    )
    args = parser.parse_args(argv)
    config = compose_state_config(args.config, args.overrides)
    validate(config)
    if args.command == "prepare":
        root = Path(config["dataset"]).expanduser()
        if not root.is_dir():
            from huggingface_hub import snapshot_download

            root = Path(
                snapshot_download(
                    config["dataset"],
                    repo_type="dataset",
                    revision=config["revision"],
                    allow_patterns=["manifest.json", "episodes/**", "transitions/**"],
                )
            )
        result = prepare(root, config["cache"], config["data"])
        print(
            f"Prepared {result['train']['transitions']} train, "
            f"{result['validation']['transitions']} validation, "
            f"{result['test']['transitions']} test transitions"
        )
    elif args.command == "train":
        train(config, initialize=config["checkpoint"])
    elif args.command == "search":
        search(config)
    else:
        if not config["checkpoint"]:
            parser.error("checkpoint=/path/to/state-checkpoint.pt is required")
        destination = Path(config["output"])
        destination.mkdir(parents=True, exist_ok=False)
        if args.command == "evaluate":
            result = evaluate_checkpoint(
                config["checkpoint"], config["cache"], config, destination / "evaluation.json"
            )
            print(f"{result['split']} selection score: {result['selection_score']:.5f}")
        else:
            result = playback(
                config["checkpoint"], config["cache"], config, destination / "playback.json"
            )
            print(f"Played {len(result['steps'])} predictions: {result['stop_reason']}")


if __name__ == "__main__":
    main()
