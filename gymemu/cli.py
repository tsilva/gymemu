"""Gymemu command-line application."""

import argparse
import importlib
import sys
from importlib.metadata import version

COMMANDS = {
    "play-state": ("Play state dynamics with a learned RGB decoder", "play_state"),
    "dynamics": ("Train and evaluate single-ball state-only dynamics", "dynamics"),
    "train": ("Train an emulator using Hydra configs or a recipe", "train"),
    "play": ("Play a trained emulator in the browser", "play"),
    "compare": ("Compare held-out metrics across runs", "compare"),
    "cache-frames": ("Build a lossless dataset frame cache", "cache_frames"),
    "save-start-state": ("Save a named playback start state", "save_start_state"),
    "upload-checkpoints": ("Upload run checkpoints to R2", "upload_checkpoints"),
    "sync": ("Publish a completed offline Run", "sync"),
    "experiment": ("Launch a resolved recipe on dstack", "experiment"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gymemu", description=__doc__, allow_abbrev=False)
    parser.add_argument("--version", action="version", version=version("gymemu"))
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for name, (description, _) in COMMANDS.items():
        subparsers.add_parser(name, help=description, add_help=False)
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help()
        return 0
    if arguments[0] in {"-h", "--help", "--version"}:
        parser.parse_args(arguments)
        return 0
    if arguments[0] not in COMMANDS:
        parser.error(f"unknown command: {arguments[0]}")
    module = importlib.import_module(f"gymemu.commands.{COMMANDS[arguments[0]][1]}")
    original = sys.argv
    try:
        sys.argv = [f"gymemu {arguments[0]}", *arguments[1:]]
        module.main()
    finally:
        sys.argv = original
    return 0
