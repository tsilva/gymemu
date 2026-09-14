#!/bin/sh
set -eu
mkdir -p "$GYMEMU_WORKSPACE" "$WANDB_DIR" "$HF_HOME" "$XDG_CACHE_HOME"
cd "$GYMEMU_WORKSPACE"
exec python /opt/gymemu/containers/train/launch.py "$@"
