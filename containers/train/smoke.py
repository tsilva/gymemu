"""Bounded training, reload and playback proof using the installed container runtime."""

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from gymemu.checkpoints import load_model
from gymemu.config import compose_config
from gymemu.engine import train
from gymemu.scenes import load_scene
from play import Player


def fixture(root):
    def write(kind, split, rows):
        path = root / kind / split / "00000.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path)

    frames = []
    for frame_id in (60, 20, 40, 10, 50, 30):
        buffer = io.BytesIO()
        pixels = np.full((21, 17, 3), frame_id, dtype=np.uint8)
        Image.fromarray(pixels).save(buffer, format="PNG")
        frames.append({"frame_id": frame_id, "image": {"bytes": buffer.getvalue(), "path": None}})
    write("frames", "assets", frames)
    for split, eid, ids in (("train", 1, [10, 20, 30]), ("heldout", 2, [40, 50, 60])):
        write("episodes", split, [{"episode_id": eid, "initial_frame_id": ids[0], "length": 2}])
        transitions = []
        for step, action in enumerate((0, 2)):
            labels = [
                "dict",
                [
                    [name, ["scalar", value]]
                    for name, value in (
                        ("ball_x_normalized", (step + 1) / 10),
                        ("ball_y_normalized", eid / 10),
                    )
                ],
            ]
            transitions.append(
                {
                    "episode_id": eid,
                    "step": step,
                    "source_frame_id": ids[step],
                    "successor_frame_id": ids[step + 1],
                    "native_action_json": str(action),
                    "record_json": json.dumps(
                        {"structure": json.dumps(["dict", [["labels", labels]]])}
                    ),
                }
            )
        write("transitions", split, transitions[::-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--online", action="store_true", help="Also verify configured W&B and R2")
    args = parser.parse_args()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU smoke requires an allocated CUDA GPU")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("GPU smoke requires native bf16 support")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or Path("verification") / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = output / "dataset"
    fixture(data)
    from gymemu.cache import build_cache

    cache = build_cache(data, output / "cache", workers=1)
    records = []
    for recipe in ("direct", "latent", "breakout_ball"):
        cfg = compose_config([f"recipe={recipe}", "game=custom", "experiment=smoke"])
        cfg.game.dataset = str(data)
        cfg.game.env_id = "Gymemu-ContainerSmoke-v0"
        cfg.output = str(output / recipe)
        cfg.trainer.device = args.device
        cfg.trainer.precision = "bf16" if args.device == "cuda" else "fp32"
        cfg.trainer.compile = args.device == "cuda"
        cfg.trainer.frame_cache = str(cache)
        cfg.trainer.loader = "cached"
        cfg.wandb.mode = "online" if args.online else "disabled"
        cfg.r2.enabled = args.online
        trained = train(cfg)
        model, config = load_model(trained / "best.pt", torch.device(args.device))
        scene = load_scene(trained / "start-scene.npz", config)
        for initial in (scene, None):
            player = Player(model, config, torch.device(args.device), initial)
            for action in (0, 2, 0):
                player.advance(action)
            assert player.steps == (3 if initial is not None else 2)
            assert player.pixels().shape == (21, 17, 3)
            player.reset()
            assert player.steps == 0
        records.append({"recipe": recipe, "output": str(trained), "status": "passed"})
    receipt = {
        "device": args.device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "online": args.online,
        "checks": records,
    }
    (output / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
