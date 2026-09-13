"""Portable recorded histories used only to initialize playback."""

import json
from pathlib import Path

import numpy as np
import torch


class SceneHistory(list):
    """List-compatible RGB history with the executed native actions between its frames."""

    def __init__(self, frames, actions):
        super().__init__(frames)
        self.actions = actions


def load_scene(path: Path, config: dict) -> SceneHistory:
    with np.load(path, allow_pickle=False) as archive:
        frames = archive["frames"]
        actions = archive["actions"] if "actions" in archive.files else None
    if frames.dtype != np.uint8 or frames.ndim != 4:
        raise ValueError("Scene frames must be uint8 [history, channels, height, width]")
    if tuple(frames.shape[1:]) != tuple(config["shape"]):
        raise ValueError("Scene dimensions differ from the model")
    if not 1 <= len(frames) <= config["history"]:
        raise ValueError("Scene must contain between one frame and the model history length")
    required = min(config.get("action_history", 1) - 1, len(frames) - 1)
    if actions is None:
        if required > 0:
            raise ValueError("Scene lacks recorded action history required by this model")
        actions = np.empty(0, dtype=np.int64)
    if actions.ndim != 1 or not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("Scene actions must be a one-dimensional integer array")
    if not required <= len(actions) <= len(frames) - 1:
        raise ValueError("Scene action history does not match its frame history")
    if any(int(action) not in config["action_values"] for action in actions):
        raise ValueError("Scene contains an unknown executed action")
    return SceneHistory(
        torch.from_numpy(frames.copy()).float().div_(255).unbind(0), actions.tolist()
    )


def write_scene(output, game, metadata, frames, splits):
    if game["start_scene"]:
        scene = Path(game["start_scene"]).expanduser()
        load_scene(scene, metadata)
        output.write_bytes(scene.read_bytes())
        return
    selection = game["start"]
    split = selection["split"] or game["train_split"]
    if split not in splits:
        raise ValueError("Starting scene split must be the configured training or evaluation split")
    candidates = splits[split]
    episode_id = selection["episode_id"]
    episode = next(
        (e for e in candidates if episode_id is None or e.episode_id == episode_id), None
    )
    if episode is None:
        raise ValueError(f"Starting scene episode {episode_id} absent from {split}")
    position = selection["frame_position"]
    if type(position) is not int or not 0 <= position < len(episode.frames):
        raise ValueError("Starting scene frame_position must be within the selected episode")
    ids = episode.frames[max(0, position + 1 - metadata["history"]) : position + 1]
    pixels = torch.stack([frames.get(int(i)) for i in ids]).numpy()
    info = {
        "dataset": metadata["dataset"],
        "split": split,
        "episode_id": int(episode.episode_id),
        "frame_position": position,
        "frame_ids": ids.tolist(),
        "order": "oldest_to_newest",
    }
    # Native actions linking each pair of recorded history frames; no future action.
    actions = episode.actions[position + 1 - len(ids) : position]
    info["action_order"] = "between_frames_oldest_to_newest"
    np.savez_compressed(output, frames=pixels, actions=actions, metadata=json.dumps(info))
