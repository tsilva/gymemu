import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gymemu.checkpoints import load_model
from gymemu.config import compose_config
from gymemu.engine import train
from gymemu.scenes import (
    START_STATES,
    list_start_states,
    load_named_scene,
    load_scene,
    named_scene_path,
    save_start_state,
)
from play import Player
from play import main as play_main
from save_start_state import main as save_main


def contract():
    return {
        "game": {"name": "custom"},
        "history": 4,
        "action_history": 4,
        "shape": [3, 21, 17],
        "action_values": [0, 2],
    }


def scene(tmp_path):
    path = tmp_path / "source.npz"
    frames = np.arange(4 * 3 * 21 * 17, dtype=np.uint8).reshape(4, 3, 21, 17)
    np.savez_compressed(
        path,
        frames=frames,
        actions=np.array([2, 0, 2]),
        metadata=json.dumps(
            {"episode_id": 5, "frame_position": 45, "dataset": {"revision": "pinned"}}
        ),
    )
    return path


def test_named_snapshot_roundtrip_provenance_and_reset(tmp_path):
    source = scene(tmp_path)
    config = contract()
    path = save_start_state(source, "ball-up", "Upward ball", config, tmp_path / "library")
    assert path == tmp_path / "library/custom/ball-up.npz"
    info = list_start_states(config, tmp_path / "library")[0]
    assert info["name"] == "ball-up" and info["description"] == "Upward ball"
    assert info["episode_id"] == 5 and info["dataset"]["revision"] == "pinned"
    assert info["history"] == 4 and info["shape"] == [3, 21, 17]
    loaded = load_named_scene("ball-up", config, tmp_path / "library")
    original = load_scene(source, config)
    assert all(torch.equal(a, b) for a, b in zip(loaded, original, strict=True))
    assert loaded.actions == original.actions == [2, 0, 2]
    player = Player(None, config, torch.device("cpu"), loaded)
    player.past_actions.clear()
    player.history.clear()
    player.reset()
    assert torch.equal(player.frame, loaded[-1]) and list(player.past_actions) == [1, 0, 1]
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_start_state(source, "ball-up", "different", config, tmp_path / "library")
    assert path.read_bytes() == before


@pytest.mark.parametrize("name", ["../escape", "nested/name", "", "/absolute", "a" * 65])
def test_names_cannot_escape_library(name, tmp_path):
    with pytest.raises(ValueError, match="game/name"):
        named_scene_path(name, contract(), tmp_path)


def test_game_metadata_and_input_contract_checked(tmp_path):
    source = scene(tmp_path)
    path = save_start_state(source, "state", "", contract(), tmp_path / "library")
    with pytest.raises(ValueError, match="model history"):
        load_named_scene("state", {**contract(), "history": 2}, tmp_path / "library")
    with np.load(path, allow_pickle=False) as archive:
        frames, actions = archive["frames"], archive["actions"]
        info = json.loads(str(archive["metadata"]))
    info["game"] = "other"
    np.savez_compressed(path, frames=frames, actions=actions, metadata=json.dumps(info))
    with pytest.raises(ValueError, match="name/game"):
        load_named_scene("state", contract(), tmp_path / "library")


def test_cli_export_import_list_and_play(snapshot, tmp_path, capsys):
    cfg = compose_config(["recipe=breakout_actions", "game=custom", "experiment=smoke"])
    cfg.game.dataset = str(snapshot)
    cfg.output = str(tmp_path / "run")
    output = train(cfg)
    checkpoint, library = str(output / "best.pt"), str(tmp_path / "library")
    save_main(
        [
            checkpoint,
            "--name",
            "heldout-end",
            "--description",
            "Test export",
            "--episode-id",
            "2",
            "--frame-position",
            "2",
            "--state-dir",
            library,
        ]
    )
    model, config = load_model(Path(checkpoint), torch.device("cpu"))
    loaded = load_named_scene("heldout-end", config, library)
    assert loaded.actions == [2]
    assert torch.allclose(loaded[-1], torch.full((3, 21, 17), 60 / 255))
    player = Player(model, config, torch.device("cpu"), loaded)
    player.advance(0)
    assert player.steps == 1
    play_main([checkpoint, "--list-start-states", "--state-dir", library, "--device", "cpu"])
    assert "heldout-end: Test export" in capsys.readouterr().out
    target = tmp_path / "play.png"
    play_main(
        [
            checkpoint,
            "--start-state",
            "heldout-end",
            "--state-dir",
            library,
            "--device",
            "cpu",
            "--headless-actions",
            "0,2",
            "--output",
            str(target),
        ]
    )
    assert target.exists()
    save_main(
        [
            checkpoint,
            "--name",
            "copy",
            "--scene",
            str(Path(library) / "custom/heldout-end.npz"),
            "--state-dir",
            library,
        ]
    )
    assert load_named_scene("copy", config, library).actions == loaded.actions
    with pytest.raises(SystemExit):
        play_main(
            [checkpoint, "--start-state", "missing", "--state-dir", library, "--device", "cpu"]
        )
    with pytest.raises(SystemExit):
        play_main([checkpoint, "--start-state", "copy", "--empty-start"])


def test_curated_ball_up_snapshot_has_upward_motion_and_complete_actions():
    config = {
        "game": {"name": "breakout"},
        "history": 8,
        "action_history": 8,
        "shape": [3, 210, 160],
        "action_values": [0, 1, 2],
    }
    path = named_scene_path("ball-up", config)
    assert path.parent.parent == START_STATES
    loaded = load_named_scene("ball-up", config)
    assert len(loaded) == 8 and len(loaded.actions) == 7
    with np.load(path, allow_pickle=False) as a:
        frames = a["frames"].transpose(0, 2, 3, 1)
        info = json.loads(str(a["metadata"]))
    ys = []
    for frame in frames:
        mask = (frame == [200, 72, 72]).all(axis=2)
        mask[:95] = False
        mask[187:] = False
        y, _ = np.where(mask)
        assert len(y) == 8
        ys.append(y.mean())
    assert np.all(np.diff(ys) < 0) and ys[-1] == 149.5
    assert info["episode_id"] == 5 and info["frame_position"] == 45
