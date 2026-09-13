import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from play import Player, handle_event
from train import (
    Autoencoder,
    Frames,
    Windows,
    load_model,
    main,
    read_episodes,
    save_model,
)


def write(root, kind, split, rows):
    path = root / kind / split / "00000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "dataset"
    frames = []
    for frame_id in [30, 10, 70, 20, 40, 60, 50]:  # IDs deliberately differ from row offsets.
        buffer = io.BytesIO()
        Image.fromarray(np.full((21, 17, 3), frame_id, dtype=np.uint8)).save(
            buffer, format="WEBP", lossless=True
        )
        frames.append({"frame_id": frame_id, "image": {"bytes": buffer.getvalue(), "path": None}})
    write(root, "frames", "assets", frames)
    for split, episodes in [
        ("train", [(1, [10, 20, 30], [2, 0]), (3, [70, 10], [0])]),
        ("heldout", [(2, [40, 50, 60], [0, 2])]),
    ]:
        metadata, steps = [], []
        for eid, ids, actions in episodes:
            metadata.append({"episode_id": eid, "initial_frame_id": ids[0], "length": len(actions)})
            for step, action in enumerate(actions):
                steps.append(
                    {
                        "episode_id": eid,
                        "step": step,
                        "source_frame_id": ids[step],
                        "successor_frame_id": ids[step + 1],
                        "native_action_json": str(action),
                    }
                )
        write(root, "episodes", split, metadata)
        write(root, "transitions", split, list(reversed(steps)))
    return root


def test_windows_keep_empty_start_actions_and_boundaries(snapshot):
    frames = Frames(snapshot)
    windows = Windows(frames, read_episodes(snapshot, "train"), 4, [0, 2])
    assert len(windows) == 5  # Three transitions plus two initial frames.
    history, action, target = windows[0]
    assert not history.any() and action == 2
    assert torch.equal(target, frames.get(10))
    history, action, target = windows[1]
    assert not history[:-1].any() and torch.equal(history[-1], frames.get(10))
    assert action == 1 and torch.equal(target, frames.get(20))
    history, action, target = windows[2]
    assert torch.equal(history[-2], frames.get(10))
    assert torch.equal(history[-1], frames.get(20))
    assert action == 0 and torch.equal(target, frames.get(30))
    history, action, target = windows[3]
    assert not history.any() and action == 2 and torch.equal(target, frames.get(70))


@pytest.mark.parametrize("field,value", [("source_frame_id", 999), ("step", 10)])
def test_broken_trajectory_rejected(snapshot, field, value):
    path = snapshot / "transitions/train/00000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0][field] = value
    write(snapshot, "transitions", "train", rows)
    with pytest.raises(ValueError, match="broken frame chain|inconsistent steps"):
        read_episodes(snapshot, "train")


def test_unseen_eval_action_and_unknown_frame_rejected(snapshot):
    frames = Frames(snapshot)
    with pytest.raises(ValueError, match="Unknown frame ID"):
        frames.get(999)
    with pytest.raises(ValueError, match="unknown frame ID"):
        frames.check_ids(np.array([10, 999]))
    with pytest.raises(ValueError, match="absent from the training vocabulary"):
        Windows(frames, read_episodes(snapshot, "heldout"), 4, [0])


def test_model_geometry_gradients_and_checkpoint(tmp_path):
    torch.set_num_threads(2)
    config = {
        "format_version": 1,
        "history": 4,
        "action_values": [0, 2],
        "shape": [3, 21, 17],
        "width": 4,
    }
    model = Autoencoder(4, 2, (3, 21, 17), 4)
    # Keep every ReLU path active: random tiny models can legitimately have dead paths.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.01)
    x, a = torch.rand(2, 4, 3, 21, 17), torch.tensor([0, 2])
    y = model(x, a)
    assert y.shape == (2, 3, 21, 17) and torch.isfinite(y).all()
    y.square().mean().backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    save_model(tmp_path / "model.pt", model, config)
    restored, loaded = load_model(tmp_path / "model.pt", torch.device("cpu"))
    assert loaded == config
    assert torch.equal(restored(x, a), y)


class Spy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, history, action):
        self.calls.append((history.clone(), action.clone()))
        return torch.full((1, 3, 21, 17), len(self.calls) / 10)


def test_player_empty_bootstrap_predicted_feedback_and_reset():
    spy = Spy()
    config = {"history": 4, "shape": [3, 21, 17], "action_values": [0, 2]}
    player = Player(spy, config, torch.device("cpu"))
    assert len(spy.calls) == 0 and not player.history and player.frame is None
    player.advance(0)
    assert len(spy.calls) == 1 and not spy.calls[0][0].any()
    assert spy.calls[0][1].item() == 2 and player.steps == 0
    first = player.frame.clone()
    player.advance(2)
    assert spy.calls[-1][1].item() == 1
    assert torch.equal(spy.calls[-1][0][0, -1], first)
    assert player.steps == 1
    player.reset()
    assert len(spy.calls) == 2 and not player.history and player.steps == 0
    player.advance(2)
    assert not spy.calls[-1][0].any() and len(player.history) == 1 and player.steps == 0


def test_only_fresh_action_key_presses_advance():
    import pygame

    spy = Spy()
    config = {"history": 4, "shape": [3, 21, 17], "action_values": [0, 2]}
    player = Player(spy, config, torch.device("cpu"))
    keymap = {pygame.K_LEFT: 2}
    for event in [
        pygame.event.Event(pygame.MOUSEMOTION),
        pygame.event.Event(pygame.KEYUP, key=pygame.K_LEFT),
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT, repeat=True),
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_x),
    ]:
        assert handle_event(player, event, keymap)
    assert len(spy.calls) == 0
    handle_event(player, pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT), keymap)
    assert player.steps == 0 and len(spy.calls) == 1
    handle_event(player, pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT), keymap)
    assert player.steps == 1 and len(spy.calls) == 2
    assert not handle_event(player, pygame.event.Event(pygame.QUIT), keymap)


@pytest.mark.parametrize("workers", [0, 1])
def test_training_cli_and_player_load(snapshot, tmp_path, workers):
    torch.set_num_threads(2)
    output = tmp_path / "run"
    main(
        [
            "--dataset",
            str(snapshot),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--history",
            "2",
            "--width",
            "4",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--workers",
            str(workers),
        ]
    )
    model, config = load_model(output / "best.pt", torch.device("cpu"))
    player = Player(model, config, torch.device("cpu"))
    player.advance(2)
    assert player.pixels().shape == (21, 17, 3)
    assert (output / "metrics.jsonl").is_file()
    latest, progress = load_model(output / "latest.pt", torch.device("cpu"))
    assert progress["train_samples_completed"] == 5
    assert progress["epoch_complete"] is False
    assert torch.equal(next(latest.parameters()), next(model.parameters()))


def test_training_split_overlap_rejected(snapshot, tmp_path):
    # A different split name does not make the same episode independent evaluation data.
    for kind in ["episodes", "transitions"]:
        rows = pq.read_table(snapshot / kind / "train/00000.parquet").to_pylist()
        write(snapshot, kind, "heldout", rows)
    with pytest.raises(ValueError, match="episode IDs overlap"):
        main(["--dataset", str(snapshot), "--output", str(tmp_path / "bad"), "--device", "cpu"])


def test_pygame_loop_waits_for_actions(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    import play

    spy = Spy()
    config = {"history": 4, "shape": [3, 21, 17], "action_values": [0, 1, 2]}
    monkeypatch.setattr(play, "load_model", lambda *_: (spy, config))
    batches = iter(
        [
            [pygame.event.Event(pygame.MOUSEMOTION)],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)],
            [
                pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE),
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, repeat=True),
            ],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT)],
            [pygame.event.Event(pygame.QUIT)],
        ]
    )
    monkeypatch.setattr(pygame.event, "get", lambda: next(batches))
    calls_per_refresh = []
    monkeypatch.setattr(pygame.display, "flip", lambda: calls_per_refresh.append(len(spy.calls)))
    play.main([str(tmp_path / "unused.pt"), "--device", "cpu", "--scale", "1"])
    assert calls_per_refresh == [0, 1, 1, 2, 2]


def test_compact_windows_preserve_every_pixel_action_and_bootstrap(snapshot):
    episodes = read_episodes(snapshot, "train")
    original = Windows(Frames(snapshot), episodes, 4, [0, 2])
    compact = Windows(Frames(snapshot, compact=True), episodes, 4, [0, 2])
    for i in range(len(original)):
        history, action, target = original[i]
        packed_history, packed_action, packed_target = compact[i]
        assert packed_history.dtype == packed_target.dtype == torch.uint8
        assert action == packed_action
        assert torch.equal(history, packed_history.float() / 255)
        assert torch.equal(target, packed_target.float() / 255)
