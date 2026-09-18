import json

import numpy as np
import pytest
import torch

from gymemu.checkpoints import load_model
from gymemu.commands.compare import collect_runs
from gymemu.commands.play import main as play_main
from gymemu.data import Episode, Frames, read_episodes
from gymemu.engine import train
from gymemu.replay import ReconstructionPlayer, load_replay
from gymemu.web_player import PlaybackSession
from tests.test_experiments import config_for


class CodecSpy(torch.nn.Module):
    playback_modes = ("reconstruction",)

    def __init__(self):
        super().__init__()
        self.calls = []

    def reconstruct(self, frames):
        self.calls.append(frames.clone())
        return frames * 0.5

    def forward(self, *args):
        raise AssertionError("Reconstruction must never call the transition predictor")


def test_reconstruction_alignment_initial_seek_reset_and_end(snapshot):
    frames = Frames(snapshot)
    spy = CodecSpy()
    config = {"history": 8, "shape": [3, 21, 17], "action_values": [0, 2]}
    player = ReconstructionPlayer(
        spy, config, torch.device("cpu"), frames, read_episodes(snapshot, "train")
    )
    session = PlaybackSession(player, "codec.pt", {}, mode_factory=lambda _: player)
    assert player.steps == 0 and not player.continuous and player.has_prediction
    assert torch.equal(spy.calls[-1][0], frames.get(10))
    assert torch.equal(player.target, frames.get(10))
    player.advance(999)  # Actions cannot change the reconstruction input.
    assert torch.equal(spy.calls[-1][0], frames.get(20))
    assert torch.equal(player.target, frames.get(20))
    assert player.mse == pytest.approx((10 / 255) ** 2)
    session.control({"type": "seek", "position": 2})
    assert torch.equal(player.target, frames.get(30))
    assert player.finished and not player.continuous
    calls = len(spy.calls)
    player.advance()
    assert len(spy.calls) == calls
    session.control({"type": "seek", "position": 0})
    assert torch.equal(player.frame, frames.get(10) * 0.5)
    state = session.snapshot()
    assert state["mode"] == "reconstruction"
    assert state["available_modes"] == ["reconstruction"]
    assert state["history_length"] == 1 and not state["history_editable"]
    assert state["action"] is None
    with pytest.raises(ValueError, match="unavailable"):
        session.control({"type": "mode", "mode": "autoregressive"})
    with pytest.raises(ValueError, match="history edits"):
        session.control({"type": "reorder_history", "order": [0]})
    session.control({"type": "next"})
    assert player.episode.episode_id == 3
    assert torch.equal(player.target, frames.get(70))


def test_reconstruction_single_frame_episode_and_capability(snapshot):
    config = {"history": 1, "shape": [3, 21, 17], "action_values": [0, 2]}
    player = ReconstructionPlayer(
        CodecSpy(),
        config,
        torch.device("cpu"),
        Frames(snapshot),
        [Episode(99, np.array([10]), np.array([], dtype=np.int64))],
    )
    assert player.finished and player.has_prediction
    assert player.mse is not None
    with pytest.raises(ValueError, match="no frame reconstruction"):
        load_replay(torch.nn.Identity(), config, "cpu", reconstruction=True)


def test_reconstruction_training_validation_checkpoints_and_playback(snapshot, tmp_path):
    output = tmp_path / "reconstruction"
    cfg = config_for(snapshot, output, "reconstruction")
    cfg.trainer.epochs = 3
    cfg.trainer.eval_batches = None
    train(cfg)
    model, metadata = load_model(output / "best.pt", torch.device("cpu"))
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 3
    assert all(r["stage"] == "representation" for r in records)
    assert all(r["validation"]["samples"] == 3 for r in records)
    assert metadata["evaluation"]["metric"] == "reconstruction_rgb_mse"
    assert metadata["validation"]["mse"] == min(r["validation"]["mse"] for r in records)
    assert all(name.startswith("codec.") for name in model.state_dict())
    for record in records:
        path = output / "stages/representation" / f"epoch-{record['epoch']:04d}.pt"
        _, saved = load_model(path, torch.device("cpu"))
        assert saved["validation"] == record["validation"]
    history, action, target = (
        torch.rand(2, 2, 3, 21, 17),
        torch.tensor([0, 2]),
        torch.rand(2, 3, 21, 17),
    )
    assert torch.equal(
        model.loss(history, action, target), model.loss(history * 0, action + 1, target)
    )
    _, reconstruction = model.evaluate(history, action, target)
    assert torch.equal(reconstruction, model.reconstruct(target))
    for flags in ([], ["--reconstruction"]):
        play_main(
            [
                str(output / "best.pt"),
                *flags,
                "--device",
                "cpu",
                "--headless-steps",
                "2",
                "--output",
                str(output / "reconstruction.png"),
            ]
        )
    with pytest.raises(SystemExit):
        play_main([str(output / "best.pt"), "--autoregressive", "--empty-start"])
    direct = config_for(snapshot, tmp_path / "direct")
    direct.trainer.eval_batches = None
    train(direct)
    assert len(collect_runs([tmp_path])) == 2
