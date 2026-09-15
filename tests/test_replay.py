import numpy as np
import pytest
import torch
from PIL import Image

from gymemu.data import Episode, Frames, read_episodes
from gymemu.player import handle_key
from gymemu.replay import ReplayPlayer, load_replay
from play import main


class Spy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, history, actions):
        self.calls.append((history.clone(), actions.clone()))
        return torch.ones_like(history[:, -1])

    def predict_step(self, history, actions, states):
        result = self(history, actions)
        self.calls[-1] += (states.clone(),)
        return result, torch.ones_like(states[:, -1])


def config(**kwargs):
    return {"history": 3, "shape": [3, 21, 17], "action_values": [0, 2], **kwargs}


@pytest.mark.parametrize("action_history", [1, 3])
@pytest.mark.parametrize("state_fields", [[], ["ball_x", "ball_y"]])
def test_replay_uses_only_aligned_recorded_inputs(snapshot, action_history, state_fields):
    frames = Frames(snapshot)
    episodes = read_episodes(snapshot, "train")
    for episode in episodes:
        episode.states = np.array(
            [[0, 0, 0]] + [[0.2, 0.3, 1]] * len(episode.actions), dtype=np.float32
        )
    spy = Spy()
    player = ReplayPlayer(
        spy,
        config(action_history=action_history, state_fields=state_fields),
        torch.device("cpu"),
        frames,
        episodes,
    )
    assert not spy.calls and player.frame is None and not player.continuous
    assert torch.equal(player.target, frames.get(10))
    player.advance(999)  # Replay always takes the recorded action, never a keyboard value.
    history, actions, *states = spy.calls[0]
    assert not history[0, :-1].any()
    assert torch.equal(history[0, -1], frames.get(10))
    assert actions.tolist() == ([1] if action_history == 1 else [[2, 2, 1]])
    if states:
        assert not states[0].any()  # No reset state labels or future state leakage.
    assert player.recorded_action == 2 and player.steps == 1
    assert torch.equal(player.target, frames.get(20))
    assert player.mse == pytest.approx((1 - 20 / 255) ** 2)
    assert np.all(player.pixels()[:, :17] == 255)
    assert np.all(player.pixels()[:, 17:34] == 20)

    player.continuous = True
    player.tick({})
    history, actions, *states = spy.calls[1]
    assert not history[0, 0].any()
    assert torch.equal(history[0, -2:], torch.stack([frames.get(10), frames.get(20)]))
    assert actions.tolist() == ([0] if action_history == 1 else [[2, 1, 0]])
    if states:
        assert torch.equal(states[0][0, -1], torch.tensor([0.2, 0.3, 1]))
    assert torch.equal(player.input_stack, history[0])
    assert torch.equal(player.target, frames.get(30))
    assert player.finished and not player.continuous
    player.advance()
    assert len(spy.calls) == 2  # Never cross an episode boundary.
    player.reset()
    assert player.steps == 0 and player.mse is None and not player.has_prediction
    assert len(spy.calls) == 2 and torch.equal(player.target, frames.get(10))
    player.reset(cycle=True)
    assert player.episode.episode_id == 3
    player.advance()
    history, actions, *states = spy.calls[-1]
    assert not history[0, :-1].any()
    assert torch.equal(history[0, -1], frames.get(70))
    assert actions.tolist() == ([0] if action_history == 1 else [[2, 2, 0]])
    player.reset(cycle=True)
    assert player.episode.episode_id == 1


def test_replay_loader_defaults_selection_and_validation(snapshot):
    spy, device = Spy(), torch.device("cpu")
    metadata = config(
        dataset={"dataset": str(snapshot), "revision": None}, game={"eval_split": "heldout"}
    )
    player = load_replay(spy, metadata, device)
    assert player.episode.episode_id == 2
    player = load_replay(spy, metadata, device, split="train", episode_id=3)
    assert player.episode.episode_id == 3
    with pytest.raises(ValueError, match="Episode ID 99"):
        load_replay(spy, metadata, device, episode_id=99)
    with pytest.raises(ValueError, match="dimensions differ"):
        load_replay(spy, {**metadata, "shape": [3, 22, 17]}, device)
    with pytest.raises(ValueError, match="absent from the training vocabulary"):
        load_replay(spy, {**metadata, "action_values": [0]}, device)
    with pytest.raises(ValueError, match="no dataset provenance"):
        load_replay(spy, config(), device)
    with pytest.raises(ValueError, match="Missing episodes/missing"):
        load_replay(spy, metadata, device, split="missing")


def test_replay_empty_episode_stays_paused(snapshot):
    player = ReplayPlayer(
        Spy(),
        config(),
        torch.device("cpu"),
        Frames(snapshot),
        [Episode(42, np.array([10]), np.array([], dtype=np.int64))],
    )
    assert player.finished
    player.continuous = True
    player.tick({})
    assert not player.continuous and not player.model.calls


def test_replay_headless_without_start_scene(snapshot, tmp_path, monkeypatch, capsys):
    import play

    spy = Spy()
    monkeypatch.setattr(play, "load_model", lambda *_: (spy, config()))
    output = tmp_path / "comparison.png"
    main(
        [
            str(tmp_path / "unused.pt"),
            "--teacher-forcing",
            "--dataset",
            str(snapshot),
            "--headless-steps",
            "10",
            "--output",
            str(output),
            "--device",
            "cpu",
        ]
    )
    assert len(spy.calls) == 2
    with Image.open(output) as image:
        assert image.size == (51, 57)
        assert np.all(np.asarray(image)[36:, 17:34] == 60)
        assert np.all(np.asarray(image)[36:, 34:] == 225)
    assert "RGB MSE" in capsys.readouterr().out


def test_difference_preserves_rgb_sign_and_zero(snapshot):
    player = load_replay(Spy(), config(), torch.device("cpu"), dataset=str(snapshot))
    assert np.all(player.pixels()[:, 34:] == 128)  # Pending, no artificial error at reset.
    player.target = torch.tensor([1.0, 0.5, 0.0]).view(3, 1, 1).expand(3, 21, 17)
    player.frame = torch.tensor([0.0, 0.5, 1.0]).view(3, 1, 1).expand(3, 21, 17)
    assert np.all(player.pixels()[:, 34:] == [0, 128, 255])  # -1, 0, +1 per channel.
    player.frame = player.target.clone()
    assert np.all(player.pixels()[:, 34:] == 128)
    player.reset()
    assert np.all(player.pixels()[:, 34:] == 128)


@pytest.mark.parametrize(
    "options",
    [
        ["--dataset", "unused"],
        ["--headless-steps", "1"],
        ["--teacher-forcing", "--empty-start"],
        ["--teacher-forcing", "--headless-actions", "0"],
        ["--teacher-forcing", "--key-action", "space=2"],
    ],
)
def test_replay_rejects_conflicting_cli_options(options):
    with pytest.raises(SystemExit) as error:
        main(["unused.pt", *options])
    assert error.value.code == 2


def test_replay_end_cannot_resume(snapshot):
    player = load_replay(Spy(), config(), torch.device("cpu"), dataset=str(snapshot))
    player.advance()
    player.advance()
    handle_key(player, "tab", {})
    assert player.finished and not player.continuous
