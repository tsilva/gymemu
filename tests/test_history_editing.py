"""Pixel edits change only the selected diagnostic inputs, using the frame palette."""

import numpy as np
import pytest
import torch

from gymemu.data import Frames, read_episodes
from gymemu.player import Player
from gymemu.replay import ReplayPlayer
from gymemu.scenes import SceneHistory
from gymemu.web_player import PlaybackSession


class SensitiveModel:
    def __init__(self):
        self.calls = []

    def __call__(self, history, tokens):
        self.calls.append((history.clone(), tokens.clone()))
        return history.mean(dim=1)

    def predict_step(self, history, tokens, states):
        prediction = self(history, tokens)
        self.calls[-1] += (states.clone(),)
        return prediction, states[:, -1]


def make_session(snapshot, replay, with_states):
    cfg = {"history": 4, "shape": [3, 21, 17], "action_values": [0, 2], "action_history": 3}
    model = SensitiveModel()
    if with_states:
        cfg["state_fields"] = ["ball_x", "ball_y"]
    if replay:
        frames = Frames(snapshot)
        get = frames.get

        def colored(frame_id):
            frame = get(frame_id).clone()
            frame[:, 0, :2] = torch.tensor([0.0, 1.0])
            return frame

        frames.get = colored
        episodes = read_episodes(snapshot, "train")
        for episode in episodes:
            episode.states = np.array(
                [[0, 0, 0]] + [[0.2, 0.3, 1]] * len(episode.actions), dtype=np.float32
            )
        player = ReplayPlayer(model, cfg, torch.device("cpu"), frames, episodes)
    else:
        frame = torch.full((3, 21, 17), 0.12345)
        frame[:, 0, :2] = torch.tensor([0.0, 1.0])
        history = SceneHistory([frame] * 4, [0, 2, 0], torch.tensor([[0.2, 0.3, 1.0]] * 4))
        player = Player(model, cfg, torch.device("cpu"), history)
    session = PlaybackSession(player, "model.pt", {})
    session.control({"type": "step", "action": 0})
    return session


def edit(session, pixels, frame=3, revision=None):
    session.control(
        {
            "type": "edit_history",
            "frame": frame,
            "pixels": pixels,
            "history_revision": session.snapshot()["history_revision"]
            if revision is None
            else revision,
        }
    )
    return session.snapshot()


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("with_states", [False, True])
def test_paint_repredicts_preserves_context_and_follows_reorders(snapshot, replay, with_states):
    s = make_session(snapshot, replay, with_states)
    base = s.snapshot()
    original = s.player.input_stack.clone()
    context = s.player.model.calls[-1]
    predicted = s.player.frame.clone()
    edited = edit(s, [[0, 255, 255, 255]])
    assert edited["step"] == base["step"]
    assert edited["images"]["prediction"] != base["images"]["prediction"]
    assert edited["history"] == base["history"]
    assert edited["history_edited_frames"] == [3]
    if replay:
        assert edited["images"]["original"] == base["images"]["original"]
        assert edited["mse"] != base["mse"]
    assert torch.equal(s.player.frame, predicted)
    assert torch.equal(s.player.input_stack, original)
    supplied = s.player.model.calls[-1][0][0]
    assert supplied[3, :, 0, 0].eq(1).all()
    assert torch.equal(supplied[:3], original[:3])
    assert torch.equal(supplied[3, :, 1:], original[3, :, 1:])
    for actual, expected in zip(s.player.model.calls[-1][1:], context[1:]):
        assert torch.equal(actual, expected)
    s.control(
        {
            "type": "reorder_history",
            "order": [3, 0, 1, 2],
            "history_revision": edited["history_revision"],
        }
    )
    assert s.player.model.calls[-1][0][0, 0, :, 0, 0].eq(1).all()
    edit(s, [[1, 0, 0, 0]])  # Edit original frame 3 again in its new display slot.
    assert s.player.model.calls[-1][0][0, 0, :, 0, 0].eq(1).all()
    assert s.player.model.calls[-1][0][0, 0, :, 0, 1].eq(0).all()
    if replay:
        s.control({"type": "seek", "position": 0})
        s.control({"type": "seek", "position": 1})
        assert s.snapshot()["images"] == base["images"]
    else:
        expected = torch.stack(list(s.player.history))
        s.control({"type": "step", "action": 0})
        assert torch.equal(s.player.model.calls[-1][0][0], expected)
    assert s.snapshot()["history_edited_frames"] == []
    with pytest.raises(ValueError, match="history has changed"):
        edit(s, [[0, 255, 255, 255]], revision=edited["history_revision"])


@pytest.mark.parametrize(
    "pixels",
    [
        [[0, 12, 34, 56]],
        [[0, 256, 255, 255]],
        [[-1, 0, 0, 0]],
        [[21 * 17, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 255, 255, 255]],
        [[True, 0, 0, 0]],
        [[0, 0.0, 0, 0]],
        [],
        "invalid",
    ],
)
def test_rejects_invalid_pixels_atomically(snapshot, pixels):
    s = make_session(snapshot, False, False)
    base = s.snapshot()
    calls = len(s.player.model.calls)
    with pytest.raises(ValueError):
        edit(s, pixels)
    assert s.snapshot() == base
    assert len(s.player.model.calls) == calls


def test_invalid_frame_and_pending_prediction(snapshot):
    s = make_session(snapshot, True, False)
    for frame in (-1, 4, True, "3"):
        with pytest.raises(ValueError, match="Unknown history frame"):
            edit(s, [[0, 0, 0, 0]], frame)
    s.control({"type": "reset"})
    with pytest.raises(ValueError, match="Predict a frame"):
        edit(s, [[0, 0, 0, 0]])


def test_http_accepts_full_frame_painting(snapshot, tmp_path):
    import json
    import threading
    from urllib.request import Request, urlopen

    from gymemu.web_player import make_server

    s = make_session(snapshot, False, False).start()
    server, url = make_server(s, workspace_path=tmp_path / "workspace.json")
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    root, token = url.split("#token=")
    body = json.dumps(
        {
            "type": "edit_history",
            "frame": 3,
            "history_revision": s.current["history_revision"],
            "pixels": [[i, 255, 255, 255] for i in range(21 * 17)],
        }
    ).encode()
    assert len(body) > 4096  # Painting a whole frame exceeds the old control payload limit.
    try:
        request = Request(
            root + "api/command",
            data=body,
            headers={"X-Player-Token": token, "Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            assert response.status == 200
        assert s.player.model.calls[-1][0][0, 3].eq(1).all()
        assert s.current["history_edited_frames"] == [3]
    finally:
        server.shutdown()
        server.server_close()
        s.close()
        worker.join(timeout=3)


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("with_states", [False, True])
def test_revert_restores_exact_frame_preserving_other_edits_and_order(
    snapshot, replay, with_states
):
    s = make_session(snapshot, replay, with_states)
    if replay:
        s.control({"type": "seek", "position": 2})
    baseline = s.snapshot()
    original = s.player.input_stack.clone()
    baseline_context = s.player.model.calls[-1][1:]
    edit(s, [[0, 255, 255, 255]], frame=2)
    edited = edit(s, [[0, 255, 255, 255]], frame=3)
    order = [3, 0, 1, 2]
    s.control(
        {"type": "reorder_history", "order": order, "history_revision": edited["history_revision"]}
    )
    revision = s.snapshot()["history_revision"]
    calls = len(s.player.model.calls)
    s.control({"type": "revert_history", "frame": 3, "history_revision": revision})
    reverted = s.snapshot()
    assert len(s.player.model.calls) == calls + 1
    assert reverted["history_edited_frames"] == [2]
    assert reverted["history_order"] == order
    supplied = s.player.model.calls[-1][0][0]
    assert torch.equal(supplied[0], original[3])  # Preserve original float precision.
    assert supplied[3, :, 0, 0].eq(1).all()
    assert torch.equal(s.player.input_stack, original)
    assert reverted["step"] == baseline["step"]
    assert reverted["history"] == baseline["history"]
    for actual, expected in zip(s.player.model.calls[-1][1:], baseline_context):
        assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match="history has changed"):
        s.control({"type": "revert_history", "frame": 2, "history_revision": revision})
    s.control(
        {"type": "revert_history", "frame": 2, "history_revision": reverted["history_revision"]}
    )
    assert s.snapshot()["history_edited_frames"] == []
    assert torch.equal(s.player.model.calls[-1][0][0], original[order])
    for frame in (2, -1, 4, True, "2"):
        with pytest.raises(ValueError, match="no edits to revert"):
            s.control(
                {
                    "type": "revert_history",
                    "frame": frame,
                    "history_revision": s.snapshot()["history_revision"],
                }
            )
