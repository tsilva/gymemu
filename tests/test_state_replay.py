import copy
import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from gymemu.config import CONFIG_DIR
from gymemu.models.state_renderer import StateRenderer
from gymemu.state_data import FIELDS
from gymemu.state_hidden_context import PaddleController
from gymemu.state_replay import load_state_replay, reconstruct_episode
from gymemu.web_player import PlaybackSession


def records():
    thresholds = json.loads((CONFIG_DIR / "state_replay_controller.json").read_text())["thresholds"]
    seed = 42
    controller = PaddleController(thresholds)
    for _ in range(int(np.random.default_rng(seed).integers(1, 31, dtype=np.uint64))):
        controller.step(0)
    ids = [500, 40, 300, 80, 600, 90, 750]
    rows = []
    for t, y in enumerate([0, 100, 101, 102, 0, 100]):
        action = t % 3
        controller.step(action + 1)
        labels = dict(
            zip(
                FIELDS,
                [80 / 160, y / 255, 0.5, 1.125 / 3.375, controller.x / 160, controller.vx / 160, 1],
            )
        )
        labels["lives"] = 3 if t < 4 else 2
        tree = [
            "dict",
            [["labels", ["dict", [[key, ["scalar", value]] for key, value in labels.items()]]]],
        ]
        rows.append(
            dict(
                episode_id=6,
                step=t,
                source_frame_id=ids[t],
                successor_frame_id=ids[t + 1],
                record_json=json.dumps({"structure": json.dumps(tree)}),
                brick_grid=np.ones((6, 18), dtype=int).tolist(),
                brick_grid_suspect=False,
                is_initial_brick_layout=False,
                terminated=False,
                truncated=t == 5,
                configured_frame_skip=1,
                native_action_json=json.dumps(action),
                effective_action_json=json.dumps(action),
            )
        )
    return dict(episode_id=6, seed=seed, length=6, initial_frame_id=ids[0]), rows, thresholds


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sources = []
        self.stop = False

    def predict(self, source):
        self.sources.append(source.clone())
        result = torch.zeros(1, 118)
        if self.stop:
            result[:, 117] = 1
        else:
            result[:, :4] = torch.tensor([20, 150.5, 1, -1])
            result[:, 114:117] = torch.tensor([16, 2000, 60])
        return result


class Decoder(torch.nn.Module):
    from_dynamics = staticmethod(StateRenderer.from_dynamics)

    def __init__(self):
        super().__init__()
        self.calls = 0

    def render(self, state):
        self.calls += 1
        return torch.zeros(1, 3, 210, 160)


@pytest.fixture
def replay(tmp_path):
    meta, rows, _ = records()

    def write(name, data):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(data), path)
        return name

    root = "splits/fixed"
    tables = {
        write(f"{root}/episodes/validation/00000.parquet", [meta]): {},
        write(f"{root}/transitions/validation/00000.parquet", rows[::-1]): {},
    }
    publication = "trajectories/published/publication.json"
    (tmp_path / root / "manifest.json").write_text(
        json.dumps(
            dict(
                split_id="fixed",
                episodes={"validation": [6], "test": [7]},
                tables=tables,
                source_publication=publication,
                source_revision="pinned",
            )
        )
    )
    images = []
    for i in [750, 90, 600, 80, 300, 40, 500]:
        buf = io.BytesIO()
        Image.new("RGB", (160, 210), (i % 255, 0, 0)).save(buf, format="PNG")
        images.append({"frame_id": i, "image": {"bytes": buf.getvalue()}})
    write("trajectories/published/frames/assets/00000.parquet", images)
    (tmp_path / publication).write_text(json.dumps({"tables": {"frames/assets/00000.parquet": {}}}))
    config = dict(
        action_values=[0, 1, 2],
        playback_fps=60,
        replay=dict(repository=str(tmp_path), revision="pinned", split_id="fixed"),
    )
    return load_state_replay(Model(), Decoder(), config, "cpu")


def test_recorded_state_action_and_sparse_frame_ids_are_used_each_step(replay):
    assert replay.episode.recorded_steps.tolist() == [2, 3, 4]
    assert replay.episode.source_frames.tolist() == [300, 80, 600]
    assert replay.episode.target_frames.tolist() == [80, 600, 90]
    replay.advance(0)  # Recorded action 2 wins over the supplied override.
    assert replay.recorded_action == 2
    assert torch.equal(replay.model.sources[0], replay.episode.sources[:1])
    assert replay.target[0, 0, 0] == pytest.approx(80 / 255)
    assert replay.input_stack[0, 0, 0, 0] == pytest.approx(45 / 255)
    replay.advance(2)
    assert replay.recorded_action == 0
    assert torch.equal(replay.model.sources[1], replay.episode.sources[1:2])
    assert not replay.state_exact and replay.mse is not None
    assert replay.pixels().shape == (210, 480, 3)


def test_false_terminal_does_not_stop_teacher_forcing_or_decode_placeholder(replay):
    replay.model.stop = True
    replay.advance()
    assert replay.predicted_terminal and not replay.target_terminal
    assert not replay.finished and replay.decoder.calls == 0
    assert replay.mse is None and replay.state_exact is False
    replay.model.stop = False
    replay.advance()
    assert torch.equal(replay.model.sources[1], replay.episode.sources[1:2])
    replay.advance()
    assert replay.finished and replay.target_terminal and replay.state_exact is None
    assert replay.mse is None
    assert "not scored" in replay.context_note
    replay.reset()
    assert replay.steps == 0 and replay.frame is None and not replay.finished


def test_browser_session_seeks_and_reports_state_comparison(replay):
    session = PlaybackSession(replay, "test", {"left": 2}, mode_factory=lambda mode: replay)
    state = session.snapshot()
    assert state["available_modes"] == ["autoregressive", "teacher-forcing"]
    assert state["keymap"] == {"space": None} and not state["history_editable"]
    session.control({"type": "seek", "position": 2})
    assert replay.steps == 2
    assert torch.equal(replay.model.sources[-1], replay.episode.sources[1:2])
    assert session.snapshot()["context_note"] == replay.context_note
    session.control({"type": "seek", "position": 0})
    assert replay.steps == 0 and replay.frame is None


def test_source_memory_is_causal_and_keeps_eighths():
    meta, rows, thresholds = records()
    original = reconstruct_episode(meta, rows, thresholds)
    assert original.sources[0, 8] == 0
    assert original.sources[1, 8] == 1  # Prior observed 1.125 motion, not next prediction.
    changed = copy.deepcopy(rows)
    tree = json.loads(json.loads(changed[3]["record_json"])["structure"])
    fields = dict(dict(tree[1])["labels"][1])
    fields["ball_vy_normalized"][1] = 1 / 3.375
    changed[3]["record_json"] = json.dumps({"structure": json.dumps(tree)})
    following = reconstruct_episode(meta, changed, thresholds)
    assert torch.equal(original.sources[:2], following.sources[:2])
    assert not torch.equal(original.targets[1], following.targets[1])


@pytest.mark.parametrize("change", ["gap", "frameskip", "chain", "controller", "boundary"])
def test_bad_recording_fails_explicitly(change):
    meta, rows, thresholds = records()
    if change == "gap":
        rows.pop(1)
    elif change == "frameskip":
        rows[1]["configured_frame_skip"] = 2
    elif change == "chain":
        rows[2]["source_frame_id"] = 999
    elif change == "controller":
        tree = json.loads(json.loads(rows[1]["record_json"])["structure"])
        dict(dict(tree[1])["labels"][1])["paddle_x_normalized"][1] += 1 / 160
        rows[1]["record_json"] = json.dumps({"structure": json.dumps(tree)})
    else:
        rows[1]["terminated"] = True
    with pytest.raises(ValueError):
        reconstruct_episode(meta, rows, thresholds)


def test_recorded_start_loader_needs_no_images(replay, tmp_path):
    from gymemu.state_replay import load_state_episode

    (tmp_path / "trajectories/published/frames/assets/00000.parquet").unlink()
    (tmp_path / "trajectories/published/publication.json").unlink()
    config = dict(replay=dict(repository=str(tmp_path), revision="pinned", split_id="fixed"))
    episode = load_state_episode(config, split="validation", episode_id=6)
    assert torch.equal(episode.sources, replay.episode.sources)
    assert episode.episode_id == 6
    with pytest.raises(ValueError, match="absent"):
        load_state_episode(config, split="validation", episode_id=7)
