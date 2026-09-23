import json

import pytest
import torch

from gymemu.models import build_model
from gymemu.models.state_renderer import StateRenderer
from gymemu.player import handle_key
from gymemu.state_playback import StatePlayback, load_component, load_source, next_source
from gymemu.web_player import PlaybackSession


class Dynamics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.stop = False

    def predict(self, source):
        self.inputs.append(source.clone())
        result = torch.zeros(1, 118)
        if self.stop:
            result[0, 117] = 1
        else:
            result[0, :4] = torch.tensor([82.5, 100.875, -1.5, 3.375])
            result[0, 4:112] = source[0, 10:118]
            result[0, 4] = 0
            result[0, 112:117] = torch.tensor([1, 7, 12, 1234, 70])
        return result


class Decoder(torch.nn.Module):
    from_dynamics = staticmethod(StateRenderer.from_dynamics)

    def __init__(self):
        super().__init__()
        self.inputs = []

    def render(self, state):
        self.inputs.append(state.clone())
        return torch.full((1, 3, 210, 160), state[0, 0].item() / 160)


def make_player():
    source = torch.ones(1, 119)
    source[0, :10] = torch.tensor([80, 99, 1, 1, 75, 16, 1200, 6, 2, 0])
    return StatePlayback(
        Dynamics(), Decoder(), source, {"action_values": [0, 1, 2], "playback_fps": 60}
    )


def test_keypress_predicts_state_and_feedback_preserves_hidden_fields():
    p = make_player()
    initial = p.frame.clone()
    keys = {"left": 2, "right": 1}
    assert not p.model.inputs and len(p.decoder.inputs) == 1
    handle_key(p, "left", keys)
    handle_key(p, "left", keys, repeat=True)
    handle_key(p, "left", keys)
    assert p.steps == 1 and p.model.inputs[0][0, 118] == 2
    assert torch.equal(p.input_stack[0], initial)
    assert p.decoder.inputs[-1][0, :4].tolist() == [82.5, 100, 70, 12]
    p.frame = torch.zeros_like(p.frame)  # The visible RGB must never become a dynamics input.
    handle_key(p, "right", keys)
    actual = p.model.inputs[1][0]
    assert actual[:10].tolist() == [82.5, 100, -1.5, 3.375, 70, 12, 1234, 7, 7, 1]
    assert actual[10] == 0 and actual[11:118].eq(1).all() and actual[118] == 1
    p.reset()
    assert p.steps == 0 and not p.continuous and not p.held_keys
    assert torch.equal(p.frame, initial) and torch.equal(p.source, p.initial_source)


def test_terminal_stops_without_rendering_or_feedback_and_resets():
    p = make_player()
    p.model.stop = True
    before = p.frame.clone()
    p.continuous = True
    p.advance(0)
    assert p.finished and not p.continuous and p.steps == 1
    assert len(p.decoder.inputs) == 1 and torch.equal(p.frame, before)
    p.advance(1)
    handle_key(p, "tab", {})
    assert not p.continuous and len(p.model.inputs) == 1
    session = PlaybackSession(p, "test", {"space": 0})
    session.control({"type": "play"})
    assert not p.continuous
    assert session.snapshot()["finished"]
    assert not session.snapshot()["history_editable"]
    with pytest.raises(ValueError, match="state inputs"):
        session.control({"type": "reorder_history", "order": [0]})
    p.reset()
    assert not p.finished and p.steps == 0
    with pytest.raises(ValueError, match="Terminal"):
        next_source(torch.ones(1, 118), 0)


@pytest.mark.parametrize("action", [True, 3, -1, 1.0])
def test_bad_actions_rejected(action):
    with pytest.raises(ValueError, match="Unknown action"):
        make_player().advance(action)


def test_portable_decoder_loads_local_registered_code(tmp_path):
    spec = {"palette": [[0, 0, 0], [200, 72, 72]], "width": 4, "layers": 1, "hud_height": 17}
    model = build_model({"kind": "state_renderer", **spec})
    (tmp_path / "config.json").write_text(json.dumps(spec))
    torch.save(model.state_dict(), tmp_path / "pytorch_model.bin")
    (tmp_path / "model.py").write_text('raise RuntimeError("must not execute")')
    loaded, folder = load_component(str(tmp_path), None, "state_renderer", "cpu")
    assert folder == tmp_path and not loaded.training
    assert not any(p.requires_grad for p in loaded.parameters())
    (tmp_path / "source.json").write_text(json.dumps({"source": make_player().source.tolist()}))
    assert load_source(tmp_path / "source.json").shape == (1, 119)


def test_invalid_start_rejected():
    p = make_player()
    for column, value in [(8, 8), (5, 15), (10, 0.5), (1, 0), (0, float("nan"))]:
        source = p.source.clone()
        source[0, column] = value
        with pytest.raises(ValueError):
            p._set_initial_history(source)


def test_direct_autoregressive_launch_uses_recorded_state(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from gymemu.commands import play_state

    recorded = make_player().source.clone()
    recorded[0, 4:7] = torch.tensor([26, 16, 2041])
    (tmp_path / "example_source.json").write_text("invalid synthetic example")
    calls = []
    captured = []

    def load_episode(config, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(episode_id=6, sources=recorded)

    monkeypatch.setattr(play_state, "load_state_episode", load_episode, raising=False)
    monkeypatch.setattr(
        play_state,
        "load_component",
        lambda **kw: (Dynamics() if kw["kind"] == "unified_state_mlp" else Decoder(), tmp_path),
    )
    monkeypatch.setattr(play_state, "serve", lambda player, **kw: captured.append(player))
    play_state.main(["--autoregressive", "--episode-id", "6", "--device", "cpu", "--no-browser"])
    assert calls == [{"dataset": None, "split": "validation", "episode_id": 6}]
    assert torch.equal(captured[0].initial_source, recorded)
    assert captured[0].start_name == "Recorded start | episode 6"
    captured[0].advance(1)
    captured[0].reset()
    assert torch.equal(captured[0].source, recorded)


def test_explicit_start_does_not_load_recorded_data(monkeypatch, tmp_path):
    from gymemu.commands import play_state

    source = make_player().source
    path = tmp_path / "custom.json"
    path.write_text(json.dumps({"source": source.tolist()}))
    captured = []
    monkeypatch.setattr(
        play_state,
        "load_component",
        lambda **kw: (Dynamics() if kw["kind"] == "unified_state_mlp" else Decoder(), tmp_path),
    )
    monkeypatch.setattr(
        play_state,
        "load_state_episode",
        lambda *a, **kw: pytest.fail("Unexpected dataset read"),
        raising=False,
    )
    monkeypatch.setattr(play_state, "serve", lambda player, **kw: captured.append(player))
    play_state.main(["--start-source", str(path), "--device", "cpu", "--no-browser"])
    assert torch.equal(captured[0].source, source)
