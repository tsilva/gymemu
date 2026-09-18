"""Single-ball alignment, leakage, recursive gradients, and checkpoint contract."""

import copy
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from gymemu.commands.dynamics import compose_state_config
from gymemu.state_approach import StateApproach, StatePlayer
from gymemu.state_data import FIELDS, StateWindows, prepare, segment_episode, state_values
from gymemu.state_training import evaluate_checkpoint, load_checkpoint, playback, train


@pytest.fixture
def state_snapshot(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "manifest.json").write_text("{}\n")
    for split, ids in (("train", [10, 20, 30]), ("heldout", [40, 50])):
        episodes, transitions = [], []
        for eid in ids:
            episodes.append({"episode_id": eid, "length": 10, "initial_frame_id": eid * 100})
            lives = 5
            for t in range(10):
                if t in (3, 6):
                    lives -= 1
                labels = dict(zip(FIELDS, [0.2 + t * 0.01, 0.4, 0.5, -0.3, 0.5, -0.01, 1.0]))
                labels.update(lives=lives)
                if t == 3:
                    labels["ball_y_normalized"] = 0
                # t=6 loses a life but has already re-served by the captured successor.
                tree = [
                    "dict",
                    [["labels", ["dict", [[k, ["scalar", v]] for k, v in labels.items()]]]],
                ]
                transitions.append(
                    {
                        "episode_id": eid,
                        "step": t,
                        "source_frame_id": eid * 100 + t * 7,
                        "successor_frame_id": eid * 100 + (t + 1) * 7,
                        "selected_action_json": str(t % 3),
                        "native_action_json": str(t % 3),
                        "action_override_rule_id": None,
                        "configured_frame_skip": 2,
                        "record_json": json.dumps({"structure": json.dumps(tree)}),
                        "brick_grid": [[1] * 18 for _ in range(6)],
                        "brick_grid_suspect": False,
                        "is_initial_brick_layout": False,
                        "terminated": False,
                        "truncated": t == 9,
                    }
                )
        for name, rows in (("episodes", episodes), ("transitions", list(reversed(transitions)))):
            folder = root / name / split
            folder.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist(rows), folder / "000.parquet")
    return root


@pytest.fixture
def state_cache(state_snapshot, tmp_path):
    cache = tmp_path / "cache"
    prepare(
        state_snapshot,
        cache,
        {"split_seed": 9, "train_episodes": 2, "validation_episodes": 1, "test_episodes": 1},
    )
    return cache


def config(cache, output):
    cfg = compose_state_config()
    cfg.update(cache=str(cache), output=str(output))
    cfg["model"].update(width=8, depth=1, history=3, past_actions=4)
    cfg["trainer"].update(
        epochs=1,
        rollout_epochs=1,
        rollout_horizons=[3],
        batch_size=8,
        threads=1,
        train_samples=0,
        validation_samples=0,
        rollout_starts=8,
    )
    cfg["evaluation"].update(horizons=[1, 3, 8], starts=8)
    return cfg


def test_signed_state_and_missing_y_cannot_fabricate_death():
    labels = dict(zip(FIELDS, [0.2, 0.3, -1.0, -0.5, 0.4, -0.03, 0.75]))
    state, valid = state_values(labels, np.ones((6, 18)))
    assert valid and state[2] == -1
    labels.pop("ball_y_normalized")
    missing, valid = state_values(labels, np.ones((6, 18)))
    assert not valid and np.isnan(missing[1])
    parts = segment_episode(
        np.stack([state, missing]),
        np.array([5, 5]),
        np.array([True, False]),
        np.array([1]),
        np.array([False]),
        episode_id=1,
    )
    assert parts == []


def test_virtual_segments_no_respawn_leakage_or_false_censored_terminal(state_cache):
    data = StateWindows(state_cache, "train")
    segments = json.loads((state_cache / "train/segments.json").read_text())
    assert [(s["first_step"], s["transitions"], s["terminal"]) for s in segments[:3]] == [
        (1, 3, True),
        (5, 2, True),
        (7, 3, False),
    ]
    a = data.arrays
    for idx in data.indices:
        batch = data.batch([idx], history=8, past_actions=10, horizon=12)
        ncontext = min(8, idx - a["starts"][idx] + 1)
        assert batch["history_valid"].sum() == ncontext
        assert batch["valid"].sum() == a["ends"][idx] - idx
        assert (batch["action_history"][0, : 10 - (idx - a["starts"][idx])].eq(3)).all()
        np.testing.assert_array_equal(batch["history"][0, -1], a["states"][idx])
        assert batch["actions"][0, 0] == a["steps"][idx] % 3
        if batch["terminal"].any():
            terminal_index = int(batch["terminal"].nonzero()[0, 1])
            assert not batch["valid"][0, terminal_index + 1 :].any()
    train_ids = set(data.manifest["episode_ids"]["train"])
    assert not train_ids & set(data.manifest["episode_ids"]["validation"])
    assert not train_ids & set(data.manifest["episode_ids"]["test"])


def test_quality_gap_censors_without_becoming_a_life_loss():
    states = np.ones((5, 115), np.float32) * 0.5
    parts = segment_episode(
        states,
        np.ones(5) * 5,
        np.array([True, True, False, True, True]),
        np.array([0, 1, 2, 0]),
        np.zeros(4, bool),
        episode_id=2,
    )
    assert [p["first_step"] for p in parts] == [0, 3]
    assert all(not p["terminal"].any() for p in parts)


@pytest.mark.parametrize("kind", ["state_mlp", "state_gru"])
def test_terminal_targets_masked_and_predicted_termination_does_not_mask_training(
    state_cache, tmp_path, kind
):
    cfg = config(state_cache, tmp_path / "run")
    cfg["model"]["kind"] = kind
    data = StateWindows(state_cache, "train")
    terminals = data.indices[data.arrays["terminal"][data.indices]]
    batch = data.batch(terminals, 3, 4)
    approach = StateApproach(cfg["model"], cfg["loss"])
    first, parts = approach.loss(batch)
    assert all(parts[k] == 0 for k in ("motion", "width", "bricks"))
    changed = {k: v.clone() for k, v in batch.items()}
    changed["target"].fill_(0.75)
    second, _ = approach.loss(changed)
    torch.testing.assert_close(first, second)
    first.backward()
    assert approach.predictor.head.bias.grad[-1].abs() > 0
    batch = data.batch(data.indices[:2], 3, 4, horizon=3)
    with torch.no_grad():
        approach.predictor.head.bias[-1] = 100
    _, states = approach.rollout(batch)
    assert states.shape[1] == 3  # Early predicted death cannot escape supervision.
    loss, parts = approach.loss(batch)
    assert parts["motion"] > 0
    loss.backward()


@pytest.mark.parametrize("kind", ["state_mlp", "state_gru"])
def test_player_matches_rollout_and_stops_without_another_prediction(state_cache, tmp_path, kind):
    cfg = config(state_cache, tmp_path / "run")
    cfg["model"]["kind"] = kind
    data = StateWindows(state_cache, "validation")
    batch = data.batch(data.indices[:1], 3, 4, horizon=3)
    approach = StateApproach(cfg["model"], cfg["loss"]).eval()
    with torch.no_grad():
        approach.predictor.head.weight.normal_(std=0.001)
        _, states = approach.rollout(batch)
    player = StatePlayer(
        approach,
        batch["history"],
        batch["history_valid"],
        batch["action_history"][:, :-1],
        threshold=1.0,
    )
    for step in range(3):
        row = player.step(int(batch["actions"][0, step]))
        torch.testing.assert_close(torch.tensor(row["state"]), states[0, step])
    player.threshold = 0
    assert player.step(1)["terminal"]
    with pytest.raises(RuntimeError, match="terminal"):
        player.step(1)


def test_train_checkpoint_validation_threshold_and_headless_play(state_cache, tmp_path):
    cfg = config(state_cache, tmp_path / "run")
    summary = train(cfg)
    assert summary["status"] == "complete" and not summary["test_evaluated"]
    path = tmp_path / "run/best.pt"
    model, payload = load_checkpoint(path)
    assert payload["validation"]["split"] == "validation"
    test_config = copy.deepcopy(cfg)
    test_config["evaluation"]["split"] = "test"
    result = evaluate_checkpoint(path, state_cache, test_config, tmp_path / "test.json")
    assert result["threshold"] == payload["validation"]["threshold"]
    assert result["split"] == "test"
    result = playback(path, state_cache, cfg, tmp_path / "playback.json")
    assert len(result["steps"]) > 0
    assert result["steps"][0]["requested_action"] == 1
    with pytest.raises(ValueError, match="already exists"):
        train(cfg)
    assert model.predictor.history == 3


def test_prepare_never_writes_source(state_snapshot, tmp_path):
    before = {p: p.read_bytes() for p in state_snapshot.rglob("*") if p.is_file()}
    prepare(
        state_snapshot,
        tmp_path / "cache",
        {"split_seed": 1, "train_episodes": 2, "validation_episodes": 1, "test_episodes": 1},
    )
    assert all(p.read_bytes() == value for p, value in before.items())
