"""Contradiction accounting and independent-target gradients."""

import hashlib

import numpy as np
import pytest
import torch

from gymemu.commands.dynamics import compose_state_config
from gymemu.state_data import FIELDS, RULES, StateWindows, save_segments, write_json
from gymemu.state_observability import contradictions, fingerprints
from gymemu.state_probes import SingleTargetApproach, train_target


def test_contradictions_are_per_field_and_terminal_targets_are_undefined():
    keys = np.array([b"a" * 32] * 3 + [b"b" * 32] * 2, dtype="V32")
    y = np.zeros((5, 115), np.float32)
    y[1, 0] = 2 / 160
    y[2, :] = 999  # Terminal target must not affect any state-variable statistic.
    y[4, 7] = 1
    done = np.array([False, False, True, False, False])
    ids = np.column_stack((np.arange(5), np.ones(5, int)))
    r = contradictions(keys, y, done, ids)
    assert r["fields"][FIELDS[0]]["conflicting_groups"] == 1
    assert r["fields"][FIELDS[1]]["conflicting_groups"] == 0
    assert r["fields"]["terminal"]["conflicting_groups"] == 1
    assert r["fields"]["brick_grid"]["conflicting_groups"] == 1
    assert r["fields"]["brick_grid"]["minimum_empirical_classification_errors"] == 1
    assert r["fields"][FIELDS[0]]["empirical_mse_floor_native"] == pytest.approx(0.5)


@pytest.fixture
def probe_cache(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    manifest = {"status": "complete", "format_version": 1, "rules": RULES, "identity": "synthetic"}
    for split, eid in [("train", 1), ("validation", 2)]:
        states = np.zeros((13, 115), np.float32)
        states[:, :7] = [0.2, 0.4, 0.5, -0.3, 0.5, 0.01, 1]
        states[:, 0] += np.arange(13) * 0.001
        states[:8, 7:] = 1
        states[8:, 7:] = 1
        states[8:, 7] = 0
        segments = [
            {
                "states": states,
                "actions": np.arange(12) % 3,
                "terminal": np.r_[np.zeros(11, bool), True],
                "episode_id": eid,
                "first_step": 0,
            }
        ]
        manifest[split] = save_segments(root / split, segments)
    write_json(root / "manifest.json", manifest)
    return root


def test_fingerprints_match_exact_windows_and_full_prefix(probe_cache):
    d = StateWindows(probe_cache, "train")
    b = d.batch(d.indices, 3, 4)
    keys = fingerprints(d, 3, 4)
    for j in range(len(keys)):
        raw = (
            b["history"][j].numpy().tobytes()
            + b["history_valid"][j].numpy().tobytes()
            + b["action_history"][j].numpy().tobytes()
        )
        assert keys[j].tobytes() == hashlib.sha256(raw).digest()
    full = fingerprints(d, None, None)
    assert len(np.unique(full)) == len(d.indices)


def test_scalar_loss_has_no_other_output_gradient(probe_cache):
    cfg = compose_state_config()
    cfg["model"].update(width=8, depth=1, history=1, past_actions=1)
    model = SingleTargetApproach(cfg, FIELDS[0], 100)
    d = StateWindows(probe_cache, "train")
    b = d.batch(d.indices[:4], 1, 1)
    loss = model.loss(model(b), b)
    loss.backward()
    grad = model.predictor.head.bias.grad
    assert grad[0].abs() > 0
    assert torch.count_nonzero(grad[1:]) == 0
    changed = {k: v.clone() for k, v in b.items()}
    changed["target"][..., 1:] = 999
    torch.testing.assert_close(loss, model.loss(model(changed), changed))
    terminal = d.batch(d.indices[-1:], 1, 1)
    assert model.loss(model(terminal), terminal) == 0


def test_isolated_training_saves_target_contract_without_test(probe_cache, tmp_path):
    cfg = compose_state_config()
    cfg["cache"] = str(probe_cache)
    cfg["model"].update(width=8, depth=1, history=1, past_actions=1)
    torch.set_num_threads(1)
    result = train_target(cfg, FIELDS[0], tmp_path / "run", 1, 10, 91)
    assert result["target"] == FIELDS[0]
    assert not result["test_evaluated"]
    checkpoint = torch.load(tmp_path / "run/best.pt", weights_only=True)
    assert checkpoint["format"] == "gymemu-state-probe-v1"
    assert checkpoint["cache_identity"] == "synthetic"


def test_restored_paddle_context_keeps_startup_and_clears_at_loss(tmp_path):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from gymemu.state_paddle_context import RecordedPaddleContext

    source = tmp_path / "source"
    (source / "episodes/train").mkdir(parents=True)
    (source / "transitions/train").mkdir(parents=True)
    (source / "manifest.json").write_text("{}\n")
    pq.write_table(
        pa.Table.from_pylist([{"episode_id": 1, "length": 6, "initial_frame_id": 100}]),
        source / "episodes/train/000.parquet",
    )
    states = np.zeros((7, 115), np.float32)
    states[:, :7] = [0.2, 0.4, 0.5, -0.3, 0.5, 0.01, 1]
    states[:, 4] += np.arange(7) * 0.01
    states[4, 1] = 0
    rows = []
    for t in range(6):
        labels = dict(zip(FIELDS, states[t + 1, :7].tolist()))
        labels["lives"] = 5 if t < 3 else 4
        tree = ["dict", [["labels", ["dict", [[k, ["scalar", v]] for k, v in labels.items()]]]]]
        rows.append(
            {
                "episode_id": 1,
                "step": t,
                "selected_action_json": str(t % 3),
                "record_json": json.dumps({"structure": json.dumps(tree)}),
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), source / "transitions/train/000.parquet")
    cache = tmp_path / "warmup-cache"
    cache.mkdir()
    segments = []
    for start, done in [(3, True), (5, False)]:
        segments.append(
            {
                "states": states[start : start + 2],
                "actions": np.array([start % 3]),
                "terminal": np.array([done]),
                "episode_id": 1,
                "first_step": start,
            }
        )
    counts = save_segments(cache / "train", segments)
    write_json(
        cache / "manifest.json",
        {
            "status": "complete",
            "format_version": 1,
            "rules": RULES,
            "identity": "warmup",
            "dataset": str(source),
            "dataset_manifest_sha256": hashlib.sha256(
                (source / "manifest.json").read_bytes()
            ).hexdigest(),
            "episode_ids": {"train": [1]},
            "train": counts,
        },
    )
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    d = StateWindows(cache, "train")
    context = RecordedPaddleContext(d, 4)
    b = context.attach(d.indices, d.batch(d.indices, 1, 0))
    assert b["paddle_context"][:, :, 3].tolist() == [[0, 1, 1, 1], [0, 0, 1, 1]]
    torch.testing.assert_close(b["paddle_context"][:, -1, :3], b["history"][:, -1, 4:7])
    assert b["paddle_context"][:, -1, 4:].argmax(-1).tolist() == [0, 2]
    assert all(p.read_bytes() == value for p, value in before.items())
    cfg = compose_state_config()
    cfg["model"].update(
        kind="state_paddle_context_probe",
        width=8,
        depth=1,
        history=1,
        past_actions=0,
        paddle_context=4,
    )
    model = SingleTargetApproach(cfg, FIELDS[4], 100)
    loss = model.loss(model(b), b)
    assert torch.isfinite(loss)
    loss.backward()


def test_event_diagnostics_mask_terminal_successors_and_preserve_targets(probe_cache):
    from gymemu.state_probe_diagnostics import diagnose_target, event_masks

    previous = np.zeros((5, 115), np.float32)
    previous[:, :7] = [80 / 160, 180 / 255, 0.5, 0.5, 0.5, 0.01, 1]
    target = previous.copy()
    target[1, 3] = -0.5  # Paddle proxy, far from wall/brick geometry.
    previous[2, 0] = 8 / 160
    target[2, 2] = -0.5
    target[3, 7] = 1
    target[4, :] = 999  # Undefined terminal successor must enter no state category.
    terminal = np.array([False, False, False, False, True])
    masks = event_masks(previous, target, terminal)
    assert masks["free_flight_proxy"].tolist() == [True, False, False, False, False]
    assert masks["paddle_collision_proxy"].tolist() == [False, True, False, False, False]
    assert masks["wall_collision_proxy"].tolist() == [False, False, True, False, False]
    for name, mask in masks.items():
        assert bool(mask[-1]) == (name == "terminal")
    cfg = compose_state_config()
    cfg["model"].update(width=8, depth=1, history=1, past_actions=1)
    d = StateWindows(probe_cache, "validation")
    model = SingleTargetApproach(cfg, FIELDS[0], 100)
    report = diagnose_target(model, d, cfg)
    assert report["counts"]["terminal"] == 1
    assert report["events"]["all_live"]["model"]["samples"] == 11
    assert all(x["step"] < 11 for x in report["worst_examples"])
    assert report["split"] == "validation"


def test_event_sampling_is_training_only_and_reproducible(probe_cache):
    from gymemu.state_probe_diagnostics import sample_training_indices

    d = StateWindows(probe_cache, "train")
    sampled = sample_training_indices(d, d.indices, 10, 91, "balanced_events")
    again = sample_training_indices(d, d.indices, 10, 91, "balanced_events")
    np.testing.assert_array_equal(sampled, again)
    assert set(sampled) <= set(d.indices)
    # One brick-change transition and one terminal; both are in the event stratum.
    assert np.isin(sampled, [7, 11]).sum() == 5
    with pytest.raises(ValueError, match="training trajectories"):
        sample_training_indices(
            StateWindows(probe_cache, "validation"), d.indices, 10, 91, "uniform"
        )


def test_probe_continuation_checks_target_and_data(probe_cache, tmp_path):
    cfg = compose_state_config()
    cfg["cache"] = str(probe_cache)
    cfg["model"].update(width=8, depth=1, history=1, past_actions=1)
    train_target(cfg, FIELDS[0], tmp_path / "first", 1, 10, 91)
    checkpoint = tmp_path / "first/best.pt"
    r = train_target(cfg, FIELDS[0], tmp_path / "next", 1, 10, 91, initial_checkpoint=checkpoint)
    assert r["event_diagnostics"]["split"] == "validation"
    with pytest.raises(ValueError, match="does not match"):
        train_target(cfg, FIELDS[1], tmp_path / "wrong", 1, 10, 91, initial_checkpoint=checkpoint)


def test_hidden_hit_count_uses_only_prior_transitions_and_resets():
    from types import SimpleNamespace

    from gymemu.state_hidden_context import source_hit_counts

    states = np.zeros((6, 115), np.float32)
    states[:, :7] = [0.5, 180 / 255, 0.5, 1 / 3.375, 0.5, 0, 1]
    states[[1, 2, 4, 5], 3] *= -1
    data = SimpleNamespace(
        indices=np.array([0, 1, 3, 4]),
        arrays={
            "states": states,
            "starts": np.array([0, 0, 0, 3, 3, 3]),
            "terminal": np.zeros(6, bool),
        },
    )
    counts = source_hit_counts(data)
    assert counts.tolist() == [0, 1, 0, 0, 1, 0]
    states[4:, 3] *= -1
    changed = source_hit_counts(data)
    np.testing.assert_array_equal(counts[:4], changed[:4])
    assert changed[4] == 0


def test_native_controller_keeps_source_state_before_action():
    from gymemu.state_hidden_context import PaddleController

    controller = PaddleController([(1, 0), (2000, 162)])
    before = controller.state()
    controller.step(2)
    controller.step(2)
    assert before == [2048, 162, 0, 0]
    assert controller.state() == [2047, 162, 1, 1]
    assert controller.x == 48
    assert before == [2048, 162, 0, 0]  # A stored source state is not mutated later.


def test_hidden_sidecar_alignment_and_checksum(probe_cache, tmp_path):
    import json

    from gymemu.state_hidden_context import HiddenInputs

    data = StateWindows(probe_cache, "train")
    values = np.zeros((len(data.arrays["states"]), 5), np.float32)
    values[:, 0] = np.arange(len(values))
    sidecar = tmp_path / "hidden"
    sidecar.mkdir()
    np.save(sidecar / "train.npy", values)
    report = {
        "format": "gymemu-hidden-inputs-v1",
        "cache_identity": "synthetic",
        "files": {"train": hashlib.sha256((sidecar / "train.npy").read_bytes()).hexdigest()},
    }
    (sidecar / "manifest.json").write_text(json.dumps(report))
    hidden = HiddenInputs(data, sidecar)
    batch = {}
    hidden.attach(np.array([3, 1]), batch)
    assert batch["hidden_state"][:, 0].tolist() == [3, 1]
    np.save(sidecar / "train.npy", values + 1)
    with pytest.raises(ValueError, match="checksum"):
        HiddenInputs(data, sidecar)


def test_hidden_probe_masks_features_without_changing_architecture(probe_cache):
    cfg = compose_state_config()
    cfg["model"].update(
        kind="state_hidden_probe",
        history=1,
        past_actions=1,
        width=8,
        depth=1,
        paddle_context=2,
        hidden_mode="controller",
    )
    model = SingleTargetApproach(cfg, FIELDS[4], 100)
    data = StateWindows(probe_cache, "train")
    batch = data.batch(data.indices[:4], 1, 1)
    batch["paddle_context"] = torch.zeros((4, 2, 8))
    batch["hidden_state"] = torch.zeros((4, 5))
    with torch.no_grad():
        model.predictor.head.weight.fill_(1)
        model.predictor.body[0].weight[:, -5:].fill_(1)
    expected = model(batch)
    batch["hidden_state"][:, 4] = 1
    torch.testing.assert_close(model(batch), expected)
    batch["hidden_state"][:, 0] = 1
    assert not torch.allclose(model(batch), expected)
    cfg["model"]["hidden_mode"] = "none"
    control = SingleTargetApproach(cfg, FIELDS[4], 100)
    assert sum(p.numel() for p in control.parameters()) == sum(
        p.numel() for p in model.parameters()
    )


def test_compact_paddle_encoding_and_discrete_output():
    from gymemu.models import build_model

    model = build_model(
        {
            "kind": "paddle_transition_mlp",
            "encoding": "hybrid",
            "objective": "classification",
            "width": 8,
            "depth": 1,
            "delta_min": -2,
            "delta_max": 2,
        }
    )
    x = torch.tensor([[80.0, 2048.0, 162.0, 60.0, 1.0, 2.0], [80.0, 2049.0, 162.0, 60.0, 1.0, 2.0]])
    encoded = model.encode(x)
    assert encoded.shape == (2, 43)
    assert torch.count_nonzero(encoded[0, 8:] != encoded[1, 8:]) == 1
    assert encoded[:, 5:8].tolist() == [[0, 0, 1], [0, 0, 1]]
    assert model.delta(torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])).item() == -1


def test_compact_paddle_data_and_checkpoint_roundtrip(probe_cache, tmp_path):
    from gymemu.paddle_transition_training import load_checkpoint, load_data, train

    hidden = tmp_path / "internal"
    hidden.mkdir()
    files = {}
    for split in ["train", "validation"]:
        data = StateWindows(probe_cache, split)
        values = np.zeros((len(data.arrays["states"]), 5), np.float32)
        values[:, :4] = [0.5, 0.5, 1, 1]
        np.save(hidden / f"{split}.npy", values)
        files[split] = hashlib.sha256((hidden / f"{split}.npy").read_bytes()).hexdigest()
    write_json(
        hidden / "manifest.json",
        {"format": "gymemu-hidden-inputs-v1", "cache_identity": "synthetic", "files": files},
    )
    train_data = load_data(probe_cache, hidden, "train")
    val_data = load_data(probe_cache, hidden, "validation")
    assert train_data["x"].shape == (11, 6)  # Death excluded; no ball, bricks or history.
    assert train_data["x"][0].tolist() == [80, 1928, 118, 60, 1, 0]
    assert set(train_data["episodes"]).isdisjoint(val_data["episodes"])
    result = train(train_data, val_data, tmp_path / "compact", epochs=1, width=8, depth=1)
    assert result["validation"]["samples"] == 11
    model, checkpoint = load_checkpoint(tmp_path / "compact/best.pt")
    assert checkpoint["format"] == "gymemu-paddle-transition-v1"
    assert torch.isfinite(model(val_data["x"])).all()
    assert not result["test_evaluated"]
    with pytest.raises(ValueError, match="train/validation"):
        load_data(probe_cache, hidden, "test")


def test_controller_early_stop_uses_current_last_life_state():
    from gymemu.state_hidden_context import controller_native_frames

    assert controller_native_frames(None) == 2
    assert controller_native_frames({"lives": 1, "ball_y": 207}) == 2
    assert controller_native_frames({"lives": 1, "ball_y": 208}) == 1
    assert controller_native_frames({"lives": 2, "ball_y": 208}) == 2


def test_charge_targets_join_by_episode_step_and_reject_changed_files(probe_cache, tmp_path):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from gymemu.cache import file_hash
    from gymemu.paddle_transition_training import load_data

    manifest = json.loads((probe_cache / "manifest.json").read_text())
    manifest["dataset_manifest_sha256"] = "source-identity"
    write_json(probe_cache / "manifest.json", manifest)
    data = StateWindows(probe_cache, "train")
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    values = np.zeros((len(data.arrays["states"]), 5), np.float32)
    values[:, 0] = 0.5
    np.save(hidden / "train.npy", values)
    write_json(
        hidden / "manifest.json",
        {
            "format": "gymemu-hidden-inputs-v1",
            "cache_identity": "synthetic",
            "files": {"train": file_hash(hidden / "train.npy")},
        },
    )
    annotated = tmp_path / "annotated"
    receipts = annotated / "annotations/breakout-paddle-controller-v1"
    receipts.mkdir(parents=True)
    write_json(annotated / "manifest.json", {})
    write_json(
        receipts / "validation.json",
        {
            "status": "validated",
            "source_manifest_sha256": "source-identity",
        },
    )
    path = annotated / "transitions/train/00000.parquet"
    path.parent.mkdir(parents=True)
    rows = [
        {"episode_id": 1, "step": step, "source_paddle_charge": 1928, "paddle_charge": 1928 + step}
        for step in reversed(range(12))
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    write_json(
        receipts / "files.json",
        [{"relative": "transitions/train/00000.parquet", "sha256": file_hash(path)}],
    )
    result = load_data(probe_cache, hidden, "train", target="paddle_charge", annotations=annotated)
    assert result["y"].tolist() == list(range(11))
    assert result["target"] == "paddle_charge"
    rows[0]["paddle_charge"] += 1
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_data(probe_cache, hidden, "train", target="paddle_charge", annotations=annotated)
