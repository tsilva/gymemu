"""Checkpoint and input-isolation contracts for the isolated velocity probe."""

import io

import pytest
import torch

from gymemu.models import build_model


def source_rows():
    values = torch.zeros(3, 118)
    values[:, :10] = torch.tensor([80.5, 173, -1.5, 3.375, 72, 16, 2048, 7, 5, 1])
    values[:, 10:] = 1
    return values


@pytest.mark.parametrize("encoding", ["scalar", "hybrid"])
def test_memory_ablation_removes_only_collision_memory(encoding):
    model = build_model({"kind": "ball_velocity_mlp", "encoding": encoding, "memory": False})
    source = source_rows()
    original = source.clone()
    altered = source.clone()
    altered[:, 7:10] = torch.tensor([12, 0, 0])
    assert torch.equal(model.encode(source), model.encode(altered))
    altered[:, 6] += 60
    assert not torch.equal(model.encode(source), model.encode(altered))
    assert torch.equal(source, original)


def test_hybrid_encoding_distinguishes_fraction_and_signed_velocity():
    model = build_model({"kind": "ball_velocity_mlp", "encoding": "hybrid"})
    source = source_rows()
    source[1, 8] = 4
    source[2, 2] = 1.5
    bits = model.encode(source)[:, 118:]
    assert not torch.equal(bits[0], bits[1])
    assert not torch.equal(bits[0], bits[2])


def test_relative_encoding_is_invariant_to_joint_horizontal_translation():
    model = build_model(
        {"kind": "ball_velocity_mlp", "encoding": "hybrid", "relative_offset": True}
    )
    source = source_rows()
    translated = source.clone()
    translated[:, [0, 4]] += 13
    original = model.encode(source)
    assert original.shape == (3, 195)
    assert torch.equal(original[:, -13:], model.encode(translated)[:, -13:])
    translated[:, 0] += 0.5
    assert not torch.equal(original[:, -13:], model.encode(translated)[:, -13:])


@pytest.mark.parametrize("objective", ["classification", "regression", "direction"])
@pytest.mark.parametrize("paddle_only", [False, True])
@pytest.mark.parametrize("spatial_features", [False, True])
def test_probe_checkpoint_roundtrip_preserves_predictions(objective, paddle_only, spatial_features):
    spec = {
        "kind": "ball_velocity_mlp",
        "encoding": "hybrid",
        "objective": objective,
        "paddle_only": paddle_only,
        "spatial_features": spatial_features,
    }
    model = build_model(spec).eval()
    source = source_rows()[:, :9] if paddle_only else source_rows()
    expected = model.predict(source)
    buffer = io.BytesIO()
    torch.save({"model_spec": spec, "model": model.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    restored = build_model(checkpoint["model_spec"]).eval()
    restored.load_state_dict(checkpoint["model"])
    assert torch.equal(expected, restored.predict(source))
    if objective == "classification":
        assert torch.isin(expected, torch.tensor([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2])).all()
    elif objective == "direction":
        assert model(source).shape == (3, 2)
        assert torch.isin(expected, torch.tensor([-1.0, 1.0])).all()


def test_paddle_probe_excludes_bricks_and_contact_from_its_interface():
    model = build_model(
        {
            "kind": "ball_velocity_mlp",
            "encoding": "hybrid",
            "relative_offset": True,
            "paddle_only": True,
        }
    )
    assert model.encode(source_rows()[:, :9]).shape == (3, 85)
    with pytest.raises(ValueError, match="Expected 9 source features"):
        model(source_rows())


def test_spatial_features_respect_memory_mask():
    model = build_model(
        {"kind": "ball_velocity_mlp", "spatial_features": True, "memory": False}
    )
    source = source_rows()
    changed = source.clone()
    changed[:, 8] = 0
    assert torch.equal(model.encode(source), model.encode(changed))
