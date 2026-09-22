"""Shared-trunk gradients, label masking, discrete decoding and portable inference."""

import io

import numpy as np
import pytest
import torch

from gymemu.models import build_model
from gymemu.unified_state_training import encode_targets, fit_vocabulary, joint_loss


def example():
    source = np.zeros((3, 119), dtype=np.float32)
    source[:, :10] = [50.5, 174, 1, 2, 45, 16, 2000, 2, 3, 0]
    source[:, 10:118] = 1
    source[:, 118] = [0, 1, 2]
    source[1, 7] = 12
    source[2, 1] = 210
    target = np.zeros((3, 117), dtype=np.float32)
    target[:, :4] = [52.5, 178.375, 1, 2]
    target[:, 4:112] = 1
    target[:, 112:] = [0, 2, 16, 2000, 44]
    target[1, :4] = [48.5, 170.375, -1, -2]
    target[1, 112:] = [1, 12, 12, 1940, 43]
    target[1, 4 + 23] = 0
    target[2] = np.nan
    terminal = np.array([False, False, True])
    vocabulary = fit_vocabulary(source, target, terminal)
    spec = dict(kind="unified_state_mlp", vocabulary=vocabulary, width=32, blocks=2)
    return source, target, terminal, spec


def test_labels_decode_exactly_and_terminal_targets_have_no_state_gradient():
    source, target, terminal, spec = example()
    model = build_model(spec)
    labels = torch.from_numpy(encode_targets(source, target, terminal, spec["vocabulary"]))
    assert (labels[-1, :-1] == -100).all()
    logits = {}
    for j, (name, size) in enumerate(zip(model.names, model.sizes, strict=True)):
        values = torch.full((3, size), -20.0)
        values.scatter_(1, labels[:, j].clamp(min=0)[:, None], 20)
        logits[name] = values.requires_grad_()
    output = model.decode(torch.from_numpy(source), logits).detach().numpy()
    np.testing.assert_array_equal(output[:2, :117], target[:2])
    np.testing.assert_array_equal(output[:, 117], terminal)
    assert not output[2, :117].any()
    joint_loss(logits, labels).backward()
    for name in model.names[:-1]:
        assert not logits[name].grad[-1].any()
    all_terminal = labels.clone()
    all_terminal[:, :-1] = -100
    assert torch.isfinite(joint_loss(logits, all_terminal))


def test_shared_network_trains_and_checkpoint_round_trips():
    source, target, terminal, spec = example()
    torch.manual_seed(91)
    model = build_model(spec)
    xx = torch.from_numpy(source)
    labels = torch.from_numpy(encode_targets(source, target, terminal, spec["vocabulary"]))
    encoded = model.encode(xx)
    assert encoded.shape == (3, 188)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    initial = float(joint_loss(model(xx), labels).detach())
    for _ in range(40):
        optimizer.zero_grad()
        loss = joint_loss(model(xx), labels)
        loss.backward()
        optimizer.step()
    assert float(joint_loss(model(xx), labels).detach()) < initial * 0.4
    assert all(p.grad is not None for p in model.parameters())
    for name in model.names:
        model.zero_grad()
        model(xx)[name].sum().backward()
        assert model.trunk[0].weight.grad.abs().sum() > 0
    buffer = io.BytesIO()
    torch.save(dict(model_spec=spec, model=model.state_dict()), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = build_model(saved["model_spec"])
    restored.load_state_dict(saved["model"])
    assert torch.equal(restored.predict(xx), model.predict(xx))
    assert model.predict(xx[:0]).shape == (0, 118)


def test_absorbing_stop_skips_invalid_previous_state(monkeypatch):
    source, _, _, spec = example()
    model = build_model(spec)
    xx = torch.from_numpy(source)
    xx[1:] = float("nan")
    seen = []
    original = model.forward

    def record(rows):
        seen.append(len(rows))
        return original(rows)

    monkeypatch.setattr(model, "forward", record)
    result = model.predict(xx, torch.tensor([False, True, True]))
    assert seen == [1]
    assert not result[1:, :117].any() and result[1:, 117].all()
    seen.clear()
    assert model.predict(xx, torch.ones(3, dtype=torch.bool))[:, 117].all()
    assert not seen
    with pytest.raises(ValueError, match="finite"):
        model.predict(xx)


def test_unsupported_labels_and_metadata_fail_explicitly():
    source, target, terminal, spec = example()
    target[0, 0] += 0.125
    with pytest.raises(ValueError, match="outside training vocabulary"):
        encode_targets(source, target, terminal, spec["vocabulary"])
    target[0, 4:6] = 0
    with pytest.raises(ValueError, match="one occupied-cell"):
        fit_vocabulary(source, target, terminal)
    spec["vocabulary"]["dx"] = [float("nan")]
    with pytest.raises(ValueError, match="vocabulary"):
        build_model(spec)
