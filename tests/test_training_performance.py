import torch
from torch.utils.data import DataLoader

from gymemu.approaches import build_approach
from gymemu.data import Frames, Windows, read_episodes
from gymemu.engine import run_epoch


def test_direct_validation_uses_one_prediction_per_batch(snapshot):
    torch.set_num_threads(2)
    frames = Frames(snapshot, compact=True)
    loader = DataLoader(
        Windows(frames, read_episodes(snapshot, "heldout"), 2, [0, 2]), batch_size=2
    )
    spec = {"kind": "direct", "models": {"predictor": {"kind": "direct_cnn", "width": 4}}}
    model = build_approach(spec, 2, 2, frames.shape)
    model.prepare_stage("next_frame")
    calls = []
    model.predictor.register_forward_hook(lambda *_: calls.append(1))
    result = run_epoch(model, loader, torch.device("cpu"))
    assert len(calls) == len(loader)
    assert result["loss"] == result["mse"]


def test_compiled_loss_preserves_gradients_and_eager_checkpoint_loading(snapshot, tmp_path):
    from gymemu.engine import batch_loss

    frames = Frames(snapshot, compact=True)
    loader = DataLoader(Windows(frames, read_episodes(snapshot, "train"), 2, [0, 2]), batch_size=2)
    spec = {"kind": "direct", "models": {"predictor": {"kind": "direct_cnn", "width": 4}}}
    eager = build_approach(spec, 2, 2, frames.shape)
    compiled = build_approach(spec, 2, 2, frames.shape)
    compiled.load_state_dict(eager.state_dict())
    history, action, target = next(iter(loader))
    expected, _ = batch_loss(eager, history, action, target, True, True)
    actual, _ = torch.compile(batch_loss, backend="eager")(
        compiled, history, action, target, True, True
    )
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected)
    for a, b in zip(eager.parameters(), compiled.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)
    path = tmp_path / "weights.pt"
    torch.save(compiled.state_dict(), path)
    eager.load_state_dict(torch.load(path, weights_only=True))


def test_deferred_nonfinite_check_rejects_before_checkpoint(snapshot):
    import pytest

    frames = Frames(snapshot, compact=True)
    loader = DataLoader(Windows(frames, read_episodes(snapshot, "train"), 2, [0, 2]), batch_size=2)
    spec = {"kind": "direct", "models": {"predictor": {"kind": "direct_cnn", "width": 4}}}
    model = build_approach(spec, 2, 2, frames.shape)
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    checkpoints = []
    with pytest.raises(ValueError, match="Non-finite"):
        run_epoch(
            model,
            loader,
            torch.device("cpu"),
            sync_batches=100,
            checkpoint=lambda *_: checkpoints.append(1),
            checkpoint_seconds=0,
        )
    assert not checkpoints
