import numpy as np
import torch
from torch import nn

from gymemu.models import build_model
from gymemu.models.latent import FrameCodec
from probe_paddle_rgb import windows


def test_full_rgb_encoder_matches_codec_and_preserves_whole_canvas():
    torch.set_num_threads(2)
    model = build_model({"kind": "paddle_state_cnn", "history": 2, "action_history": 4})
    reference = FrameCodec(3)
    assert [
        (type(x), getattr(x, "in_channels", None), getattr(x, "out_channels", None))
        for x in model.encoder
    ] == [
        (type(x), getattr(x, "in_channels", None), getattr(x, "out_channels", None))
        for x in reference.encoder
    ]
    assert sum(isinstance(x, nn.Conv2d) for x in model.encoder) == 3
    captured = []
    hook = model.encoder[0].register_forward_pre_hook(lambda _, inputs: captured.append(inputs[0]))
    rgb = torch.randint(0, 256, (2, 2, 3, 210, 160), dtype=torch.uint8)
    output = model(rgb, torch.tensor([[0, 1, 2, 3], [1, 1, 2, 3]]))
    hook.remove()
    assert output.shape == (2, 2)
    assert output.dtype == torch.float32
    torch.testing.assert_close(captured[0][:, :, :210], rgb.flatten(0, 1).float() / 255)
    assert not captured[0][:, :, 210:].any()
    output.square().mean().backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0


def test_rgb_history_and_normalized_targets_are_aligned_without_extracted_inputs(tmp_path):
    path = tmp_path / "data.npz"
    np.savez(
        path,
        **{
            "17_frames": np.array([100, 500, 700, 900]),
            "17_actions": np.array([0, 1, 2]),
            "17_labels": np.array([[0, 0], [0.2, -0.01], [0.3, 0.02], [0.4, 0.03]]),
        },
    )
    frames, actions, labels, ids = windows(path, 1, 4, 0, 0)
    np.testing.assert_array_equal(frames, [[500], [700]])
    np.testing.assert_array_equal(actions, [[0, 0, 1, 2], [0, 1, 2, 3]])
    np.testing.assert_allclose(labels, [[0.3, 0.02], [0.4, 0.03]])
    np.testing.assert_array_equal(ids, [[17, 2], [17, 3]])
    current, _, _, _ = windows(path, 2, 1, 0, 0, current=True)
    np.testing.assert_array_equal(current, [[500, 700], [700, 900]])


def test_learned_position_head_uses_only_observed_position_labels(tmp_path):
    from probe_paddle_rgb import position_targets

    path = tmp_path / "labels.npz"
    np.savez(path, **{"17_labels": np.array([[np.nan, np.nan], [0.2, -0.01], [0.3, 0.02]])})
    target = position_targets(path, np.array([[17, 2]]), 2, False, "cpu")
    torch.testing.assert_close(target, torch.tensor([[-1, 32]]))
    model = build_model({"kind": "paddle_position_cnn", "history": 1, "action_history": 4})
    rgb = torch.randint(0, 256, (2, 1, 3, 210, 160), dtype=torch.uint8)
    prediction, logits = model.forward_with_positions(rgb, torch.tensor([[1, 2, 3, 1]] * 2))
    assert prediction.shape == (2, 2)
    assert logits.shape == (2, 1, 160)
    loss = prediction.square().mean() + torch.nn.functional.cross_entropy(
        logits[:, 0], torch.tensor([32, 64])
    )
    loss.backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0


def test_current_state_three_outputs_and_width_normalization(tmp_path):
    from probe_paddle_rgb import state_metrics

    path = tmp_path / "current.npz"
    np.savez(
        path,
        **{
            "17_frames": np.array([100, 500, 700, 900]),
            "17_actions": np.array([0, 1, 2]),
            "17_labels": np.array(
                [[np.nan] * 3, [0.2, -0.01, 1], [0.3, 0.02, 0.5], [0.4, 0.03, 0.5]]
            ),
        },
    )
    frames, actions, labels, ids = windows(path, 2, 1, 0, 0, current=True)
    np.testing.assert_array_equal(frames, [[500, 700], [700, 900]])
    np.testing.assert_array_equal(actions, [[2], [3]])
    np.testing.assert_array_equal(labels, [[0.3, 0.02, 0.5], [0.4, 0.03, 0.5]])
    # Latest input image and target describe the same state. No action after it is used.
    _, no_actions, _, _ = windows(path, 1, 0, 0, 0, current=True)
    assert no_actions.shape == (2, 0)
    model = build_model(
        {"kind": "paddle_state_cnn", "history": 2, "action_history": 1, "outputs": 3}
    )
    rgb = torch.randint(0, 256, (2, 2, 3, 210, 160), dtype=torch.uint8)
    output = model(rgb, torch.tensor(actions))
    assert output.shape == (2, 3)
    output.square().mean().backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    metrics = state_metrics(labels + [1 / 160, 1 / 160, 1 / 16], labels, ids, [160, 160, 16])
    np.testing.assert_allclose(metrics["mae"], [1, 1, 1])
    np.testing.assert_allclose(metrics["normalized_mae"], [1 / 160, 1 / 160, 1 / 16])


def test_seven_state_outputs_preserve_paddle_initialization_and_ball_scales(tmp_path):
    from probe_paddle_rgb import ball_regime_metrics, load_expanded_outputs, state_metrics

    targets = [
        "paddle_x_normalized",
        "paddle_vx_normalized",
        "paddle_width_normalized",
        "ball_x_normalized",
        "ball_y_normalized",
        "ball_vx_normalized",
        "ball_vy_normalized",
    ]
    spec = {"kind": "paddle_state_cnn", "history": 2, "action_history": 1, "outputs": 3}
    old = build_model(spec).eval()
    expanded = build_model(dict(spec, outputs=7)).eval()
    load_expanded_outputs(
        expanded,
        {"spec": spec, "state_dict": old.state_dict(), "manifest": {"targets": targets[:3]}},
        targets,
    )
    rgb = torch.randint(0, 256, (2, 2, 3, 210, 160), dtype=torch.uint8)
    action = torch.tensor([[1], [3]])
    output = expanded(rgb, action)
    assert output.shape == (2, 7)
    torch.testing.assert_close(output[:, :3], old(rgb, action))
    output.square().mean().backward()
    assert expanded.encoder[0].weight.grad.abs().sum() > 0
    scales = np.array([160, 160, 16, 160, 255, 2, 3.375])
    target = np.array(
        [[0.2, 0.01, 1, 0.3, 0.4, -0.5, 0.5], [0.3, -0.01, 0.75, 0.4, 0.5, 0.5, -0.5]]
    )
    metrics = state_metrics(target + 1 / scales, target, np.array([[17, 2], [17, 3]]), scales)
    np.testing.assert_allclose(metrics["mae"], np.ones(7))
    np.testing.assert_allclose(metrics["normalized_mae"], 1 / scales)
    assert set(metrics["by_width"]) == {"0.75", "1.0"}
    source = tmp_path / "ball.npz"
    np.savez(source, **{"17_labels": np.vstack([np.full(7, np.nan), target[0], target])})
    regimes = ball_regime_metrics(
        target + 1 / scales, target, np.array([[17, 2], [17, 3]]), source, scales
    )
    assert regimes["velocity_unchanged"]["n"] == 1
    assert regimes["velocity_changed"]["n"] == 1
    assert regimes["position_unchanged"]["n"] == 1
    np.testing.assert_allclose(regimes["velocity_changed"]["mae"], np.ones(7))


def test_robust_loss_preserves_small_errors_and_caps_large_error_gradients():
    from probe_paddle_rgb import regression_loss

    scales = torch.tensor([160, 160, 16, 160, 255, 2, 3.375])
    zero = torch.zeros(2, 7)
    prediction = torch.stack([0.5 / scales, 2 / scales]).requires_grad_()
    mse = regression_loss(prediction, zero, scales / 10, scales, "mse")
    robust = regression_loss(prediction, zero, scales / 10, scales, "huber")
    torch.testing.assert_close(mse, torch.tensor((0.25 + 4) / 200))
    torch.testing.assert_close(robust, torch.tensor((0.25 + 3) / 200))
    robust.backward()
    assert torch.isfinite(prediction.grad).all()
