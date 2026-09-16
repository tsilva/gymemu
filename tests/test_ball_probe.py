import numpy as np
import torch

from gymemu.models.action_history import ActionHistoryAutoencoder
from probe_ball_latents import Encoder, Scores, targets


def test_probe_latents_reproduce_model_encoder_and_cannot_train_it():
    torch.manual_seed(3)
    model = ActionHistoryAutoencoder(2, 3, (3, 17, 18), width=4)
    encoder = Encoder(model)
    history = torch.rand(3, 2, 3, 17, 18)
    actions = torch.tensor([[3, 0], [0, 1], [1, 2]])
    captured = []
    hook = model.encoder.register_forward_hook(lambda _, args, out: captured.append(out))
    expected = model(history, actions)
    hook.remove()
    latent = encoder(history, actions)
    torch.testing.assert_close(latent, captured[0].flatten(1))
    decoded = model.decoder(latent.reshape(captured[0].shape))[..., :17, :18]
    torch.testing.assert_close(decoded, expected)
    assert not latent.requires_grad
    assert all(not p.requires_grad for p in encoder.parameters())


def test_current_next_labels_do_not_leak_target_or_bootstrap():
    history = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.3, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.1, 0.2, 1.0], [0.2, 0.0, 1.0]],
        ]
    )
    successor = torch.tensor([[0.4, 0.5, 1.0], [0.3, 0.4, 1.0], [0.3, 0.5, 1.0]])
    positions, valid = targets(history, successor)
    torch.testing.assert_close(positions[0], torch.tensor([[0.2, 0.3], [0.4, 0.5]]))
    assert valid.tolist() == [[True, True], [False, True], [False, True]]


def test_error_units_and_r_squared():
    score = Scores()
    y = np.array([[0.2, 0.3], [0.4, 0.6]])
    score.add(y + np.array([2 / 160, 3 / 255]), y)
    result = score.result()
    np.testing.assert_allclose(result["mae_xy_native"], [2, 3])
    np.testing.assert_allclose(result["rmse_xy_native"], [2, 3])
    assert result["within_5_native"] == 1
    assert result["within_2_native"] == 0


def test_recorded_successor_labels_attach_to_frame_after_source(tmp_path):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from gymemu.data import Episode
    from probe_ball_latents import attach_labels

    def record(x, y):
        tree = [
            "dict",
            [
                [
                    "labels",
                    [
                        "dict",
                        [
                            ["ball_x_normalized", ["scalar", x]],
                            ["ball_y_normalized", ["scalar", y]],
                        ],
                    ],
                ]
            ],
        ]
        return json.dumps({"structure": json.dumps(tree)})

    folder = tmp_path / "transitions/train"
    folder.mkdir(parents=True)
    # Deliberately shuffled rows and noncontiguous image IDs.
    pq.write_table(
        pa.table(
            {
                "episode_id": [7, 7],
                "step": [1, 0],
                "successor_frame_id": [903, 42],
                "record_json": [record(0.6, 0.7), record(0.2, 0.3)],
            }
        ),
        folder / "part.parquet",
    )
    episode = Episode(7, np.array([101, 42, 903]), np.array([0, 1]))
    attach_labels(tmp_path, "train", [episode], tmp_path / "labels")
    np.testing.assert_allclose(episode.states, [[0, 0, 0], [0.2, 0.3, 1], [0.6, 0.7, 1]])
