import numpy as np

from probe_paddle_history import examples, extract_paddle


def test_rgb_extractor_uses_visible_left_edge():
    rgb = np.zeros((3, 210, 160), dtype=np.uint8)
    rgb[:, 189:193, 35:51] = np.array([200, 72, 72])[:, None, None]
    assert extract_paddle(rgb) == [35, 16, 1]
    rgb[:, 189:193, 35:51] = 0
    assert extract_paddle(rgb) == [0, 0, 0]


def test_independent_histories_do_not_include_successor_or_cross_episode(tmp_path):
    path = tmp_path / "data.npz"
    data = {}
    for eid, start in [(7, 20), (99, 100)]:
        data[f"{eid}_rgb"] = np.array([[start + i, 16, 1] for i in range(5)], np.float32)
        data[f"{eid}_labels"] = np.array([[start + i, i] for i in range(5)], np.float32)
        data[f"{eid}_actions"] = np.array([0, 1, 2, 0])
    np.savez(path, **data)
    x, y, base, ids = examples(path, 2, 4, 0, 1)
    np.testing.assert_array_equal(ids, [[7, 2], [7, 3], [7, 4], [99, 2], [99, 3], [99, 4]])
    np.testing.assert_array_equal(base, [21, 22, 23, 101, 102, 103])
    np.testing.assert_array_equal(y[0], [22, 2])
    np.testing.assert_allclose(x[0, :2], [-0.1, 0])
    # Four action one-hots after seven visual features: PAD, PAD, button, right.
    np.testing.assert_array_equal(x[0, 7:].reshape(4, 4).argmax(1), [0, 0, 1, 2])
    np.testing.assert_array_equal(x[3, 7:].reshape(4, 4).argmax(1), [0, 0, 1, 2])
    cx, cy, cbase, _ = examples(path, 1, 1, 0, 1, current=True)
    assert cbase[0] == cy[0, 0] == 22
    assert cx[0, -4:].argmax() == 2


def test_right_edge_decoration_does_not_hide_paddle():
    rgb = np.zeros((3, 210, 160), dtype=np.uint8)
    rgb[:, 189:193, 137:160] = np.array([200, 72, 72])[:, None, None]
    assert extract_paddle(rgb) == [137, 15, 1]
