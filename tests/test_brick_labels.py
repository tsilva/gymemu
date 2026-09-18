import numpy as np

from gymemu.brick_labels import COLORS, Issue, episode_issues, extract_wall_batch
from prototype_brick_grid import calibrate, extract


def image():
    rgb = np.zeros((210, 160, 3), dtype=np.uint8)
    for r, color in enumerate(COLORS):
        rgb[57 + r * 6 : 63 + r * 6, 8:152] = color
    return rgb


def test_vectorized_detector_matches_prototype_with_deletions_and_sprites():
    rng = np.random.default_rng(91)
    geometry = calibrate(image())
    frames = []
    for _ in range(32):
        rgb = image()
        for r, c in zip(*np.where(rng.random((6, 18)) < 0.5), strict=True):
            rgb[57 + r * 6 : 63 + r * 6, 8 + c * 8 : 16 + c * 8] = 0
        x, y = int(rng.integers(8, 150)), int(rng.integers(57, 89))
        for sy in range(y, y + 4):
            rgb[sy, x : x + 2] = COLORS[(sy - 57) // 6]
        frames.append(rgb)
    actual, counts, _ = extract_wall_batch(np.stack(frames)[:, 57:93, 8:152])
    for i, frame in enumerate(frames):
        grid, support = extract(frame, geometry)
        np.testing.assert_array_equal(actual[i], grid)
        np.testing.assert_array_equal(counts[i], np.rint(support * 48))


def test_initial_animation_can_decrease_and_stops_at_full_wall():
    grids = np.zeros((5, 6, 18), dtype=np.int8)
    grids[0].flat[:30] = 1
    grids[1].flat[:10] = 1
    grids[3:] = 1
    grids[4, 5, 2] = 0
    flags, initial = episode_issues(
        grids,
        np.zeros(5, np.uint16),
        np.array([108, 108, 108, 107]),
        np.zeros(4),
        np.array([0, 0, 0, 1]),
    )
    assert initial.tolist() == [True, True, True, False, False]
    assert flags[0] & Issue.NATIVE_COUNT_UNAVAILABLE
    assert flags[1] & Issue.COUNT_MISMATCH
    assert flags[3] == flags[4] == 0


def test_later_wall_reset_is_not_initial_animation():
    grids = np.ones((4, 6, 18), dtype=np.int8)
    grids[:3, 5] = 0
    flags, initial = episode_issues(
        grids,
        np.zeros(4, np.uint16),
        np.array([90, 90, 108]),
        np.array([0, 0, 1]),
        np.array([18, 18, 108]),
    )
    assert not initial.any()
    assert flags[3] == Issue.WALL_RESET_UNVALIDATED


def test_unexplained_cell_swap_flags_both_frames_even_if_count_matches():
    grids = np.ones((3, 6, 18), dtype=np.int8)
    grids[:2, 5, 0] = 0
    grids[2, 5, 1] = 0
    flags, _ = episode_issues(
        grids, np.zeros(3, np.uint16), np.array([107, 107]), np.zeros(2), np.ones(2)
    )
    assert flags[1] == flags[2] == Issue.TEMPORAL_CHANGE


def test_unknown_colors_and_partial_bricks_are_suspect():
    rgb = image()
    rgb[58:62, 10:12] = 0
    rgb[80, 80] = [255, 0, 255]
    _, _, flags = extract_wall_batch(rgb[None, 57:93, 8:152])
    assert flags[0] & Issue.PARTIAL_BRICK
    assert flags[0] & Issue.UNEXPECTED_COLOR
