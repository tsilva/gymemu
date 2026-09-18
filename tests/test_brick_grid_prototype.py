"""Check the prototype's geometry and its small-sprite ambiguity rule."""

import numpy as np

from prototype_brick_grid import calibrate, extract


def wall():
    rgb = np.zeros((210, 160, 3), dtype=np.uint8)
    colors = [
        (200, 72, 72),
        (192, 104, 56),
        (176, 120, 48),
        (160, 160, 40),
        (72, 160, 72),
        (64, 72, 200),
    ]
    for row, color in enumerate(colors):
        rgb[57 + row * 6 : 63 + row * 6, 8:152] = color
    return rgb


def test_geometry_and_independent_corner_cells():
    rgb = wall()
    geometry = calibrate(rgb)
    assert (geometry["x0"], geometry["x1"], geometry["columns"]) == (8, 152, 18)
    rgb[57:63, 8:16] = 0
    rgb[87:93, 144:152] = 0
    grid, _ = extract(rgb, geometry)
    expected = np.ones((6, 18), dtype=np.int8)
    expected[0, 0] = expected[5, 17] = 0
    np.testing.assert_array_equal(grid, expected)


def test_row_colored_ball_does_not_become_a_brick():
    rgb = wall()
    geometry = calibrate(rgb)
    rgb[87:93, 8:24] = 0
    # Ball spans the boundary of two missing blue cells.
    rgb[88:92, 15:17] = geometry["bands"][5]["color"]
    grid, _ = extract(rgb, geometry)
    assert grid[5, 0] == grid[5, 1] == 0
    assert grid.sum() == 106


def test_large_partial_patch_stays_unknown():
    rgb = wall()
    geometry = calibrate(rgb)
    rgb[57:63, 8:16] = 0
    rgb[58:62, 9:13] = geometry["bands"][0]["color"]
    grid, _ = extract(rgb, geometry)
    assert grid[0, 0] == -1


def test_small_occlusion_preserves_present_brick():
    rgb = wall()
    geometry = calibrate(rgb)
    rgb[58:62, 10:12] = 0
    grid, support = extract(rgb, geometry)
    assert grid[0, 0] == 1
    np.testing.assert_allclose(support[0, 0], 40 / 48)
