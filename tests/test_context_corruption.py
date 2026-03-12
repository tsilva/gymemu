import unittest

import torch

from context_corruption import apply_seed_history_corruption


class SeedHistoryCorruptionTests(unittest.TestCase):
    def test_zero_strength_is_noop(self):
        history = torch.tensor(
            [[[[0.0, 1.0], [1.0, 0.0]]], [[[1.0, 0.0], [0.0, 1.0]]]],
            dtype=torch.float32,
        )
        strengths = torch.zeros(history.size(0), dtype=torch.float32)

        corrupted = apply_seed_history_corruption(
            history,
            strengths,
            max_strength=0.08,
            foreground_dropout_max=0.06,
        )

        self.assertTrue(torch.equal(corrupted, history))

    def test_nonzero_strength_preserves_shape_dtype_and_range(self):
        generator = torch.Generator().manual_seed(123)
        history = torch.ones((2, 4, 8, 8), dtype=torch.float32) * 0.75
        strengths = torch.tensor([0.08, 0.04], dtype=torch.float32)

        corrupted = apply_seed_history_corruption(
            history,
            strengths,
            max_strength=0.08,
            foreground_dropout_max=0.06,
            generator=generator,
        )

        self.assertEqual(corrupted.shape, history.shape)
        self.assertEqual(corrupted.dtype, history.dtype)
        self.assertGreaterEqual(float(corrupted.min().item()), 0.0)
        self.assertLessEqual(float(corrupted.max().item()), 1.0)
        self.assertFalse(torch.equal(corrupted, history))

    def test_foreground_dropout_can_remove_active_pixels(self):
        generator = torch.Generator().manual_seed(0)
        history = torch.ones((1, 2, 4, 4), dtype=torch.float32)
        strengths = torch.tensor([1.0], dtype=torch.float32)

        corrupted = apply_seed_history_corruption(
            history,
            strengths,
            max_strength=1.0,
            foreground_dropout_max=1.0,
            generator=generator,
        )

        self.assertEqual(int(torch.count_nonzero(corrupted)), 0)

    def test_max_strength_reaches_configured_dropout_cap(self):
        generator = torch.Generator().manual_seed(0)
        history = torch.ones((1, 3, 8, 8), dtype=torch.float32)
        strengths = torch.tensor([0.08], dtype=torch.float32)

        corrupted = apply_seed_history_corruption(
            history,
            strengths,
            max_strength=0.08,
            foreground_dropout_max=1.0,
            generator=generator,
        )

        self.assertEqual(int(torch.count_nonzero(corrupted)), 0)

    def test_corruption_is_temporally_consistent_for_identical_frames(self):
        generator = torch.Generator().manual_seed(7)
        frame = torch.tensor(
            [[[0.0, 1.0], [1.0, 0.0]]],
            dtype=torch.float32,
        )
        history = frame.repeat(1, 4, 1, 1)
        strengths = torch.tensor([0.08], dtype=torch.float32)

        corrupted = apply_seed_history_corruption(
            history,
            strengths,
            max_strength=0.08,
            foreground_dropout_max=0.5,
            generator=generator,
        )

        self.assertTrue(torch.equal(corrupted[:, 0], corrupted[:, 1]))
        self.assertTrue(torch.equal(corrupted[:, 1], corrupted[:, 2]))


if __name__ == "__main__":
    unittest.main()
