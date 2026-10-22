import unittest
import numpy as np
from research_upgrade.metric_decomposition_v3 import score


class EmptyReferenceTests(unittest.TestCase):
    def test_one_pixel_crossing_threshold_changes_empty_dice_by_one(self):
        a = np.full((20, 20), .1)
        b = a.copy(); a[0, 0] = .49999; b[0, 0] = .50001
        gold = np.zeros((20, 20), bool)
        self.assertEqual(score(a, gold)['dice'], 1.)
        self.assertEqual(score(b, gold)['dice'], 0.)
        self.assertEqual(score(b, gold)['predicted_pixels'], 1)
        self.assertFalse(score(b, gold)['reference_present'])
