import unittest
import numpy as np
from research_upgrade.reproducibility_observer_v1 import compare_predictions, fingerprint


class ObserverTests(unittest.TestCase):
    def test_small_numeric_shift_can_change_binary_mask(self):
        a = np.array([.49999, .7], dtype=np.float32)
        b = np.array([.50001, .7], dtype=np.float32)
        r = compare_predictions(a, b)
        self.assertLess(r['maximum_absolute_difference'], .0001)
        self.assertEqual(r['threshold_05_disagreement_fraction'], .5)
        self.assertNotEqual(fingerprint(a)['sha256'], fingerprint(b)['sha256'])

    def test_invalid_arrays_are_not_diagnosed_as_equivalent(self):
        for a, b in [(np.zeros(2), np.zeros(3)), (np.array([np.nan]), np.zeros(1)), (np.array([]), np.array([]))]:
            with self.assertRaises(ValueError):
                compare_predictions(a, b)

    def test_noncontiguous_view_fingerprint_is_stable(self):
        a = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
        self.assertEqual(fingerprint(a), fingerprint(a.copy()))
        self.assertTrue(compare_predictions(a, a.copy())['exact_numeric_equality'])


if __name__ == '__main__':
    unittest.main()
