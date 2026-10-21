import unittest
import numpy as np
from research_upgrade.completion_ablations_v1 import keep_columns, decide, evaluate
from research_upgrade.bank import SCORE


class DiagnosticTests(unittest.TestCase):
    def test_group_projection_excludes_unselected_and_gold(self):
        cols = ['new_entropy', 'old_entropy', 'delta_entropy', 'new_anchor_support',
                'new_nearest_paid_seconds', 'new_target_uterus']
        self.assertEqual(keep_columns(cols, 'entropy'), [0, 1, 2, 5])
        self.assertEqual(keep_columns(cols, 'temporal'), [4, 5])
        self.assertEqual(keep_columns(cols, 'observable'), [0, 1, 2, 4, 5])
        with self.assertRaises(ValueError):
            keep_columns(cols, 'unregistered')

    def test_reject_fallback_cannot_access_scoring_labels(self):
        class Reject:
            def predict(self, x): return np.full(len(x), -1.)
        clip = dict(cid='x', source='s', history=[(0, 0), (24, 0)],
                    paths={1: (1, 0), 2: (3, 0)}, names=['entropy'],
                    features={(0, 3, state, t): {'entropy': .5} for state in [0, 1, 3] for t in SCORE})
        a, _ = decide(clip, Reject(), 'previous')
        b, _ = decide(clip, Reject(), 'baseline')
        self.assertTrue(all(a[2][0, t] == 1 for t in SCORE))
        self.assertTrue(all(b[2][0, t] == 0 for t in SCORE))
        self.assertNotIn('labels', clip)

    def test_common_pool_scoring_ignores_excluded_snapshot(self):
        clip = {'cid': 'x', 'source': 's', 'labels': {(k, 0, t): {'dice': t / 124, 'iou': t / 124}
                                                    for k in range(2) for t in SCORE}}
        states = {n: {(k, t): 0 for k in range(2) for t in SCORE} for n in [0, 1, 2, 5]}
        self.assertAlmostEqual(evaluate(clip, states, (124,))[0]['dice'], 1.)


if __name__ == '__main__':
    unittest.main()
