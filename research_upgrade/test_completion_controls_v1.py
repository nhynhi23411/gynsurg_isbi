import unittest
import numpy as np
from research_upgrade.completion_controls_v1 import eva_action, predicted_reset_schedule, run_eva, run_predicted


class ControlTests(unittest.TestCase):
    def test_farthest_uses_same_head_paid_points_and_stable_ties(self):
        emb = [np.array([[0.], [1.], [4.]]), np.array([[0.], [1.], [2.]])]
        p = np.full((125, 2, 1, 1), .5)
        self.assertEqual(eva_action(emb, p, [1, 1]), (2, 0))
        self.assertEqual(eva_action(None, p, [0, 1]), (0, 0))

    def test_reset_requires_four_frames_and_counts_periodic(self):
        p = np.zeros((125, 2, 1, 1)); p[5:8, 0] = 1; p[15:20, 0] = 1
        schedules = predicted_reset_schedule(p)
        self.assertNotIn(5, schedules[0]); self.assertIn(15, schedules[0])
        self.assertEqual(schedules[1], [0, 25, 50, 75, 100])

    def test_adaptations_count_gold_and_predicted_prompts_separately(self):
        class Engine:
            prompts = []
            def correct(self, t, k, mask):
                out = p.copy(); out[:, k] = mask; return out
        p = np.full((125, 2, 2, 2), .5); purchases = []
        def buy(t, k):
            purchases.append((t, k)); return np.ones((2, 2), bool)
        features = lambda current: [np.arange(3)[:, None]] * 2
        result = run_eva(p, Engine(), buy, features)
        self.assertEqual(result['paid_masks'], 5); self.assertEqual(len(set(purchases)), 5)
        self.assertEqual(len(result['propagations']), 5)
        automatic = run_predicted(p, Engine())
        self.assertEqual(automatic['paid_masks'], 0); self.assertEqual(automatic['predicted_prompts'], 10)
        self.assertFalse(automatic['gold_budget_auc_defined'])
        self.assertEqual(set(automatic['predictions']), {0})

    def test_segment_stitch_uses_absolute_frames_and_processes_all_segments(self):
        p = np.full((125, 2, 1, 1), .5)
        starts, cleaned = [], []
        def factory(first, end):
            starts.append((first, end))
            class Segment:
                def correct(self, t, head, mask):
                    self.last_t = t
                    return np.full((end-first, 2, 1, 1), first/125.)
            return Segment(), lambda: cleaned.append(first)
        result = run_predicted(p, None, segment_factory=factory)
        np.testing.assert_allclose(result['predictions'][0][:, 0, 0, 0], [50/125, 75/125, 100/125])
        self.assertEqual(len(starts), 10); self.assertEqual(len(cleaned), 10)
        self.assertEqual(sum(c['processed_frames'] for c in result['propagations']), 250)


if __name__ == '__main__':
    unittest.main()
