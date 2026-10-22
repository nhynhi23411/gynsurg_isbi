import unittest
import numpy as np
from research_upgrade import online_replay
from research_upgrade.instrumented_replay_v1 import run_instrumented


class Engine:
    def __init__(self, baseline):
        self.baseline = baseline
        self.prompts = []

    def correct(self, t, k, mask):
        self.prompts.append((t, k, mask.copy()))
        p = self.baseline.copy()
        p[:, k] = .15 + .03 * len(self.prompts)
        for frame, head, m in self.prompts:
            p[frame, head] = m
        return p


class InstrumentationTests(unittest.TestCase):
    def test_observer_leaves_adaptive_queries_and_outputs_unchanged(self):
        baseline = np.random.default_rng(12).uniform(.1, .9, (125, 2, 3, 4))
        purchase = lambda t, k: np.ones((3, 4), dtype=bool)
        reference = online_replay.run(baseline, Engine(baseline), purchase)
        snapshots = []
        actual = run_instrumented(baseline, Engine(baseline), purchase, lambda n, p: snapshots.append((n, p)))
        self.assertEqual(actual['queries'], reference['queries'])
        self.assertEqual([n for n, p in snapshots], list(range(6)))
        for budget in reference['predictions']:
            np.testing.assert_array_equal(reference['predictions'][budget], actual['predictions'][budget])
        self.assertEqual(len(actual['propagations']), 5)

    def test_fixed_replay_is_charged_and_hook_restores_after_observer_error(self):
        p = np.full((125, 2, 2, 2), .6)
        fixed = [(0, 0), (24, 0), (49, 0), (0, 1), (24, 1)]
        bought = []
        def buy(t, k):
            bought.append((t, k))
            return np.ones((2, 2), bool)
        result = run_instrumented(p, Engine(p), buy, lambda n, x: None, fixed)
        self.assertEqual(bought, fixed)
        self.assertEqual(result['paid_masks'], 5)
        original = online_replay.select_next
        def fail(n, x):
            if n == 1:
                raise RuntimeError('Synthetic observer failure')
        with self.assertRaises(RuntimeError):
            run_instrumented(p, Engine(p), buy, fail)
        self.assertIs(online_replay.select_next, original)
        self.assertEqual(run_instrumented(p, Engine(p), buy, lambda n, x: None)['paid_masks'], 5)
