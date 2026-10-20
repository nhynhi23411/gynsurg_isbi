import unittest
import numpy as np
from research_upgrade.finite_sequence_diagnostic import SEQUENCES,charged_states,exhaustive
from research_upgrade.bank import SCORE

class FiniteSequenceTests(unittest.TestCase):
    def test_all_720_trajectories_charge_exact_counts(self):
        self.assertEqual(len(SEQUENCES),720);self.assertEqual(len(set(SEQUENCES)),720)
        for s in SEQUENCES:
            self.assertEqual([sum(x.bit_count() for x in state) for state in charged_states(s)],[0,1,2,5])
        with self.assertRaises(ValueError):charged_states([0,0,1,2,3])
    def test_known_optimum_uses_valuable_head_first_without_changing_labels(self):
        labels={(k,s,t):(1. if k==0 and s else 0.) for k in range(2) for s in range(8) for t in SCORE}
        before=dict(labels);r=exhaustive(labels,[1,3,5,0,2])
        self.assertAlmostEqual(r['best_auc'],.45);self.assertAlmostEqual(r['worst_auc'],.15)
        self.assertAlmostEqual(r['raw_auc'],.15);self.assertEqual(before,labels)
        self.assertFalse(r['deployable']);self.assertTrue(r['evaluator_gold_used'])

if __name__=='__main__':unittest.main()
