import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from oracle_diagnostic_metrics import observable,outcome

class OracleDiagnosticTest(unittest.TestCase):
    def test_rank_auc_ties_and_one_class(self):
        from analyze_oracle_diagnostics import rank_auc
        self.assertEqual(rank_auc([0,1],[False,True]),1)
        self.assertEqual(rank_auc([1,0],[False,True]),0)
        self.assertEqual(rank_auc([1,1,1,1],[False,False,True,True]),.5)
        self.assertIsNone(rank_auc([0,1],[True,True]))
    def test_oracle_chooses_per_output_not_after_macro(self):
        gold=np.array([[True,False]]);good=np.array([[1.,0.]]);bad=1-good
        a=outcome(good,bad,gold);b=outcome(bad,good,gold)
        self.assertEqual((a['raw_dice']+b['raw_dice'])/2,.5)
        self.assertEqual((a['restricted_oracle_dice']+b['restricted_oracle_dice'])/2,1)

    def test_feature_api_excludes_gold_and_handles_empty(self):
        z=np.zeros((2,3));r=observable(z,z,z,z,None)
        self.assertEqual(r['sparse_temporal_iou'],1);self.assertEqual(r['roi_disagreement'],0)
        self.assertTrue(np.isfinite(r['sam_entropy']))
        with self.assertRaises(TypeError):observable(z,z,z,z,None,gold=z)
