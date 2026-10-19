import sys,unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rescore_train_expansion_present_only import present_auc

class PresentOnlyExpansionTest(unittest.TestCase):
    def data(self):
        return pd.DataFrame([dict(n_queries=q,structure=k,frame_index=f,gold_present=(k==1 or f==74),dice=.2 if k==0 else .8) for q in [0,1,2,5] for k in [0,1] for f in [74,99,124]])

    def test_equal_head_weight_after_present_frame_mean(self):
        result=present_auc(self.data())
        self.assertAlmostEqual(result['raw_auc'],.5)
        self.assertAlmostEqual(result['baseline_auc'],.5)
        self.assertEqual(result['present_cells'],4)
        self.assertEqual(result['present_heads'],2)

    def test_empty_coverage_and_budget_dependent_pool(self):
        data=self.data();data['gold_present']=False
        self.assertIsNone(present_auc(data))
        data=self.data();data.loc[(data.n_queries==5)&(data.structure==0)&(data.frame_index==74),'gold_present']=False
        with self.assertRaises(ValueError):present_auc(data)
