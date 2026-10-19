import sys,unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rescore_train_expansion_temporal import temporal_auc

class TemporalExpansionTest(unittest.TestCase):
    def test_same_pool_applies_to_baseline_and_all_correction_budgets(self):
        frame=pd.DataFrame([dict(n_queries=q,structure=k,frame_index=f,dice=1. if f==74 else .25) for q in [0,1,2,5] for k in [0,1] for f in [74,99,124]])
        result=temporal_auc(frame,[0,0,24,24,49])
        self.assertEqual(result['common_frames'],[99,124])
        self.assertAlmostEqual(result['baseline_auc'],.25)
        self.assertAlmostEqual(result['raw_auc'],.25)
        self.assertAlmostEqual(result['coverage'],2/3)

    def test_wrong_query_frames_and_missing_scoring_cell_rejected(self):
        frame=pd.DataFrame([dict(n_queries=q,structure=k,frame_index=f,dice=.5) for q in [0,1,2,5] for k in [0,1] for f in [74,99,124]])
        with self.assertRaises(ValueError):temporal_auc(frame,[99])
        with self.assertRaises(ValueError):temporal_auc(frame.iloc[:-1],[0,24,49])
