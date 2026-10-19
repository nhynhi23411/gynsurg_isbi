import sys,unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_full_training_oracle import oracle_auc

class FullTrainingOracleTest(unittest.TestCase):
    def test_gold_choice_before_macro_not_after_macro(self):
        rows=[]
        for q in [0,1,2,5]:
            for k in [0,1]:
                for f in [74,99,124]:
                    value=(1-k) if q==0 else k
                    rows.append(dict(n_queries=q,structure=k,frame_index=f,dice=float(value)))
        result=oracle_auc(pd.DataFrame(rows))
        self.assertAlmostEqual(result['raw_auc'],.5)
        self.assertAlmostEqual(result['baseline_auc'],.5)
        self.assertAlmostEqual(result['oracle_auc'],.95)

    def test_incomplete_grid_rejected(self):
        rows=[dict(n_queries=q,structure=k,frame_index=f,dice=.5) for q in [0,1,2,5] for k in [0,1] for f in [74,99,124]]
        with self.assertRaises(ValueError):oracle_auc(pd.DataFrame(rows[:-1]))
