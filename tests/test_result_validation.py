import unittest,itertools
import pandas as pd,numpy as np
from src.result_validation import validate_metric_table
from src.surgery import run
import test_core
class ResultValidation(unittest.TestCase):
    def table(self):
        return pd.DataFrame([dict(procedure_id='v01',clip_id='001',policy='full',seed=1,density=d,actual_density=d,n_queries=d,structure=k,dice=.6,iou=.5,n_scored=3,scoring_frame_sha256='fixed') for d,k in itertools.product([0,1,2,4],[0,1])])
    def test_complete_grid_and_corruptions(self):
        t=self.table();self.assertEqual(validate_metric_table(t,expected_seeds=[1])['rows'],8)
        for bad in [t.iloc[:-1],pd.concat([t,t.iloc[:1]]),t.assign(dice=np.inf),t.assign(actual_density=np.nan)]:self.assertRaises(ValueError,validate_metric_table,bad)
        self.assertRaises(ValueError,validate_metric_table,t,expected_seeds=[1,2])
        self.assertRaises(ValueError,validate_metric_table,t,cohort={('002','v01')})
    def test_frame_outputs_reproduce_aggregate_and_distance(self):
        helper=test_core.Core();data=helper.data();cfg=helper.cfg();frames=[]
        rows,queries,_=run(data,cfg,test_core.FakeEngine,frame_sink=frames.extend)
        for row in rows:
            part=[f for f in frames if f['density']==row['density'] and f['structure']==row['structure']]
            self.assertEqual(len(part),row['n_scored']);self.assertAlmostEqual(np.mean([f['dice'] for f in part]),row['dice'])
            if row['n_queries']==0:self.assertTrue(all(f['nearest_same_structure_query_s'] is None for f in part))
        self.assertTrue(all('mask_sha256' in q for q in queries))
if __name__=='__main__':unittest.main()
