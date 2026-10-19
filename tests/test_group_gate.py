import sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from group_gate import keep_raw, candidates, macro_auc, select_from_training

class GroupGateTest(unittest.TestCase):
    def test_prediction_only_and_boundary(self):
        rule={'name':'example','distance_s':1.,'entropy':.1}
        np.testing.assert_array_equal(keep_raw([True,True,False],[1,1.01,np.nan],[.1,.1,.9],rule),[True,False,True])
        with self.assertRaises(TypeError):keep_raw([True],[1],[.1],rule,gold=[1])
        with self.assertRaises(ValueError):keep_raw([True],[np.nan],[.1],rule)

    def test_outer_outcome_invariance_and_membership(self):
        scores=pd.DataFrame({'raw':[.5,.4,.3],'gate':[.6,.6,1.]},index=['a','b','held'])
        first=select_from_training(scores,'held')
        scores.loc['held']=[1.,0.]
        self.assertEqual(first,select_from_training(scores,'held'))
        for fold in first[2]:
            self.assertNotIn('held',fold['training_sources'])
            self.assertNotEqual('held',fold['validation_source'])
            self.assertNotIn(fold['validation_source'],fold['training_sources'])

    def test_tie_order_and_macro_not_pooled_budget_mean(self):
        self.assertEqual(len(candidates()),26)
        scores=pd.DataFrame({'raw':[.5,.5,.5],'gate':[.5,.5,1.]},index=['a','b','held'])
        self.assertEqual(select_from_training(scores,'held')[0],'raw')
        rows=pd.DataFrame([dict(query_count=n,frame=f,structure=s) for n in [0,1,2,5] for f in [74,99,124] for s in [0,1]])
        dice=np.where(rows.query_count==5,1.,0.)
        self.assertAlmostEqual(macro_auc(rows,dice),.3)
        with self.assertRaises(ValueError):macro_auc(rows.iloc[:-1],dice[:-1])
