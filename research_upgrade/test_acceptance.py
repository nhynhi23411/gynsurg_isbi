import unittest
import numpy as np
from research_upgrade.acceptance_v2 import PairModel,paired_features,acceptance_decisions,source_auc,fit_subset
from research_upgrade.bank import trajectory,QUERY,SCORE

class AcceptanceTests(unittest.TestCase):
    def test_pair_feature_order(self):
        np.testing.assert_equal(paired_features({'a':3,'b':5},{'a':1,'b':2},['a','b']),[3,5,1,2,2,3])
    def test_decisions_without_label_access(self):
        selector={(k,s,t):.5 for k in range(2) for s in range(8) for t in QUERY}
        paths,history=trajectory(selector,'uncertainty')
        features={(k,a,s,t):{'x':float(s)} for k in range(2) for a in range(8) for s in range(8) if not s&~a for t in SCORE}
        clip=dict(cid='001',source='s1',paths=paths,history=history,features=features,names=['x'])
        class Model:
            def predict(self,X):return np.full(len(X),-.1)
        states,decisions=acceptance_decisions(clip,Model(),0.)
        self.assertEqual(len(decisions),15)
        for n,state in states.items():
            for (k,t),s in state.items():self.assertFalse(s&~paths[n][k])
    def test_source_macro_not_pooled_clip_mean(self):
        rows=[]
        for source,cids,val in [('A',['a','b','c'],.9),('B',['d'],.1)]:
            for cid in cids:
                for n in [0,1,2,5]:rows.append(dict(procedure_id=source,clip_id=cid,n_queries=n,dice=val))
        self.assertAlmostEqual(source_auc(rows).mean(),.5)
    def test_ablation_removes_all_anchor_columns(self):
        rng=np.random.default_rng(1);X=rng.normal(size=(30,3));y=rng.normal(size=30);groups=np.repeat(['a','b','c'],10)
        m=PairModel('ridge','controls').fit(X,y,groups,['new_entropy','new_anchor_support','delta_anchor_reconstruction_mean'])
        self.assertEqual(m.keep,[0])
    def test_outer_source_labels_not_used(self):
        rng=np.random.default_rng(1);X=rng.normal(size=(40,3));y=rng.normal(size=40);groups=np.repeat(['a','b','c','held'],10);cols=['new_entropy','old_entropy','delta_entropy']
        one=fit_subset(X,y,groups,cols,['a','b','c'],'ridge','controls')
        y[groups=='held']=1e9
        two=fit_subset(X,y,groups,cols,['a','b','c'],'ridge','controls')
        np.testing.assert_array_equal(one.predict(X),two.predict(X))
    def test_both_model_families_fit(self):
        rng=np.random.default_rng(2);X=rng.normal(size=(60,3));y=X[:,0]*.1;groups=np.repeat(['a','b','c'],20)
        for family in ['ridge','logistic']:
            m=PairModel(family).fit(X,y,groups,['x','y','z']);self.assertTrue(np.isfinite(m.predict(X)).all())

if __name__=='__main__':unittest.main()
