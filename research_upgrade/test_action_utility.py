import unittest
import numpy as np
from research_upgrade.action_utility import features,UtilityModel,choose_paths,STATIC,DYNAMIC
from research_upgrade.bank import QUERY

class UtilityTests(unittest.TestCase):
    def setUp(self):self.selector={(k,s,t):.2+.01*s+.001*t for k in range(2) for s in range(8) for t in QUERY}
    def test_only_current_or_baseline_observables(self):
        a=features(self.selector,1,1,2);other=dict(self.selector)
        for key in other:
            if key[1] not in [0,1]:other[key]=999.
        self.assertEqual(a,features(other,1,1,2))
        self.assertFalse({'dice','iou','gold_present'}&set(a));self.assertEqual(list(a),STATIC+DYNAMIC)
        with self.assertRaises(ValueError):features(self.selector,0,1,0)
    def test_accounting_and_deterministic_ties(self):
        class Zero:
            names=STATIC
            def predict(self,X):return np.zeros(len(X))
        paths,history=choose_paths(self.selector,Zero())
        self.assertEqual(len(set(history)),5);self.assertEqual(history[0],(0,0))
        for n,paid in paths.items():self.assertEqual(sum(p.bit_count() for p in paid),n)
        self.assertFalse(set(t for t,k in history)&{74,99,124})
    def test_weighted_constant_normalization_finite(self):
        X=np.ones((20,len(STATIC)));X[:,0]=np.arange(20);y=np.arange(20)/20;groups=np.repeat(['a','b'],10)
        model=UtilityModel('static').fit(X,y,groups)
        self.assertTrue(np.isfinite(model.predict(X)).all());self.assertTrue((model.scale>0).all())
    def test_held_source_labels_cannot_affect_fit(self):
        rng=np.random.default_rng(2);X=rng.normal(size=(30,len(STATIC)));y=rng.normal(size=30);groups=np.repeat(['a','b','held'],10);train=groups!='held'
        a=UtilityModel('static').fit(X[train],y[train],groups[train]);y[~train]=1e12
        b=UtilityModel('static').fit(X[train],y[train],groups[train]);np.testing.assert_array_equal(a.predict(X),b.predict(X))

if __name__=='__main__':unittest.main()
