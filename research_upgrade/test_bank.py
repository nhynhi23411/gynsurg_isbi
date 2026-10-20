import tempfile,unittest
from pathlib import Path
import numpy as np
from research_upgrade.bank import PaidView,observable_features,trajectory,FRAMES,QUERY,build_bank
from src.policy import choose

class BankTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(3)
        self.states={(k,s):rng.uniform(.01,.99,(6,5,6)).astype('float32') for k in range(2) for s in range(8)}
        self.anchors={(k,j):rng.random((5,6))>.6 for k in range(2) for j in range(3)}
    def test_history_access_guard(self):
        v=PaidView(self.states,self.anchors,0,1)
        with self.assertRaises(ValueError):v.probability(2,74)
        with self.assertRaises(ValueError):v.mask(1)
    def test_no_scored_reference_feature_input(self):
        values=observable_features(PaidView(self.states,self.anchors,0,3),3,124)
        forbidden={'dice','iou','gold_present','scoring_gold','target_quality'}
        self.assertFalse(forbidden & set(values));self.assertTrue(all(np.isfinite(x) for x in values.values()))
    def test_anchor_self_overwrite_not_used(self):
        v=PaidView(self.states,self.anchors,0,3);before=observable_features(v,3,124)
        # Corrupt conditioned anchor images; LOO must reconstruct with another state.
        self.states[0,3][0]=1-self.states[0,3][0];self.states[0,3][1]=1-self.states[0,3][1]
        after=observable_features(v,3,124)
        self.assertEqual(before['anchor_reconstruction_mean'],after['anchor_reconstruction_mean'])
    def test_single_anchor_explicit_missing(self):
        a=observable_features(PaidView(self.states,self.anchors,0,1),1,124)
        self.assertEqual(a['anchor_support'],0);self.assertEqual(a['anchor_reconstruction_mean'],0)
    def test_random_uniform_exact_original_selection(self):
        selector={(k,s,t):0. for k in range(2) for s in range(8) for t in QUERY}
        p=np.zeros((125,2,1,1),np.float32);cfg={'organ_ids':[0],'instrument_ids':[1],'sigma':.03,'tau_s':2.,'fps':25.,'alpha':1.,'beta':.25}
        for policy in ['random','uniform']:
            for seed in [100,101,20261017]:
                _,history=trajectory(selector,policy,seed);expected=[];rng=np.random.default_rng(seed)
                for _ in range(5):expected.append(choose(p,p,[(t,k) for t in QUERY for k in range(2)],expected,policy,cfg,rng))
                self.assertEqual(history,expected)
    def test_masks_never_double_charged(self):
        selector={(k,s,t):.5 for k in range(2) for s in range(8) for t in QUERY}
        for policy in ['uncertainty','uniform','random']:
            paths,history=trajectory(selector,policy);self.assertEqual(len(set(history)),5)
            for n,p in paths.items():self.assertEqual(sum(int(s).bit_count() for s in p),n)
    def test_uncharged_anchor_not_read(self):
        before=observable_features(PaidView(self.states,self.anchors,0,3),3,124)
        self.anchors[0,2]=~self.anchors[0,2]
        after=observable_features(PaidView(self.states,self.anchors,0,3),3,124)
        self.assertEqual(before,after)
    def test_synthetic_end_to_end_bank(self):
        class Engine:
            calls=0
            def __init__(self,c,jpeg,p):self.p=p;self.prompts=[]
            def correct(self,t,k,m):
                Engine.calls+=1;self.prompts.append((t,k,m));r=self.p.copy()
                for t,k,m in self.prompts:r[:,k]=m
                return r
        p=np.full((125,2,5,6),.2,np.float32);g=np.zeros(p.shape,bool)
        for t in QUERY:g[t,:,1:3,1:4]=True
        d=dict(probabilities=p,framewise=p.copy(),gold=g,query_ids=np.array(QUERY),scoring_ids=np.array([74,99,124]),fps=25.,jpeg_dir='',clip_id='001',procedure_id='source01')
        with tempfile.TemporaryDirectory() as out:
            r=build_bank(d,{},out,engine_factory=Engine)
            self.assertEqual(Engine.calls,14);self.assertEqual(r['feature_rows'],162)
            self.assertTrue((Path(out)/'evaluation_labels.csv').exists())

if __name__=='__main__':unittest.main()
