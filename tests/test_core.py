import unittest,numpy as np
from src.surgery import run,validate_sequence
from src.policy import choose
class FakeEngine:
    def __init__(self,cfg,jpeg_dir,baseline):self.p=baseline.copy()
    def correct(self,t,k,mask):self.p[t,k]=mask;return self.p.copy()
class Core(unittest.TestCase):
    def data(self):
        p=np.ones((100,2,5,5))*.4
        return dict(probabilities=p,framewise=p.copy(),gold=np.ones_like(p),query_ids=np.arange(0,100,2),scoring_ids=np.arange(1,100,2),fps=25,jpeg_dir='unused',procedure_id='synthetic',clip_id='synthetic')
    def cfg(self):return dict(seed=1,policy='full',densities=[0,1,2,4],organ_ids=[0],instrument_ids=[1],sigma=.03,alpha=1,beta=.25,tau_s=2,fps=25,exclusion_seconds=0)
    def test_no_query_scoring_leakage(self):
        rows,q,p=run(self.data(),self.cfg(),FakeEngine);self.assertEqual(len(q),4);self.assertTrue(all(v['dice']==0 for v in rows));self.assertTrue(all(v['n_scored']==50 for v in rows))
    def test_disjoint_guard(self):
        d=self.data();d['scoring_ids']=np.array([0]);self.assertRaises(ValueError,validate_sequence,d)
    def test_sparse_reference_budget_cap(self):
        d=self.data();d['query_ids']=np.array([0,2,4]);c=self.cfg();c['densities']=[0,1,2,10]
        rows,q,p=run(d,c,FakeEngine)
        self.assertEqual(len(q),6);self.assertTrue(rows[-1]['budget_capped']);self.assertEqual(rows[-1]['actual_density'],6.)
        self.assertEqual(len({r['scoring_frame_sha256'] for r in rows}),1)
    def test_all_policies_unique(self):
        for policy in ['random','uniform','uncertainty','error','error_diversity','error_proximity','full']:
            c=self.cfg();c['policy']=policy;rows,q,p=run(self.data(),c,FakeEngine);self.assertEqual(len({(r['frame'],r['structure']) for r in q}),4)
    def test_paired_statistics(self):
        from src.common import paired_bootstrap
        result=paired_bootstrap([.7,.8,.9],[.6,.7,.8]);self.assertAlmostEqual(result['delta'],.1);self.assertTrue(np.allclose(result['ci95'],[.1,.1]));self.assertEqual(result['n_groups'],3)
    def test_categorical_to_binary_adapter(self):
        import torch
        from src.export_joint import BinaryAdapter
        class FakeJoint(torch.nn.Module):
            def forward(self,x,tasks):return {'instrument':torch.zeros(x.shape[0],8,*x.shape[-2:]),'anatomy':torch.zeros(x.shape[0],4,*x.shape[-2:])}
        x=torch.zeros(1,3,5,7);p=BinaryAdapter(FakeJoint(),[1],False)(x).sigmoid()
        self.assertTrue(torch.allclose(p[:,0],torch.full((1,5,7),.25)));self.assertTrue(torch.allclose(p[:,1],torch.full((1,5,7),.125)))
        p=BinaryAdapter(FakeJoint(),[1,2],True)(x).sigmoid();self.assertTrue(torch.allclose(p[:,1],torch.full((1,5,7),.875)))
        p=BinaryAdapter(FakeJoint(),[1,5,6],False,True)(x).sigmoid();self.assertTrue(torch.allclose(p[:,1],torch.full((1,5,7),.375)))
    def test_verified_pixel_palettes(self):
        from src.labels import decode,PALETTES
        for head,values in PALETTES.items():self.assertTrue(np.array_equal(decode(np.array([values],dtype=np.uint8),head),np.arange(len(values))[None]))
        self.assertRaises(ValueError,decode,np.array([[85]],dtype=np.uint8),'auxiliary')
    def test_autolaparo_rgb_reference_decoder(self):
        from src.autolaparo import decode_targets
        raw=np.array([[0,20,160,180]],dtype=np.uint8);rgb=np.repeat(raw[...,None],3,-1);y=decode_targets(rgb)
        self.assertTrue(np.array_equal(y[0],[[False,False,False,True]]));self.assertTrue(np.array_equal(y[1],[[False,True,True,False]]))
        rgb[0,0,1]=20;self.assertRaises(ValueError,decode_targets,rgb)
        self.assertRaises(ValueError,decode_targets,np.array([[181]],dtype=np.uint8))
    def test_common_temporal_scoring(self):
        from src.scoring import common_scoring
        s=common_scoring([124,149,174,199,224],[0,24,49,74,99],25,1)
        self.assertTrue(np.array_equal(s,[149,174,199,224]))
        self.assertRaises(ValueError,common_scoring,[124],[99],25,1)
if __name__=='__main__':unittest.main()
