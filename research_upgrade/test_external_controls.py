import unittest
import numpy as np
from research_upgrade.external_controls import PaidEntropy,trajectory,score_trajectory,build_controls
from research_upgrade.external_scope import QUERY,SCORE,AUC_WEIGHTS

class ExternalControlsTests(unittest.TestCase):
    def test_paid_guard_and_all_masks_charged(self):
        table={(s,j):float(j) for s in range(8) for j in range(3)}
        v=PaidEntropy(table)
        with self.assertRaises(ValueError):v.read(1,0)
        for policy in ['uncertainty','uniform','random']:
            paths,h=trajectory(table,policy);self.assertEqual(paths[-1],7);self.assertEqual(sorted(h),list(QUERY))
            self.assertEqual([s.bit_count() for s in paths],[0,1,2,3])
        self.assertEqual(trajectory(table,'uncertainty')[1],[31,15,0])
        self.assertEqual(trajectory(table,'uniform')[1],[31,0,15])
    def test_unpaid_counterfactual_entropy_does_not_change_first_choice(self):
        table={(s,j):float(j) for s in range(8) for j in range(3)}
        modified={k:(v if k[0]==0 else -100*v) for k,v in table.items()}
        self.assertEqual(trajectory(table,'uncertainty')[1][0],trajectory(modified,'uncertainty')[1][0])
    def test_all_void_never_perfect(self):
        labels={(s,t):dict(dice=None,iou=None) for s in range(8) for t in SCORE}
        rows=score_trajectory([0,1,3,7],labels)
        self.assertTrue(all(r['dice'] is None and r['missing_frames']==3 for r in rows))
        self.assertAlmostEqual(sum(AUC_WEIGHTS),1.)
    def test_seven_independent_charged_replays_and_scoring_isolated(self):
        calls=[]
        class Fake:
            def __init__(self,cfg,jpeg_dir,base):self.base=base;self.prompts=[]
            def correct(self,t,k,m):
                self.prompts.append((t,k,m));calls.append(tuple(x[0] for x in self.prompts));return self.base
        base=np.full((80,1,2,3),.6,np.float16)
        query={t:np.ones((2,3),bool) for t in QUERY};gold={t:np.ones((2,3),bool) for t in SCORE}
        valid={t:np.ones((2,3),bool) for t in SCORE}
        r=build_controls(base,query,gold,valid,{},'unused',Fake)
        self.assertEqual(len(calls),7);self.assertEqual(len(set(calls)),7)
        self.assertTrue(all(set(c)<=set(QUERY) and not set(c)&set(SCORE) for c in calls))
        flipped={t:~g for t,g in gold.items()}
        r2=build_controls(base,query,flipped,valid,{},'unused',Fake)
        self.assertEqual(r['traces'],r2['traces']);self.assertNotEqual(r['evaluation'],r2['evaluation'])

if __name__=='__main__':unittest.main()
