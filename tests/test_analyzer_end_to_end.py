import unittest,tempfile,subprocess,sys,json,itertools
from pathlib import Path
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1]
class AnalyzerEndToEnd(unittest.TestCase):
    def test_nonuniform_density_auc_and_paired_procedure_ci(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=[]
            for pid,k,j in itertools.product(['proc_a','proc_b','proc_c'],[0,1],range(4)):
                base.append(dict(procedure_id=pid,clip_id=pid+'_clip',policy='uncertainty',seed=1,density=[0,1,2,4][j],actual_density=[0,.8,1.6,4][j],n_queries=[0,1,2,5][j],structure=k,dice=[0,.2,.4,.6][j]+k*.2,iou=.2,n_scored=3,scoring_frame_sha256='same_frame_set'))
            b=pd.DataFrame(base);p=b.copy();p['dice']+=.1;p['policy']='full'
            b.to_csv(root/'b.csv',index=False);p.to_csv(root/'p.csv',index=False)
            result=subprocess.run([sys.executable,'-m','src.analyze','--kind','surgery','--proposed',str(root/'p.csv'),'--baseline',str(root/'b.csv'),'--expected-seeds','1','--output',str(root/'result')],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            table=pd.read_csv(root/'result/per_procedure_auc.csv');self.assertTrue(np.allclose(table.baseline_auc,.48));self.assertTrue(np.allclose(table.proposed_auc,.58))
            stats=json.loads((root/'result/statistics.json').read_text());self.assertAlmostEqual(stats['delta'],.1);self.assertTrue(np.allclose(stats['ci95'],[.1,.1]));self.assertEqual(stats['n_groups'],3)
if __name__=='__main__':unittest.main()
