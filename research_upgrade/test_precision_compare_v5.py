import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from research_upgrade.precision_compare_v5 import analyze
from research_upgrade.reproducibility_observer_v1 import fingerprint


class PrecisionComparisonTests(unittest.TestCase):
    def test_paired_precision_and_repeat_are_separate_with_fixed_gold_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'source'; source.mkdir()
            (root/'gold/132').mkdir(parents=True)
            np.savez(root/'gold/132/sequence.npz', gold=np.zeros((125,2,2,2),bool))
            sha=hashlib.sha256((root/'gold/132/sequence.npz').read_bytes()).hexdigest()
            cells=[]
            for repeat in [0,1]:
                for precision in ['bfloat16','float32']:
                    cell=dict(id=f'{precision}_{repeat}',clip_id='132',pair_id=f'pair{repeat}',mode='fixed',repeat=repeat,precision=precision)
                    cells.append(cell); folder=source/cell['id']; folder.mkdir()
                    snapshots=[]
                    for step in range(6):
                        data=np.full((6,2,2,2),.4 if step==0 else (.49 if precision=='bfloat16' else .51),np.float32)
                        np.save(folder/f'state_{step}.npy',data)
                        snapshots.append(dict(step=step,**fingerprint(data)))
                    queries=[[0,0],[24,0],[49,0],[0,1],[24,1]]
                    metrics=[dict(budget=b,dice=float(b==0 or precision=='bfloat16'),iou=float(b==0 or precision=='bfloat16')) for b in [0,1,2,5]]
                    (folder/'diagnostics.json').write_text(json.dumps(dict(snapshots=snapshots,bought_queries=queries,metrics=metrics)))
                    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
                    record=dict(cell=cell,status='complete',actual_gpu='NVIDIA A10',device_report='synthetic',input_sha256=sha,paid_masks=5,sam_precision=precision,files=files,environment_after_engine_init=dict(tf32_matmul=False,tf32_cudnn=False))
                    (folder/'execution.json').write_text(json.dumps(record))
            result=analyze(source,cells,root/'analysis',root/'gold')
            rows=pd.read_csv(root/'analysis/paired_steps.csv')
            self.assertEqual(result['completed_arms'],4)
            self.assertEqual(len(rows),48)
            precision=rows[rows.factor.eq('within_device_precision')&rows.step.gt(0)]
            self.assertTrue(precision.same_physical_device.all())
            self.assertTrue(np.allclose(precision.maximum_absolute_difference,.02))
            self.assertTrue(precision.threshold_05_disagreement_fraction.eq(1).all())
            self.assertTrue(rows[rows.factor.eq('same_precision_repeat')].exact_numeric_equality.all())
