import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from research_upgrade.cross_environment_compare_v4 import analyze
from research_upgrade.reproducibility_observer_v1 import fingerprint


def fixture(folder, gpu, value):
    folder.mkdir(parents=True)
    snapshots = []
    for step in range(6):
        state = np.full((6, 2, 2, 2), .5 if step == 0 else value, dtype=np.float32)
        np.save(folder / f'state_{step}.npy', state)
        snapshots.append(dict(step=step, **fingerprint(state)))
    (folder / 'diagnostics.json').write_text(json.dumps(dict(bought_queries=[[0, 0]]*5, snapshots=snapshots)))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
    cell = dict(id=folder.name, clip_id='003', mode='adaptive', repeat=0, expected_gpu=None)
    (folder / 'execution.json').write_text(json.dumps(dict(cell=cell, actual_gpu=gpu, status='complete', input_sha256='synthetic', files=files)))


class AllocatorComparisonTests(unittest.TestCase):
    def test_same_class_and_cross_class_are_separate_factors(self):
        for current_gpu, expected_factor in [('NVIDIA A10', 'additional_repeat'), ('NVIDIA A10G', 'observed_hardware_environment')]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture(root/'old/a', 'NVIDIA A10', .499)
                fixture(root/'new/b', current_gpu, .501)
                summary = analyze(root/'old', root/'new', root/'analysis')
                rows = pd.read_csv(root/'analysis/paired_steps.csv')
                self.assertEqual(set(rows.factor), {expected_factor})
                self.assertEqual(summary['allocator_complete'], 1)
                self.assertFalse(summary['hardware_causality_established'])
                self.assertEqual(len(pd.read_csv(root/'analysis/snapshot_fingerprints.csv')), 24)
