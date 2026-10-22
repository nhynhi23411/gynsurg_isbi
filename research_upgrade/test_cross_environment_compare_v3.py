import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from research_upgrade.cross_environment_compare_v3 import analyze


def fixture(folder, gpu, value, complete=True):
    folder.mkdir(parents=True)
    cell = dict(id=folder.name, clip_id='003', mode='adaptive', repeat=0, expected_gpu=gpu)
    if not complete:
        (folder / 'execution.json').write_text(json.dumps(dict(status='hardware_mismatch_no_inference', cell=cell)))
        return
    for step in range(6):
        np.save(folder / f'state_{step}.npy', np.full((6, 2, 2, 2), .5 if step == 0 else value, dtype=np.float32))
    (folder / 'diagnostics.json').write_text(json.dumps(dict(bought_queries=[[0, 0]] * 5)))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
    (folder / 'execution.json').write_text(json.dumps(dict(status='complete', cell=cell, files=files, input_sha256='synthetic', actual_gpu=gpu)))


class CrossEnvironmentTests(unittest.TestCase):
    def test_only_matched_cells_generate_two_declared_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / 'old' / 'a', 'NVIDIA A10', .499)
            fixture(root / 'new' / 'b', 'NVIDIA A10G', .501)
            result = analyze(root / 'old', root / 'new', root / 'analysis')
            self.assertEqual(result['row_count'], 12)
            self.assertEqual(result['environment_score_step_pairs'], 6)
            self.assertEqual(result['maximum_threshold_flip_fraction'], 1.)
            self.assertFalse(result['hardware_causality_established'])

    def test_missing_new_environment_is_not_zero_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / 'old' / 'a', 'NVIDIA A10', .499)
            fixture(root / 'new' / 'b', 'NVIDIA A10G', .501, complete=False)
            result = analyze(root / 'old', root / 'new', root / 'analysis')
            self.assertEqual(result['aws_complete'], 0)
            self.assertIsNone(result['maximum_probability_difference'])
