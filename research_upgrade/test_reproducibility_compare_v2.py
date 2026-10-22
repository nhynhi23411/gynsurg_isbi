import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from research_upgrade.reproducibility_compare_v2 import analyze


class ComparisonTests(unittest.TestCase):
    def test_private_numeric_pairs_preserve_skips_and_detect_flips(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cells = []
            for repeat in range(3):
                cell = dict(id=f'x{repeat}', clip_id='003', expected_gpu='NVIDIA A10', mode='adaptive', repeat=repeat)
                cells.append(cell)
                folder = root / cell['id']
                folder.mkdir()
                files = {}
                if repeat < 2:
                    for step in range(6):
                        np.save(folder / f'state_{step}.npy', np.full((6, 2, 2, 2), .499 if repeat == 0 else .501))
                    (folder / 'diagnostics.json').write_text(json.dumps(dict(bought_queries=[[0, 0]] * 5)))
                    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
                record = dict(cell=cell, status='complete' if repeat < 2 else 'hardware_mismatch_no_inference', files=files)
                (folder / 'execution.json').write_text(json.dumps(record))
            summary = analyze(root, cells)
            self.assertEqual(summary['completed_cells'], 2)
            self.assertEqual(summary['hardware_mismatch_cells'], ['x2'])
            self.assertEqual(summary['comparisons'], 6)
            self.assertEqual(summary['maximum_threshold_flip_fraction'], 1.)
            self.assertFalse(summary['hardware_causality_established'])
