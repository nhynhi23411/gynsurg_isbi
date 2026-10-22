"""Private snapshot comparisons. Missing cells stay missing; no reference labels."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import numpy as np
from research_upgrade.reproducibility_observer_v1 import compare_predictions


def analyze(root, cells):
    root = Path(root)
    records = {c['id']: json.loads((root / c['id'] / 'execution.json').read_text()) for c in cells}
    completed = {key: record for key, record in records.items() if record['status'] == 'complete'}
    diagnostics = {}
    for key, record in completed.items():
        for name, expected in record['files'].items():
            with (root / key / name).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError('Snapshot/file identity mismatch')
        diagnostics[key] = json.loads((root / key / 'diagnostics.json').read_text())
    rows = []
    for a, b in itertools.combinations(completed, 2):
        ca, cb = completed[a]['cell'], completed[b]['cell']
        if ca['clip_id'] != cb['clip_id']:
            continue
        differing = [name for name in ['expected_gpu', 'mode', 'repeat'] if ca[name] != cb[name]]
        if len(differing) != 1:
            continue
        factor = differing[0]
        for step in range(6):
            pa = np.load(root / a / f'state_{step}.npy', mmap_mode='r', allow_pickle=False)
            pb = np.load(root / b / f'state_{step}.npy', mmap_mode='r', allow_pickle=False)
            qa = diagnostics[a]['bought_queries'][:step]
            qb = diagnostics[b]['bought_queries'][:step]
            rows.append(dict(clip_id=ca['clip_id'], cell_a=a, cell_b=b, factor=factor, step=step,
                             paid_query_prefix_identical=qa == qb, **compare_predictions(pa, pb)))
    summary = dict(submitted_cells=len(cells), completed_cells=len(completed),
                   hardware_mismatch_cells=[key for key, record in records.items() if record['status'] != 'complete'],
                   comparisons=len(rows), exact_step_pairs=sum(r['exact_numeric_equality'] for r in rows),
                   maximum_probability_difference=max((r['maximum_absolute_difference'] for r in rows), default=None),
                   maximum_threshold_flip_fraction=max((r['threshold_05_disagreement_fraction'] for r in rows), default=None),
                   original_primary_gate_modified=False, final_test=False, hardware_causality_established=False)
    destination = root / 'analysis'
    destination.mkdir(exist_ok=False)
    if rows:
        with (destination / 'paired_steps.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        (destination / 'paired_steps.csv').write_text('clip_id,cell_a,cell_b,factor,step\n')
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary
