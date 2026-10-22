"""Posthoc technical environment contrasts; immutable private prediction files."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import numpy as np
from research_upgrade.reproducibility_observer_v1 import compare_predictions, fingerprint


def analyze(reference_root, current_root, destination, gold_root=None):
    roots = {'reference': Path(reference_root), 'allocator': Path(current_root)}
    complete, folders, diagnostics = {}, {}, {}
    for origin, root in roots.items():
        for file in sorted(root.glob('*/execution.json')):
            record = json.loads(file.read_text())
            if record['status'] != 'complete':
                continue
            key = origin + ':' + record['cell']['id']
            for name, expected in record['files'].items():
                with (file.parent / name).open('rb') as stream:
                    assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
            complete[key], folders[key] = record, file.parent
            diagnostics[key] = json.loads((file.parent / 'diagnostics.json').read_text())
    fingerprints = []
    for key in complete:
        for step in range(6):
            data = np.load(folders[key] / f'state_{step}.npy', mmap_mode='r', allow_pickle=False)
            full = fingerprint(data)
            assert full['sha256'] == diagnostics[key]['snapshots'][step]['sha256']
            for scope, array in [('query_and_score', data), ('score_only', data[3:])]:
                f = fingerprint(array)
                fingerprints.append(dict(cell=key, step=step, scope=scope, sha256=f['sha256'], dtype=f['dtype'], shape='x'.join(map(str, f['shape']))))
    rows = []
    for a, b in itertools.combinations(complete, 2):
        ra, rb = complete[a], complete[b]
        ca, cb = ra['cell'], rb['cell']
        if ca['clip_id'] != cb['clip_id']:
            continue
        cross = a.split(':')[0] != b.split(':')[0]
        if cross:
            if ca['mode'] != cb['mode'] or ca['repeat'] != cb['repeat']:
                continue
            factor = 'observed_hardware_environment' if ra['actual_gpu'] != rb['actual_gpu'] else 'additional_repeat'
        else:
            # Old pairs have their own preserved v2 analysis. Recompute only new.
            if a.startswith('reference:'):
                continue
            factors = [f for f in ['mode', 'repeat'] if ca[f] != cb[f]]
            if len(factors) != 1:
                continue
            factor = factors[0] + ('_mixed_hardware' if ra['actual_gpu'] != rb['actual_gpu'] else '')
        assert ra['input_sha256'] == rb['input_sha256']
        for step in range(6):
            pa = np.load(folders[a] / f'state_{step}.npy', mmap_mode='r', allow_pickle=False)
            pb = np.load(folders[b] / f'state_{step}.npy', mmap_mode='r', allow_pickle=False)
            assert pa.shape == pb.shape and pa.dtype == pb.dtype
            prefix = diagnostics[a]['bought_queries'][:step] == diagnostics[b]['bought_queries'][:step]
            for scope, left, right in [('query_and_score', pa, pb), ('score_only', pa[3:], pb[3:])]:
                rows.append(dict(clip_id=ca['clip_id'], cell_a=a, cell_b=b, factor=factor, mode_a=ca['mode'], mode_b=cb['mode'],
                    repeat_a=ca['repeat'], repeat_b=cb['repeat'], actual_gpu_a=ra['actual_gpu'], actual_gpu_b=rb['actual_gpu'],
                    step=step, scope=scope, paid_query_prefix_identical=prefix, **compare_predictions(left, right)))
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    if fingerprints:
        with (destination / 'snapshot_fingerprints.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fingerprints[0]))
            writer.writeheader(); writer.writerows(fingerprints)
    else:
        (destination / 'snapshot_fingerprints.csv').write_text('cell,step,scope,sha256,dtype,shape\n')
    if rows:
        with (destination / 'paired_steps.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    else:
        (destination / 'paired_steps.csv').write_text('clip_id,cell_a,cell_b,factor,step,scope\n')
    summary = dict(reference_complete=sum(k.startswith('reference:') for k in complete), allocator_complete=sum(k.startswith('allocator:') for k in complete),
        row_count=len(rows), environment_score_step_pairs=sum(r['factor'] == 'observed_hardware_environment' and r['scope'] == 'score_only' for r in rows),
        maximum_probability_difference=max((r['maximum_absolute_difference'] for r in rows), default=None),
        maximum_threshold_flip_fraction=max((r['threshold_05_disagreement_fraction'] for r in rows), default=None),
        cloud_constrained_extension=False, allocator_observed_replication=True, hardware_causality_established=False, original_primary_gate_modified=False, final_test=False)
    if gold_root is not None:
        from research_upgrade.metric_decomposition_v3 import evaluate
        summary['reference_decomposition'] = evaluate(complete, folders, diagnostics, gold_root, destination)
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary
