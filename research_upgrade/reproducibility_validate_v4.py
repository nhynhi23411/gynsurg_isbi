"""Independent local audit of all retained observed-allocator records and pairs."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import math
import numpy as np
import pandas as pd


def validate(root, reference):
    root, reference = Path(root), Path(reference)
    disposition = json.loads((root / 'cell_disposition.json').read_text())
    assert len(disposition) == len({r['id'] for r in disposition}) == 24
    records, diagnostics = {}, {}
    for origin, folder in [('reference', reference), ('allocator', root)]:
        for file in sorted((folder / 'cells').glob('*/execution.json')):
            record = json.loads(file.read_text())
            if record['status'] != 'complete':
                continue
            assert not record['final_test'] and not record['primary_historical_gate_modified']
            assert record['actual_gpu'] in ['NVIDIA A10', 'NVIDIA A10G']
            assert record['paid_masks'] == 5 and record['seed'] == 20261017
            target = file.parent / 'diagnostics.json'
            assert hashlib.sha256(target.read_bytes()).hexdigest() == record['files']['diagnostics.json']
            diag = json.loads(target.read_text())
            assert len(diag['bought_queries']) == len(set(map(tuple, diag['bought_queries']))) == 5
            assert [s['step'] for s in diag['snapshots']] == list(range(6))
            key = origin + ':' + record['cell']['id']
            records[key], diagnostics[key] = record, diag
    fingerprints = pd.read_csv(root / 'analysis/snapshot_fingerprints.csv').set_index(['cell', 'step', 'scope'])
    assert len(fingerprints) == len(records) * 12
    for key, diag in diagnostics.items():
        for snapshot in diag['snapshots']:
            assert fingerprints.loc[(key, snapshot['step'], 'query_and_score'), 'sha256'] == snapshot['sha256']
    pairs = pd.read_csv(root / 'analysis/paired_steps.csv', dtype={'clip_id': str})
    expected = set()
    for a, b in itertools.combinations(records, 2):
        ra, rb = records[a], records[b]; ca, cb = ra['cell'], rb['cell']
        if ca['clip_id'] != cb['clip_id']:
            continue
        cross = a.split(':')[0] != b.split(':')[0]
        if cross:
            if ca['mode'] != cb['mode'] or ca['repeat'] != cb['repeat']:
                continue
            factor = 'observed_hardware_environment' if ra['actual_gpu'] != rb['actual_gpu'] else 'additional_repeat'
        else:
            if a.startswith('reference:'):
                continue
            changing = [name for name in ['mode', 'repeat'] if ca[name] != cb[name]]
            if len(changing) != 1:
                continue
            factor = changing[0] + ('_mixed_hardware' if ra['actual_gpu'] != rb['actual_gpu'] else '')
        assert ra['input_sha256'] == rb['input_sha256']
        expected.update((a, b, factor, step, scope) for step in range(6) for scope in ['query_and_score', 'score_only'])
    actual = set(zip(pairs.cell_a, pairs.cell_b, pairs.factor, pairs.step, pairs.scope))
    assert actual == expected and len(pairs) == len(actual)
    for row in pairs.itertuples(index=False):
        a = fingerprints.loc[(row.cell_a, row.step, row.scope)]
        b = fingerprints.loc[(row.cell_b, row.step, row.scope)]
        assert a['shape'] == b['shape'] and a['dtype'] == b['dtype']
        same = a.sha256 == b.sha256
        assert bool(row.exact_numeric_equality) == same
        values = [row.maximum_absolute_difference, row.mean_absolute_difference, row.threshold_05_disagreement_fraction]
        assert all(math.isfinite(v) for v in values)
        assert 0 <= values[1] <= values[0] <= 1 and 0 <= values[2] <= 1
        if same:
            assert values == [0, 0, 0]
        if row.step == 0:
            assert same
        assert bool(row.paid_query_prefix_identical) == (diagnostics[row.cell_a]['bought_queries'][:row.step] == diagnostics[row.cell_b]['bought_queries'][:row.step])
    scores = pd.read_csv(root / 'analysis/per_head_scores.csv', dtype={'clip_id': str})
    assert len(scores) == len(records) * 36
    assert not scores.duplicated(['cell', 'step', 'frame', 'head']).any()
    for key, diag in diagnostics.items():
        for metric in diag['metrics']:
            group = scores[(scores.cell == key) & (scores.step == metric['budget'])]
            assert len(group) == 6
            assert np.allclose(group[['dice', 'iou']].mean().to_numpy(), [metric['dice'], metric['iou']], atol=1e-12, rtol=0)
    assert (scores.groupby(['clip_id', 'frame', 'head']).reference_pixels.nunique() == 1).all()
    empty = scores[~scores.reference_present]
    assert np.array_equal(empty.dice.to_numpy(), empty.predicted_pixels.eq(0).astype(float).to_numpy())
    decomp = pd.read_csv(root / 'analysis/reference_decomposition.csv', dtype={'clip_id': str})
    indexed = scores.set_index(['cell', 'step', 'frame', 'head'])
    for row in decomp.itertuples(index=False):
        left = indexed.loc[(row.cell_a, row.step, row.frame, row.head)]
        right = indexed.loc[(row.cell_b, row.step, row.frame, row.head)]
        assert np.allclose([row.dice_a, row.dice_b, row.delta_dice, row.delta_iou], [left.dice, right.dice, right.dice-left.dice, right.iou-left.iou], atol=1e-12, rtol=0)
        assert row.predicted_pixels_a == left.predicted_pixels and row.predicted_pixels_b == right.predicted_pixels
    count_new = sum(k.startswith('allocator:') for k in records)
    assert sum(c['status'] == 'complete' for c in disposition) == count_new
    result = dict(passed=True, registered_cells=24, completed_new=count_new, complete_design=count_new == 24,
        completed_reference=len(records)-count_new, verified_pair_scope_steps=len(pairs), per_head_rows=len(scores),
        head_contrasts=len(decomp), allocator_hardware_counts=pd.Series([r['actual_gpu'] for k,r in records.items() if k.startswith('allocator:')]).value_counts().to_dict(),
        original_primary_gate_modified=False, validation_executed=False, final_test=False, hardware_causality_established=False)
    (root / 'independent_integrity.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path); parser.add_argument('reference', type=Path)
    args = parser.parse_args(); validate(args.root, args.reference)
