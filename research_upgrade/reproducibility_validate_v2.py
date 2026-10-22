"""Independent local metadata/pair coverage audit after private CPU comparisons."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path


def check_pair(row, a, b):
    step = int(row['step'])
    pa, pb = a['snapshots'][step], b['snapshots'][step]
    same = pa['sha256'] == pb['sha256']
    assert (row['exact_numeric_equality'] == 'True') == same
    maximum, mean, flips = (float(row[k]) for k in ['maximum_absolute_difference', 'mean_absolute_difference', 'threshold_05_disagreement_fraction'])
    assert all(math.isfinite(v) for v in [maximum, mean, flips])
    assert 0 <= mean <= maximum <= 1 and 0 <= flips <= 1
    if same:
        assert maximum == mean == flips == 0
    assert (row['paid_query_prefix_identical'] == 'True') == (a['bought_queries'][:step] == b['bought_queries'][:step])


def validate(root):
    root = Path(root)
    disposition = json.loads((root / 'cell_disposition.json').read_text())
    assert len(disposition) == 48 and len({c['id'] for c in disposition}) == 48
    records = {p.parent.name: json.loads(p.read_text()) for p in sorted((root / 'cells').glob('*/execution.json'))}
    complete, diagnostics = {}, {}
    for key, record in records.items():
        assert record['cell']['id'] == key and not record['final_test']
        assert not record['primary_historical_gate_modified']
        if record['status'] == 'hardware_mismatch_no_inference':
            assert record['actual_gpu'] != record['cell']['expected_gpu'] and not record['files']
            continue
        assert record['status'] == 'complete' and record['actual_gpu'] == record['cell']['expected_gpu']
        file = root / 'cells' / key / 'diagnostics.json'
        assert hashlib.sha256(file.read_bytes()).hexdigest() == record['files']['diagnostics.json']
        diag = json.loads(file.read_text())
        assert len(diag['snapshots']) == 6 and len(diag['queries']) == len(diag['bought_queries']) == 5
        assert len(set(map(tuple, diag['bought_queries']))) == record['paid_masks'] == 5
        assert all(q[0] in [0, 24, 49] and q[1] in [0, 1] for q in diag['bought_queries'])
        complete[key], diagnostics[key] = record, diag
    expected = set()
    for a, b in itertools.combinations(complete, 2):
        ca, cb = complete[a]['cell'], complete[b]['cell']
        differing = [f for f in ['expected_gpu', 'mode', 'repeat'] if ca[f] != cb[f]]
        if ca['clip_id'] == cb['clip_id'] and len(differing) == 1:
            expected.update((a, b, differing[0], step) for step in range(6))
    with (root / 'analysis' / 'paired_steps.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    found = set()
    for row in rows:
        key = (row['cell_a'], row['cell_b'], row['factor'], int(row['step']))
        assert key not in found
        found.add(key)
        check_pair(row, diagnostics[row['cell_a']], diagnostics[row['cell_b']])
    assert found == expected, 'Missing or extra paired step comparisons'
    for cell in disposition:
        if cell['id'] in records:
            assert cell['status'] == records[cell['id']]['status']
        else:
            assert cell['status'].startswith('not_dispatched')
    result = dict(passed=True, registered_cells=48, attempted_cells=len(records), completed_cells=len(complete),
                  skipped_hardware_cells=sum(r['status'] == 'hardware_mismatch_no_inference' for r in records.values()),
                  not_dispatched_cells=48-len(records), verified_step_pairs=len(rows),
                  actual_hardware=sorted({r['actual_gpu'] for r in complete.values()}),
                  complete_full_factorial_design=len(complete) == 48, original_primary_gate_modified=False,
                  validation_executed=False, final_test=False, scope='technical integrity only, no efficacy promotion')
    (root / 'independent_integrity.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    validate(parser.parse_args().root)
