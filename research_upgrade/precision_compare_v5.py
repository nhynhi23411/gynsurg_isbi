"""CPU-only analysis of every complete precision arm; missing arms stay explicit."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import numpy as np
from research_upgrade.precision_controls_v5 import pair_factor
from research_upgrade.reproducibility_observer_v1 import fingerprint, compare_predictions


def analyze(root, cells, destination, gold_root):
    root, destination = Path(root), Path(destination)
    destination.mkdir(exist_ok=False)
    records, diagnostics, folders, disposition, fingerprints = {}, {}, {}, [], []
    for cell in cells:
        folder = root / cell['id']
        status = 'missing_or_partial'
        if (folder / 'execution.json').exists():
            record = json.loads((folder / 'execution.json').read_text())
            status = record['status']
            if status == 'complete':
                assert record['cell'] == cell
                for name, sha in record['files'].items():
                    with (folder / name).open('rb') as stream:
                        assert hashlib.file_digest(stream, 'sha256').hexdigest() == sha
                flags = record['environment_after_engine_init']
                assert not flags['tf32_matmul'] and not flags['tf32_cudnn']
                assert record['sam_precision'] == cell['precision'] and record['paid_masks'] == 5
                records[cell['id']], folders[cell['id']] = record, folder
                diagnostics[cell['id']] = json.loads((folder / 'diagnostics.json').read_text())
        disposition.append(dict(**cell, status=status))
    for key, folder in folders.items():
        for step in range(6):
            state = np.load(folder / f'state_{step}.npy', mmap_mode='r', allow_pickle=False)
            assert fingerprint(state)['sha256'] == diagnostics[key]['snapshots'][step]['sha256']
            for scope, array in [('query_and_score', state), ('score_only', state[3:])]:
                info = fingerprint(array)
                fingerprints.append(dict(cell=key, step=step, scope=scope, sha256=info['sha256'], dtype=info['dtype'], shape='x'.join(map(str, info['shape']))))
    pairs = []
    for a, b in itertools.combinations(records, 2):
        ra, rb = records[a], records[b]
        factor = pair_factor(ra['cell'], rb['cell'])
        if factor is None:
            continue
        assert ra['input_sha256'] == rb['input_sha256']
        same_device = ra['device_report'] == rb['device_report']
        if factor == 'within_device_precision':
            assert same_device
        assert diagnostics[a]['bought_queries'] == diagnostics[b]['bought_queries']
        for step in range(6):
            pa = np.load(folders[a] / f'state_{step}.npy', mmap_mode='r', allow_pickle=False)
            pb = np.load(folders[b] / f'state_{step}.npy', mmap_mode='r', allow_pickle=False)
            for scope, left, right in [('query_and_score', pa, pb), ('score_only', pa[3:], pb[3:])]:
                pairs.append(dict(cell_a=a, cell_b=b, clip_id=ra['cell']['clip_id'], factor=factor,
                    precision_a=ra['cell']['precision'], precision_b=rb['cell']['precision'],
                    gpu_a=ra['actual_gpu'], gpu_b=rb['actual_gpu'], same_physical_device=same_device,
                    step=step, scope=scope, **compare_predictions(left, right)))
    def write(name, rows, empty):
        with (destination / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else empty)
            writer.writeheader(); writer.writerows(rows)
    write('paired_steps.csv', pairs, ['cell_a', 'cell_b', 'step', 'scope'])
    write('snapshot_fingerprints.csv', fingerprints, ['cell', 'step', 'scope', 'sha256'])
    (destination / 'cell_disposition.json').write_text(json.dumps(disposition, indent=2))
    from research_upgrade.metric_decomposition_v3 import evaluate
    scores = evaluate(records, folders, diagnostics, gold_root, destination)
    result = dict(registered_arms=len(cells), completed_arms=len(records), complete_design=len(records)==len(cells),
        pair_scope_steps=len(pairs), original_gate_modified=False, efficacy_promotion=False, final_test=False,
        runtime_intervention='precision_only_with_both_tf32_flags_disabled_after_engine_init', **scores)
    (destination / 'summary.json').write_text(json.dumps(result, indent=2))
    return result
