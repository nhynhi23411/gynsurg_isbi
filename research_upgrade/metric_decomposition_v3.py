"""Evaluator-only decomposition on released references, never selector features."""
from pathlib import Path
import csv
import hashlib
import itertools
import numpy as np


def score(prediction, reference):
    pred, gold = np.asarray(prediction) > .5, np.asarray(reference, dtype=bool)
    overlap = int((pred & gold).sum()); a, b = int(pred.sum()), int(gold.sum())
    return dict(dice=2*overlap/(a+b) if a+b else 1., iou=overlap/(a+b-overlap) if a+b-overlap else 1.,
                reference_present=b > 0, reference_pixels=b, predicted_pixels=a)


def evaluate(complete, folders, diagnostics, gold_root, destination):
    gold_root, destination = Path(gold_root), Path(destination)
    gold_cache, scores, lookup = {}, [], {}
    for key, record in complete.items():
        cid = record['cell']['clip_id']
        if cid not in gold_cache:
            path = gold_root / cid / 'sequence.npz'
            with path.open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == record['input_sha256']
            with np.load(path, allow_pickle=False) as data:
                gold_cache[cid] = data['gold'][[74, 99, 124]].astype(bool)
        gold = gold_cache[cid]
        for step in range(6):
            prediction = np.load(folders[key] / f'state_{step}.npy', mmap_mode='r', allow_pickle=False)[3:]
            for j, frame in enumerate([74, 99, 124]):
                for head in (0, 1):
                    row = dict(cell=key, clip_id=cid, step=step, frame=frame, head=head, **score(prediction[j, head], gold[j, head]))
                    scores.append(row); lookup[key, step, frame, head] = row
            if step in [0, 1, 2, 5]:
                expected = next(r for r in diagnostics[key]['metrics'] if r['budget'] == step)
                for metric in ['dice', 'iou']:
                    actual = np.mean([lookup[key, step, frame, head][metric] for frame in [74, 99, 124] for head in (0, 1)])
                    assert abs(actual - expected[metric]) < 1e-12
    contrasts = []
    for a, b in itertools.combinations(complete, 2):
        ca, cb = complete[a]['cell'], complete[b]['cell']
        if a.split(':')[0] == b.split(':')[0] or any(ca[f] != cb[f] for f in ['clip_id', 'mode', 'repeat']):
            continue
        for step in range(6):
            for frame in [74, 99, 124]:
                for head in (0, 1):
                    left, right = lookup[a, step, frame, head], lookup[b, step, frame, head]
                    assert left['reference_pixels'] == right['reference_pixels']
                    contrasts.append(dict(cell_a=a, cell_b=b, clip_id=ca['clip_id'], mode=ca['mode'], repeat=ca['repeat'],
                        step=step, frame=frame, head=head, reference_present=left['reference_present'],
                        predicted_pixels_a=left['predicted_pixels'], predicted_pixels_b=right['predicted_pixels'],
                        dice_a=left['dice'], dice_b=right['dice'], delta_dice=right['dice']-left['dice'],
                        delta_iou=right['iou']-left['iou'],
                        paid_query_prefix_identical=diagnostics[a]['bought_queries'][:step] == diagnostics[b]['bought_queries'][:step]))
    for name, rows in [('per_head_scores', scores), ('reference_decomposition', contrasts)]:
        with (destination / (name + '.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['cell_a', 'cell_b'])
            writer.writeheader(); writer.writerows(rows)
    return dict(per_head_rows=len(scores), environment_head_contrasts=len(contrasts),
                absent_reference_nonzero_contrasts=sum(not r['reference_present'] and r['delta_dice'] != 0 for r in contrasts),
                all_registered_aggregate_metrics_reconstructed=True, threshold_changed=False, evaluator_only=True)
