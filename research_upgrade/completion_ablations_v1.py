"""Fixed diagnostic feature/fallback ablations. Never a candidate promotion run."""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np
import pandas as pd
from research_upgrade import acceptance_v2 as original
from research_upgrade.bank import SCORE, BUDGETS, AUC_WEIGHTS, popcount

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / 'results/research_upgrade_20261020_v13'
OUT = ROOT / 'results/research_upgrade_completion_20261020_v1/ablations'
META = {'target_uterus', 'paid_count', 'output_prompt_count', 'output_is_raw', 'frame_seconds'}
GROUPS = {
    'entropy': META | {'entropy', 'baseline_entropy', 'roi_entropy'},
    'temporal': META | {'nearest_paid_seconds', 'temporal_mask_consistency'},
    'anchor': META | {'anchor_support', 'anchor_reconstruction_mean', 'anchor_reconstruction_min'},
    'observable': None,
    'full': None,
}
POOLS = {'original': SCORE, 'late_common': (99, 124), 'last_common': (124,)}


def keep_columns(columns, group):
    if group not in GROUPS:
        raise ValueError('Unregistered feature group')
    names = [c.split('_', 1)[1] for c in columns]
    if group == 'full':
        return list(range(len(columns)))
    if group == 'observable':
        return [i for i, name in enumerate(names) if not name.startswith('anchor_')]
    return [i for i, name in enumerate(names) if name in GROUPS[group]]


class ProjectedModel:
    def __init__(self, fitted, keep):
        self.fitted, self.keep = fitted, keep

    def predict(self, x):
        return self.fitted.predict(np.asarray(x)[:, self.keep])


def decide(clip, model, fallback):
    """Only current paid observable features; labels are evaluator-only elsewhere."""
    if fallback not in ['previous', 'baseline']:
        raise ValueError('Unregistered fallback')
    accepted = {(k, t): 0 for k in range(2) for t in SCORE}
    states = {0: dict(accepted)}
    decisions = []
    for n, (_, head) in enumerate(clip['history'], 1):
        paid = clip['paths'][n][head]
        for frame in SCORE:
            old = accepted[head, frame]
            use, value = True, None
            if popcount(paid) >= 2 and old != paid:
                new_f = clip['features'][head, paid, paid, frame]
                old_f = clip['features'][head, paid, old, frame]
                vector = original.paired_features(new_f, old_f, clip['names'])
                value = float(model.predict(vector[None])[0])
                if not np.isfinite(value):
                    raise ValueError('Nonfinite diagnostic prediction')
                use = value > 0
            accepted[head, frame] = paid if use else (old if fallback == 'previous' else 0)
            assert accepted[head, frame] & ~paid == 0
            decisions.append(dict(clip_id=clip['cid'], procedure_id=clip['source'],
                                  query=n, head=head, frame=frame, paid=paid,
                                  old_subset=old, output_subset=accepted[head, frame],
                                  accepted=use, gate_score=value))
        states[n] = dict(accepted)
    return states, decisions


def evaluate(clip, states, pool):
    rows = []
    for n in BUDGETS:
        cells = [clip['labels'][k, states[n][k, t], t] for k in range(2) for t in pool]
        rows.append(dict(clip_id=clip['cid'], procedure_id=clip['source'], n_queries=n,
                         dice=float(np.mean([r['dice'] for r in cells])),
                         iou=float(np.mean([r['iou'] for r in cells]))))
    return rows


def main():
    if OUT.exists():
        raise FileExistsError('Preserve existing diagnostic outputs')
    freeze = json.loads((ROOT / 'protocols/20261020_completion_extension_v1/CPU_FREEZE.json').read_text())
    for name, digest in freeze['hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    proof = json.loads((BANK / 'full_acceptance.json').read_text())
    assert proof['passed'] and len(proof['reports']) == 170
    original.OUT = BANK
    clips, x, y, groups, columns = original.load_banks([r['clip_id'] for r in proof['reports']])
    sources = sorted(set(groups))
    assert len(sources) == 9
    OUT.mkdir(parents=True)
    all_rows, all_decisions, folds = [], [], []
    for clip in clips.values():
        states = {n: {(k, t): clip['paths'][n][k] for k in range(2) for t in SCORE} for n in BUDGETS}
        for name, pool in POOLS.items():
            all_rows.extend(dict(method='raw', pool=name, **r) for r in evaluate(clip, states, pool))
    for group in GROUPS:
        keep = keep_columns(columns, group)
        reduced = [columns[i] for i in keep]
        assert keep and not any('dice' in c or 'gold' in c or 'iou' in c for c in reduced)
        for held in sources:
            mask = groups != held
            train_sources = sorted(set(groups[mask]))
            assert held not in train_sources and len(train_sources) == 8
            model = original.PairModel('ridge', 'anchors').fit(x[mask][:, keep], y[mask], groups[mask], reduced)
            projected = ProjectedModel(model, keep)
            folds.append(dict(group=group, held_source=held, training_sources=train_sources,
                              columns=reduced, alpha=10., threshold=0., training_pairs=int(mask.sum())))
            for clip in clips.values():
                if clip['source'] != held:
                    continue
                for fallback in ['previous', 'baseline']:
                    method = group + '_' + fallback
                    states, decisions = decide(clip, projected, fallback)
                    all_decisions.extend(dict(method=method, **r) for r in decisions)
                    for name, pool in POOLS.items():
                        all_rows.extend(dict(method=method, pool=name, **r) for r in evaluate(clip, states, pool))
            print('OOF diagnostic', group, held, flush=True)
    frame = pd.DataFrame(all_rows)
    summaries, source_rows = [], []
    for pool in POOLS:
        pool_rows = frame[frame.pool == pool]
        raw = original.source_auc(pool_rows[pool_rows.method == 'raw'].to_dict('records'))
        for method in sorted(set(frame.method)):
            values = original.source_auc(pool_rows[pool_rows.method == method].to_dict('records')).reindex(raw.index)
            assert np.isfinite(values).all()
            contrast = original.bootstrap((values - raw).to_numpy())
            summaries.append(dict(pool=pool, method=method, dice_auc=float(values.mean()),
                                  delta=contrast['delta'], lower95=contrast['ci95'][0],
                                  upper95=contrast['ci95'][1], positive_sources=int((values > raw).sum()), sources=9))
            source_rows.extend(dict(pool=pool, method=method, procedure_id=s, dice_auc=float(values[s]),
                                    delta_raw=float(values[s] - raw[s])) for s in raw.index)
    frame.to_csv(OUT / 'clip_curves.csv', index=False)
    pd.DataFrame(summaries).to_csv(OUT / 'summary.csv', index=False)
    pd.DataFrame(source_rows).to_csv(OUT / 'source_auc.csv', index=False)
    pd.DataFrame(all_decisions).to_csv(OUT / 'decisions.csv', index=False)
    (OUT / 'folds.json').write_text(json.dumps(folds, indent=2))
    record = dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  training_clips=170, sources=9, fixed_groups=list(GROUPS), fallbacks=['previous', 'baseline'],
                  pools={k: list(v) for k, v in POOLS.items()}, prospective_validation=False,
                  exploratory_post_negative_diagnostic=True, candidate_promotion=False, final_test=False,
                  protocol_freeze_sha256=hashlib.sha256((ROOT / 'protocols/20261020_completion_extension_v1/CPU_FREEZE.json').read_bytes()).hexdigest())
    (OUT / 'completion.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
