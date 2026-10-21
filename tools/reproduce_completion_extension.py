"""Reconstruct all nine-clip paid source means and intervals from released clip AUCs."""
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
folder = root / 'benchmarks/completion_extension_20261021'
clips = pd.read_csv(folder / 'paid_auc.csv', dtype={'clip_id': str})
assert clips.clip_id.nunique() == 9 and clips.procedure_id.nunique() == 9
assert not clips['mode'].str.contains('sasvi').any(), 'Automatic endpoint has no paid-mask AUC'
sources = clips.groupby(['procedure_id', 'mode', 'pool'])[['dice_auc', 'iou_auc']].mean().reset_index()
expected = pd.read_csv(folder / 'paid_summary.csv')
checked = 0
for row in expected.itertuples(index=False):
    a = sources[(sources['mode'] == row.mode) & (sources.pool == row.pool)].set_index('procedure_id').sort_index()
    b = sources[(sources['mode'] == 'raw') & (sources.pool == row.pool)].set_index('procedure_id').loc[a.index]
    delta = (a.dice_auc - b.dice_auc).to_numpy()
    rng = np.random.default_rng(20261019)
    interval = np.quantile(delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(axis=1), [.025, .975])
    actual = [a.dice_auc.mean(), a.iou_auc.mean(), delta.mean(), *interval]
    target = [row.dice_auc, row.iou_auc, row.delta_vs_contemporaneous_raw, row.lower95, row.upper95]
    assert np.allclose(actual, target, atol=1e-12, rtol=0), (row.mode, row.pool)
    assert int((delta > 0).sum()) == row.positive_sources
    checked += 1
automatic = pd.read_csv(folder / 'automatic_summary.csv')
assert (automatic.paid_masks == 0).all() and not automatic.gold_budget_auc_defined.any()
assert np.allclose(automatic.dice - automatic.baseline_dice, automatic.delta, atol=1e-12, rtol=0)
ablation = pd.read_csv(folder / 'ablations_summary.csv')
original = ablation[(ablation.pool == 'original') & (ablation.method != 'raw')]
assert len(original) == 10 and (original.delta < 0).all()
print(f'PASS: {checked} paid contrasts reconstructed to 1e-12; automatic endpoint separate; all ten original-pool ablations negative.')
