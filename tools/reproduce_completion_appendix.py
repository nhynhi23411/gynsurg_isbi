"""Reconstruct the nonconforming appendix; never converts its failed gate to pass."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

folder = Path(__file__).resolve().parents[1] / 'benchmarks/completion_appendix_20261022'
status = json.loads((folder / 'EVIDENCE_STATUS.json').read_text())
assert status['original_primary_historical_parity_passed'] is False
assert status['nonconforming_exploratory_only'] is True
assert status['final_test'] is False and status['candidate_promoted'] is False


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261019)
    return np.quantile(values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1), [.025, .975])


clips = pd.read_csv(folder / 'paid_auc.csv', dtype={'clip_id': str})
assert clips.clip_id.nunique() == 170 and clips.procedure_id.nunique() == 9
assert set(clips['mode']) == {'raw', 'eva_qnet_mask_adaptation'}
assert not clips.duplicated(['clip_id', 'mode', 'pool']).any()
assert (clips.groupby(['mode', 'pool']).size() == 170).all()
sources = clips.groupby(['procedure_id', 'mode', 'pool'])[['dice_auc', 'iou_auc']].mean().reset_index()
paid = pd.read_csv(folder / 'paid_summary.csv')
for row in paid.itertuples(index=False):
    current = sources[(sources['mode'] == row.mode) & (sources.pool == row.pool)].set_index('procedure_id').sort_index()
    raw = sources[(sources['mode'] == 'raw') & (sources.pool == row.pool)].set_index('procedure_id').loc[current.index]
    delta = (current.dice_auc - raw.dice_auc).to_numpy()
    actual = [current.dice_auc.mean(), current.iou_auc.mean(), delta.mean(), *interval(delta)]
    expected = [row.dice_auc, row.iou_auc, row.delta_vs_contemporaneous_raw, row.lower95, row.upper95]
    assert np.allclose(actual, expected, rtol=0, atol=1e-12), (row.mode, row.pool)
    assert int((delta > 0).sum()) == row.positive_sources

automatic = pd.read_csv(folder / 'automatic_clip_endpoints.csv', dtype={'clip_id': str})
assert len(automatic) == 510 and (automatic.paid_masks == 0).all()
assert not automatic.duplicated(['clip_id', 'pool']).any()
auto_summary = pd.read_csv(folder / 'automatic_summary.csv')
assert (auto_summary.paid_masks == 0).all() and not auto_summary.gold_budget_auc_defined.any()
for row in auto_summary.itertuples(index=False):
    by_source = automatic[automatic.pool == row.pool].groupby('procedure_id')[['dice', 'baseline_dice', 'delta']].mean().sort_index()
    actual = [by_source.dice.mean(), by_source.baseline_dice.mean(), by_source.delta.mean(), *interval(by_source.delta)]
    assert np.allclose(actual, [row.dice, row.baseline_dice, row.delta, row.lower95, row.upper95], rtol=0, atol=1e-12)
    assert int((by_source.delta > 0).sum()) == row.positive_sources

parity = pd.read_csv(folder / 'historical_parity_by_clip.csv', dtype={'clip_id': str})
assert len(parity) == parity.clip_id.nunique() == 170
assert int((~parity.historical_parity_pass).sum()) == 20
assert set(parity.loc[~parity.historical_query_same, 'clip_id']) == {'044', '132', '148'}
assert set(parity.loc[~parity.historical_parity_pass, 'actual_gpu']) == {'NVIDIA A10G'}
assert parity.actual_gpu.value_counts().to_dict() == {'NVIDIA A10': 127, 'NVIDIA A10G': 43}
assert np.array_equal(parity.historical_parity_pass.to_numpy(), (
    parity.historical_query_same & (parity.raw_maximum_metric_difference <= .001)
    & (parity.raw_auc_difference_bound <= .0001)).to_numpy())
compute = pd.read_csv(folder / 'compute.csv').set_index('mode')
assert compute.loc[['raw', 'eva_qnet_mask_adaptation', 'sasvi_predicted_adaptation'], 'clips'].eq(170).all()
assert compute.loc[['uniform', 'constant_reject_software'], 'clips'].eq(9).all()
print('PASS: six paid and three separate automatic summaries reconstructed to 1e-12; all170 retained, original parity gate still FAILED20/170.')
