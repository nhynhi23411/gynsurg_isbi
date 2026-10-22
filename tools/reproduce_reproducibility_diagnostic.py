"""Check released technical coverage, fingerprints and historical scalar replay."""
from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1] / 'benchmarks/reproducibility_diagnostic_20261022'
disposition = pd.DataFrame(json.loads((root / 'cell_disposition.json').read_text()))
assert len(disposition) == disposition.id.nunique() == 48
assert disposition.status.eq('complete').sum() == 19
assert disposition.status.eq('hardware_mismatch_no_inference').sum() == 6
assert disposition.status.str.startswith('not_dispatched').sum() == 23
records = pd.read_csv(root / 'cell_records.csv', dtype={'clip_id': str}).set_index('cell')
assert len(records) == 25
complete = records[records.status == 'complete']
assert len(complete) == 19 and set(complete.actual_gpu) == {'NVIDIA A10'}
fingerprints = pd.read_csv(root / 'step_fingerprints.csv').set_index(['cell', 'step'])
assert len(fingerprints) == 19 * 6
query = pd.read_csv(root / 'query_history.csv')
assert len(query) == 19 * 5
histories = {key: list(zip(group.sort_values('query').frame, group.sort_values('query')['head'])) for key, group in query.groupby('cell')}
pairs = pd.read_csv(root / 'paired_steps.csv')
assert len(pairs) == 96
expected = set()
for a, b in itertools.combinations(sorted(complete.index), 2):
    ca, cb = complete.loc[a], complete.loc[b]
    factors = [f for f in ['expected_gpu', 'mode', 'repeat'] if ca[f] != cb[f]]
    if ca.clip_id == cb.clip_id and len(factors) == 1:
        expected.update((a, b, factors[0], step) for step in range(6))
assert set(zip(pairs.cell_a, pairs.cell_b, pairs.factor, pairs.step)) == expected
for row in pairs.itertuples(index=False):
    same = fingerprints.loc[(row.cell_a, row.step), 'sha256'] == fingerprints.loc[(row.cell_b, row.step), 'sha256']
    assert bool(row.exact_numeric_equality) == same
    assert bool(row.paid_query_prefix_identical) == (histories[row.cell_a][:row.step] == histories[row.cell_b][:row.step])
    if same:
        assert row.maximum_absolute_difference == row.mean_absolute_difference == row.threshold_05_disagreement_fraction == 0
assert pairs.exact_numeric_equality.all(), 'Release-specific finding, not a general guarantee'

metrics = pd.read_csv(root / 'diagnostic_metrics.csv').set_index(['cell', 'budget'])
historical = pd.read_csv(root / 'historical_panel_curves.csv', dtype={'clip_id': str}).set_index(['clip_id', 'n_queries'])
old_queries = pd.read_csv(root / 'historical_panel_queries.csv', dtype={'clip_id': str})
expected_histories = {key: list(zip(group.sort_values('query').frame, group.sort_values('query')['head'])) for key, group in old_queries.groupby('clip_id')}
reported = pd.read_csv(root / 'historical_diagnostic.csv', dtype={'clip_id': str})
for row in reported.itertuples(index=False):
    difference = np.abs(metrics.loc[row.cell, ['dice', 'iou']].loc[[0, 1, 2, 5]].to_numpy() - historical.loc[row.clip_id, ['dice', 'iou']].loc[[0, 1, 2, 5]].to_numpy())
    maximum = difference.max()
    bound = (difference * np.array([.1, .2, .4, .3])[:, None]).sum(axis=0).max()
    assert np.allclose([maximum, bound], [row.max_metric_difference, row.auc_absolute_difference_bound], atol=1e-12, rtol=0)
    same_query = histories[row.cell] == expected_histories[row.clip_id]
    assert bool(row.diagnostic_historical_tolerance_pass) == bool(same_query and maximum <= .001 and bound <= .0001)
capability = json.loads((root / 'CAPABILITY_STATUS.json').read_text())
assert capability['scientific_calls_created'] == 0 and capability['inference_executed'] is False
assert json.loads((root / 'NUMERIC_QA.json').read_text())['original_failed_gate_repaired'] is False
print('PASS: 48 planned/25 submitted/19 complete/6 skipped/23 not dispatched;96 available step pairs verified by hashes;9 historical scalar contrasts reconstructed1e-12; original failed gate unchanged.')
