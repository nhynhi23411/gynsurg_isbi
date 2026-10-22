"""Audit published scalar/hash evidence without private predictions or labels."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_upgrade.reproducibility_validate_v4 import validate
ROOT=Path(__file__).resolve().parents[1]
data=ROOT/'benchmarks/reproducibility_allocator_20261022'
qa=validate(data/'allocator',data/'reference')
result=data/'allocator/independent_integrity.json'
result.write_text(result.read_text(),encoding='utf-8',newline='\n')
assert qa['complete_design'] and qa['verified_pair_scope_steps']==516
report=ROOT/'docs/reproducibility_20261022/allocator_observed'
expected=json.loads((report/'NUMERIC_QA.json').read_text())
pairs=pd.read_csv(data/'allocator/analysis/paired_steps.csv')
scoring=pairs[pairs.scope.eq('score_only')]
same=scoring[scoring.actual_gpu_a.eq(scoring.actual_gpu_b)]
assert len(same)==192 and same.exact_numeric_equality.all()
fixed=scoring[scoring.mode_a.eq('fixed')&scoring.mode_b.eq('fixed')&scoring.actual_gpu_a.ne(scoring.actual_gpu_b)]
assert len(fixed)==24
assert fixed.paid_query_prefix_identical.all()
assert np.isclose(fixed.maximum_absolute_difference.max(),expected['fixed_max_probability_delta'],atol=1e-12,rtol=0)
assert np.isclose(fixed.threshold_05_disagreement_fraction.max(),expected['fixed_max_mask_flip_fraction'],atol=1e-12,rtol=0)
scores=pd.read_csv(data/'allocator/analysis/per_head_scores.csv').set_index(['cell','step','frame','head'])
contributions=pd.read_csv(report/'fixed_head_contributions.csv')
for row in contributions.itertuples(index=False):
    a=scores.loc[(row.cell_a,row.step,row.frame,row.head)]; b=scores.loc[(row.cell_b,row.step,row.frame,row.head)]
    assert np.allclose([a.dice,b.dice,b.dice-a.dice],[row.dice_a,row.dice_b,row.delta_dice],atol=1e-12,rtol=0)
assert not len(contributions[~contributions.reference_present&contributions.delta_dice.ne(0)])
d=pd.read_csv(data/'allocator/analysis/reference_decomposition.csv')
e=d[~d.reference_present&d.delta_dice.ne(0)]
assert len(e)==3 and e.paid_query_prefix_identical.all()
assert set(e.step)=={1,2,3} and set(e.predicted_pixels_a)=={0} and set(e.predicted_pixels_b)=={925}
assert not expected['original_failed_gate_repaired'] and not expected['hardware_causality_established']
print('PASS: 24/24 cells;43 records;516 pair-scope steps;1548 per-head metrics;192 same-class exact steps;fixed-query cross-class and empty-reference findings reconstructed from released scalar evidence. Private probability differences are not recomputed without private arrays. Original failed gate unchanged.')
