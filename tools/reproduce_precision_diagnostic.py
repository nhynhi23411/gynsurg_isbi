"""Validate released scalar/hash evidence; private prediction arrays are excluded."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_upgrade.precision_validate_v5 import validate

ROOT=Path(__file__).resolve().parents[1]
data=ROOT/'benchmarks/precision_runtime_20261022'
report=ROOT/'docs/reproducibility_20261022/precision_runtime'
qa=validate(data,report)
result=data/'independent_integrity.json'
result.write_text(result.read_text(),encoding='utf-8',newline='\n')
expected=json.loads((report/'NUMERIC_QA.json').read_text())
assert qa['completed_arms']==expected['completed_arms']==6
assert qa['registered_arms']==8 and not qa['complete_design']
assert qa['pair_scope_steps']==expected['pair_scope_steps']==60
assert qa['per_head_rows']==216 and qa['within_device_identity_verified_pairs']==3
calls=json.loads((data/'CALL_DISPOSITION.json').read_text())
assert len(calls)==4 and all(c['attempted'] for c in calls)
assert sum(c['status']=='complete' for c in calls)==3
assert sum(c['status']=='FileExistsError' for c in calls)==1
pairs=pd.read_csv(data/'analysis/paired_steps.csv',dtype={'clip_id':str})
scoring=pairs[pairs.scope.eq('score_only')]
for precision in ['bfloat16','float32']:
    group=scoring[scoring.factor.eq('same_precision_repeat')&scoring.clip_id.eq('132')&scoring.precision_a.eq(precision)]
    observed=expected['observed_cross_class_clip132'][precision]
    assert np.allclose([group.maximum_absolute_difference.max(),group.threshold_05_disagreement_fraction.max()],
        [observed['max_probability_difference'],observed['max_flip_fraction']],atol=1e-12,rtol=0)
assert not expected['original_gate_modified'] and not expected['final_test']
scope=json.loads((report/'PUBLIC_SCOPE.json').read_text())
assert scope['original_device_identifiers_redacted'] and not scope['private_probability_arrays_released']
note=json.loads((report/'PRECISION_INTERPRETATION_NOTE.json').read_text())
assert not note['protocol_or_scientific_code_changed']
print('PASS: four paired calls retained (three complete, one guard failure);six/eight arms;60 pair-scope steps;216 per-head scores;three within-process device-alias checks;cross-class scalar findings verified. Public aliases attest to private identity verification. Pixel-level probability maxima cannot be recomputed without private arrays. Original failed gate unchanged.')
