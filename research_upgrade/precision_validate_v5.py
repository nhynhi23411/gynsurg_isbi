"""Independent local audit of paired-process identity and scalar diagnostic outputs."""
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
from research_upgrade.precision_controls_v5 import pair_factor


def validate(root, protocol):
    root,protocol=Path(root),Path(protocol)
    registration_file=protocol/'EXECUTION_FREEZE.json'
    if not registration_file.exists(): registration_file=protocol/'REGISTRATION.json'
    registration=json.loads(registration_file.read_text())
    disposition=json.loads((root/'analysis/cell_disposition.json').read_text())
    assert len(disposition)==8 and {c['id'] for c in disposition}=={c['id'] for c in registration['cells']}
    records,diagnostics={},{}
    for cell in disposition:
        if cell['status']!='complete': continue
        folder=root/'cells'/cell['id']; record=json.loads((folder/'execution.json').read_text())
        assert record['cell']==next(c for c in registration['cells'] if c['id']==cell['id'])
        assert record['sam_precision']==cell['precision'] and record['seed']==20261017 and record['paid_masks']==5
        assert not record['final_test'] and not record['primary_historical_gate_modified']
        flags=record['environment_after_engine_init']; assert not flags['tf32_matmul'] and not flags['tf32_cudnn']
        assert hashlib.sha256((folder/'diagnostics.json').read_bytes()).hexdigest()==record['files']['diagnostics.json']
        diagnostic=json.loads((folder/'diagnostics.json').read_text())
        assert diagnostic['bought_queries']==registration['historical_queries'][cell['clip_id']]
        assert len(set(map(tuple,diagnostic['bought_queries'])))==5
        records[cell['id']],diagnostics[cell['id']]=record,diagnostic
    fingerprints=pd.read_csv(root/'analysis/snapshot_fingerprints.csv').set_index(['cell','step','scope'])
    assert len(fingerprints)==len(records)*12
    for key,diagnostic in diagnostics.items():
        assert [s['step'] for s in diagnostic['snapshots']]==list(range(6))
        for item in diagnostic['snapshots']:
            assert fingerprints.loc[(key,item['step'],'query_and_score'),'sha256']==item['sha256']
    pairs=pd.read_csv(root/'analysis/paired_steps.csv')
    expected=set()
    verified_devices=0
    for a,b in itertools.combinations(records,2):
        ra,rb=records[a],records[b]; factor=pair_factor(ra['cell'],rb['cell'])
        if factor is None: continue
        assert ra['input_sha256']==rb['input_sha256']
        if factor=='within_device_precision':
            verified_devices+=1
            assert ra['device_report']==rb['device_report']
            assert ra['environment_after_engine_init']==rb['environment_after_engine_init']
            assert ra['environment']['torch']==rb['environment']['torch']
        expected.update((min(a,b),max(a,b),factor,s,scope) for s in range(6) for scope in ['query_and_score','score_only'])
    actual={(min(r.cell_a,r.cell_b),max(r.cell_a,r.cell_b),r.factor,r.step,r.scope) for r in pairs.itertuples(index=False)}
    assert expected==actual and len(actual)==len(pairs)
    for row in pairs.itertuples(index=False):
        same=fingerprints.loc[(row.cell_a,row.step,row.scope),'sha256']==fingerprints.loc[(row.cell_b,row.step,row.scope),'sha256']
        assert same==row.exact_numeric_equality
        assert row.same_physical_device==(records[row.cell_a]['device_report']==records[row.cell_b]['device_report'])
        values=[row.maximum_absolute_difference,row.mean_absolute_difference,row.threshold_05_disagreement_fraction]
        assert np.isfinite(values).all() and 0<=values[1]<=values[0]<=1 and 0<=values[2]<=1
        if same: assert values==[0,0,0]
        if row.step==0: assert same
    if not records:
        result=dict(passed=True,registered_arms=8,completed_arms=0,complete_design=False,pair_scope_steps=0,per_head_rows=0,within_device_identity_verified_pairs=0,scientific_precision_contrast_available=False,original_primary_gate_modified=False,final_test=False)
        (root/'independent_integrity.json').write_text(json.dumps(result,indent=2))
        return result
    scores=pd.read_csv(root/'analysis/per_head_scores.csv')
    assert len(scores)==len(records)*36 and not scores.duplicated(['cell','step','frame','head']).any()
    assert scores.groupby(['clip_id','frame','head']).reference_pixels.nunique().eq(1).all()
    for key,diagnostic in diagnostics.items():
        for row in diagnostic['metrics']:
            means=scores[scores.cell.eq(key)&scores.step.eq(row['budget'])][['dice','iou']].mean().to_numpy()
            assert np.allclose(means,[row['dice'],row['iou']],atol=1e-12,rtol=0)
    empty=scores[~scores.reference_present]
    assert np.array_equal(empty.dice.to_numpy(),empty.predicted_pixels.eq(0).astype(float).to_numpy())
    decomposed=pd.read_csv(root/'analysis/reference_decomposition.csv')
    lookup=scores.set_index(['cell','step','frame','head'])
    for row in decomposed.itertuples(index=False):
        a=lookup.loc[(row.cell_a,row.step,row.frame,row.head)]; b=lookup.loc[(row.cell_b,row.step,row.frame,row.head)]
        assert np.allclose([row.delta_dice,row.delta_iou],[b.dice-a.dice,b.iou-a.iou],atol=1e-12,rtol=0)
    result=dict(passed=True,registered_arms=8,completed_arms=len(records),complete_design=len(records)==8,pair_scope_steps=len(pairs),per_head_rows=len(scores),within_device_identity_verified_pairs=verified_devices,scientific_precision_contrast_available=verified_devices>0,original_primary_gate_modified=False,final_test=False)
    (root/'independent_integrity.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2)); return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('root',type=Path); parser.add_argument('protocol',type=Path)
    args=parser.parse_args(); validate(args.root,args.protocol)
