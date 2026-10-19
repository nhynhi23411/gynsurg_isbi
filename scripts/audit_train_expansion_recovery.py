"""Final consistency audit after the170-case controller and analysis have ended."""
from pathlib import Path
import hashlib,json,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import paired_bootstrap
from rescore_train_expansion_present_only import present_auc
from rescore_train_expansion_temporal import temporal_auc
from diagnose_full_training_oracle import oracle_auc
BASE=Path('C:/Users/pc/ISBI_GynSurg_runs/train_expansion_20261017_v1')

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=ROOT/'protocols/20261019_train_expansion_recovery_v1'
    freeze=json.loads((p/'FREEZE.json').read_text());driver=json.loads((p/'LOCAL_DRIVER_FREEZE.json').read_text())
    if sha(ROOT/'scripts/modal_train_expansion_recovery.py')!=freeze['launcher_sha256']:raise ValueError('Worker changed')
    if sha(ROOT/'scripts/run_train_expansion_recovery.py')!=driver['driver_sha256']:raise ValueError('Driver changed')
    if sha(p/'PROTOCOL.md')!=freeze['protocol_sha256'] or sha(p/'FREEZE.json')!=driver['worker_freeze_sha256']:raise ValueError('Recovery contract changed')
    science=ROOT/'protocols/20261017_train_expansion_v1/FREEZE.json'
    if sha(science)!=freeze['scientific_freeze_sha256']:raise ValueError('Scientific freeze changed')
    for name,h in json.loads(science.read_text())['sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Scientific artifact changed: '+name)
    complete=json.loads((BASE/'recovery_20261019/pipeline_complete.json').read_text())
    preflight=json.loads((BASE/'recovery_20261019/runtime_preflight.json').read_text())
    pilot=json.loads((BASE/'recovery_20261019/pilot_002/cohort_acceptance.json').read_text())
    report=json.loads((BASE/'analysis/report.json').read_text())
    if not complete['complete'] or not preflight['passed'] or not pilot['complete'] or not report['complete'] or report['final_test_open']:raise ValueError('Pipeline incomplete')
    compact_verified=False
    if complete.get('transport')=='compact_metrics_20261019':
        for filename in ['COMPACT_TRANSPORT_FREEZE.json','COMPACT_FINISHER_FREEZE.json']:
            contract=json.loads((p/filename).read_text(encoding='utf-8-sig'))
            for name,h in contract['sha256'].items():
                if sha(ROOT/name)!=h:raise ValueError('Compact transport source changed')
        if (complete['transport_source_freeze_sha256']!=sha(p/'COMPACT_TRANSPORT_FREEZE.json')
            or complete['finisher_freeze_sha256']!=sha(p/'COMPACT_FINISHER_FREEZE.json')
            or contract['transport_freeze_sha256']!=sha(p/'COMPACT_TRANSPORT_FREEZE.json')):
            raise ValueError('Compact transport freeze chain mismatch')
        transport=BASE/'transport_compact_20261019'
        manifest=json.loads((transport/'manifest.json').read_text())
        if sha(transport/'metrics.zip')!=manifest['archive_sha256']:raise ValueError('Compact archive changed')
        for name,h in manifest['sha256'].items():
            if sha(BASE/name)!=h:raise ValueError('Transported metric changed: '+name)
        compact_verified=True
    receipts=json.loads((BASE/'cohort_execution.json').read_text())['reports']
    if len(receipts)!=170 or len({r['clip_id'] for r in receipts})!=170:raise ValueError('Wrong receipt count')
    if sum(r['reused'] for r in receipts)!=9 or any(r['actual_gpu']!='NVIDIA A10' or r['official_split']!='train' or r['final_test'] for r in receipts):raise ValueError('Wrong hardware/scope/reuse')
    if any(r.get('recovery_freeze_sha256')!=sha(p/'FREEZE.json') for r in receipts):raise ValueError('Wrong execution recovery version')
    attempts=json.loads((BASE/'recovery_20261019/attempts.json').read_text())
    rejected=[r for r in attempts if not r['accepted']]
    if any('Expansion GPU contract' not in r['error'] for r in rejected):raise ValueError('Nonallocation retry requires review')
    table=pd.read_csv(BASE/'analysis/per_clip_auc.csv',dtype={'clip_id':str})
    if table.clip_id.duplicated().any() or set(table.clip_id)!={r['clip_id'] for r in receipts}:raise ValueError('Wrong clip AUC membership')
    source=table.groupby('procedure_id')[['raw_dice_auc','baseline_dice_auc']].mean().sort_index()
    saved=pd.read_csv(BASE/'analysis/per_source_auc.csv').set_index('procedure_id').sort_index()
    if len(table)!=170 or len(source)!=9:raise ValueError('Wrong aggregation sample count')
    np.testing.assert_allclose(source,saved[['raw_dice_auc','baseline_dice_auc']],rtol=0,atol=1e-12)
    contrast=paired_bootstrap(source.raw_dice_auc.to_numpy(),source.baseline_dice_auc.to_numpy(),seed=20261017)
    np.testing.assert_allclose(contrast['ci95'],report['raw_minus_budget_zero']['ci95'],rtol=0,atol=1e-12)
    if not np.isclose(contrast['delta'],report['raw_minus_budget_zero']['delta'],rtol=0,atol=1e-12):raise ValueError('Contrast mismatch')
    frames=aggregates=0;present_rows=[];present_cells=0;temporal_rows=[];oracle_rows=[]
    for r in receipts:
        folder=BASE/r['clip_id'];record=json.loads((folder/'execution.json').read_text())
        if record!=r:raise ValueError('Receipt mismatch')
        frame=pd.read_csv(folder/'suite/uncertainty_raw/per_frame.csv');clip=pd.read_csv(folder/'suite/uncertainty_raw/per_clip.csv')
        frames+=len(frame);aggregates+=len(clip)
        oracle_rows.append(dict(procedure_id=r['procedure_id'],**oracle_auc(frame)))
        if len(frame)!=24 or len(clip)!=8:raise ValueError('Wrong endpoint rows')
        pres=present_auc(frame)
        if pres is not None:
            present_rows.append(dict(procedure_id=r['procedure_id'],raw_auc=pres['raw_auc'],baseline_auc=pres['baseline_auc']))
            present_cells+=pres['present_cells']
        queries=pd.read_csv(folder/'suite/uncertainty_raw/queries.csv')
        temp=temporal_auc(frame,queries.frame.tolist())
        temporal_rows.append(dict(procedure_id=r['procedure_id'],raw_auc=temp['raw_auc'],baseline_auc=temp['baseline_auc']))
    if frames!=4080 or aggregates!=1360:raise ValueError('Incomplete endpoint grid')
    review=json.loads((ROOT/'configs/review_status.json').read_text())
    if review['approved_for_test']:raise ValueError('Final test unexpectedly opened')
    acceptance=json.loads((BASE/'analysis/cohort_acceptance.json').read_text())
    if not acceptance['complete'] or len(acceptance['checks'])!=170:raise ValueError('Full metric acceptance missing')
    secondary_freeze=json.loads((p/'PRESENT_ONLY_SENSITIVITY_FREEZE.json').read_text())
    for name,h in secondary_freeze['sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Secondary source changed')
    secondary=json.loads((BASE/'analysis/present_only/report.json').read_text())
    ps=pd.DataFrame(present_rows).groupby('procedure_id')[['raw_auc','baseline_auc']].mean()
    pc=paired_bootstrap(ps.raw_auc.to_numpy(),ps.baseline_auc.to_numpy(),seed=20261017)
    np.testing.assert_allclose(pc['ci95'],secondary['raw_minus_baseline']['ci95'],rtol=0,atol=1e-12)
    if not np.isclose(pc['delta'],secondary['raw_minus_baseline']['delta'],rtol=0,atol=1e-12) or present_cells!=secondary['present_frame_structure_cells']:raise ValueError('Present sensitivity mismatch')
    if len(present_rows)!=secondary['clips_with_present_reference'] or len(ps)!=secondary['source_groups']:raise ValueError('Present sensitivity coverage mismatch')
    temporal_freeze=json.loads((p/'TEMPORAL_SENSITIVITY_FREEZE.json').read_text())
    for name,h in temporal_freeze['sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Temporal sensitivity source changed')
    ts=pd.DataFrame(temporal_rows).groupby('procedure_id')[['raw_auc','baseline_auc']].mean()
    tc=paired_bootstrap(ts.raw_auc.to_numpy(),ts.baseline_auc.to_numpy(),seed=20261017)
    temporal=json.loads((BASE/'analysis/temporal_1s/report.json').read_text())
    np.testing.assert_allclose(tc['ci95'],temporal['raw_minus_baseline']['ci95'],rtol=0,atol=1e-12)
    if not np.isclose(tc['delta'],temporal['raw_minus_baseline']['delta'],rtol=0,atol=1e-12):raise ValueError('Temporal sensitivity mismatch')
    if temporal['common_scoring_frames']!=[99,124] or temporal['scored_rows']!=2720:raise ValueError('Temporal sensitivity pool mismatch')
    for filename in ['FULL_ORACLE_DIAGNOSTIC_FREEZE.json','IDEMPOTENT_STOP_FREEZE.json']:
        contract=json.loads((p/filename).read_text(encoding='utf-8-sig'))
        for name,h in contract['sha256'].items():
            if sha(ROOT/name)!=h:raise ValueError('Additional diagnostic/resume source changed: '+name)
    osource=pd.DataFrame(oracle_rows).groupby('procedure_id')[['raw_auc','oracle_auc']].mean()
    oc=paired_bootstrap(osource.oracle_auc.to_numpy(),osource.raw_auc.to_numpy(),seed=20261017)
    oracle=json.loads((BASE/'analysis/restricted_oracle/report.json').read_text())
    np.testing.assert_allclose(oc['ci95'],oracle['oracle_minus_raw_descriptive']['ci95'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(oc['delta'],oracle['oracle_minus_raw_descriptive']['delta'],rtol=0,atol=1e-12)
    files=[q for q in BASE.rglob('*') if q.is_file() and q.suffix in ['.csv','.json','.txt','.png','.pdf','.svg'] and q.name!='final_recovery_audit.json']
    result=dict(passed=True,clips=170,new_inference=161,reused=9,source_groups=9,frame_rows=frames,aggregate_rows=aggregates,allocation_mismatches=len(rejected),scientific_freeze_unchanged=True,recovery_source_verified=True,source_bootstrap_reproduced=True,present_sensitivity_reproduced=True,temporal_sensitivity_reproduced=True,independent_validation=False,clinical_review_complete=False,final_test_open=False,artifact_sha256={q.relative_to(BASE).as_posix():sha(q) for q in files},note='This final seal supersedes volatile progress/controller-log hash snapshots captured by the original analyzer while it was still printing.')
    result['full_restricted_oracle_reproduced']=True
    result['idempotent_stop_addendum_verified']=True
    result['compact_metric_transport_verified']=compact_verified
    (BASE/'final_recovery_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    (ROOT/'materials/verification/train_expansion_recovery_final_audit.json').write_text(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},indent=2))

if __name__=='__main__':main()
