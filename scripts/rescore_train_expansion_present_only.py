"""Secondary reference-present sensitivity, without models or query changes."""
from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import paired_bootstrap
BASE=Path('C:/Users/pc/ISBI_GynSurg_runs/train_expansion_20261017_v1')

def present_auc(frame):
    required=['n_queries','structure','frame_index','gold_present','dice']
    if any(c not in frame for c in required):raise ValueError('Missing sensitivity columns')
    flags=frame.gold_present.astype(str).str.lower()
    if not flags.isin(['true','false']).all():raise ValueError('Missing/unknown reference presence')
    data=frame.copy();data['reference_present']=flags.eq('true')
    if data.groupby(['structure','frame_index']).reference_present.nunique().ne(1).any():raise ValueError('Reference-presence pool changed with budget')
    selected=data[data.reference_present].copy()
    if selected.empty:return None
    # Equal heads after averaging the present frames within each head.
    curve=selected.groupby(['n_queries','structure']).dice.mean().groupby('n_queries').mean().reindex([0,1,2,5])
    if curve.isna().any():raise ValueError('Incomplete fixed sensitivity curve')
    return dict(raw_auc=float(curve.to_numpy()@np.array([.1,.2,.4,.3])),baseline_auc=float(curve.iloc[0]),present_cells=int(selected[['structure','frame_index']].drop_duplicates().shape[0]),present_heads=int(selected.structure.nunique()),curve=curve.to_dict())

def main():
    execution=json.loads((BASE/'cohort_execution.json').read_text())
    if execution['final_test'] or len(execution['reports'])!=170:raise ValueError('Complete training cohort required')
    out=BASE/'analysis/present_only';out.mkdir(exist_ok=False)
    rows=[];excluded=[];coverage=[]
    for record in execution['reports']:
        cid=record['clip_id'];frame=pd.read_csv(BASE/cid/'suite/uncertainty_raw/per_frame.csv')
        result=present_auc(frame)
        if result is None:
            excluded.append(cid);coverage.append(dict(clip_id=cid,present_cells=0,present_heads=0));continue
        coverage.append(dict(clip_id=cid,present_cells=result['present_cells'],present_heads=result['present_heads']))
        rows.append(dict(clip_id=cid,procedure_id=record['procedure_id'],raw_auc=result['raw_auc'],baseline_auc=result['baseline_auc']))
    clips=pd.DataFrame(rows);source=clips.groupby('procedure_id')[['raw_auc','baseline_auc']].mean()
    if len(source)<2:raise ValueError('Insufficient source groups for sensitivity summary')
    contrast=paired_bootstrap(source.raw_auc.to_numpy(),source.baseline_auc.to_numpy(),seed=20261017)
    clips.to_csv(out/'per_clip_auc.csv',index=False);source.to_csv(out/'per_source_auc.csv');pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    report=dict(secondary=True,primary_endpoint_unchanged=True,new_model_calls=0,query_trajectories_unchanged=True,clips_with_present_reference=len(clips),clips_without_any_present_reference=excluded,source_groups=len(source),present_frame_structure_cells=sum(r['present_cells'] for r in coverage),total_unique_frame_structure_cells=170*6,aggregation='present frames within head, heads within clip, clips within source, sources equal',mean_raw_auc=float(source.raw_auc.mean()),mean_baseline_auc=float(source.baseline_auc.mean()),raw_minus_baseline=contrast,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),final_test_open=False)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
