"""Fixed common1-second query-neighborhood exclusion; CPU score-only sensitivity."""
from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.scoring import common_scoring
from src.common import paired_bootstrap
BASE=Path('C:/Users/pc/ISBI_GynSurg_runs/train_expansion_20261017_v1')

def temporal_auc(frames, query_frames):
    if not set(query_frames)<={0,24,49}:raise ValueError('Unexpected charged query frames')
    common=common_scoring([74,99,124],query_frames,25,1)
    selected=frames[frames.frame_index.isin(common)].copy()
    curve=selected.groupby(['n_queries','structure']).dice.mean().groupby('n_queries').mean().reindex([0,1,2,5])
    if curve.isna().any():raise ValueError('Incomplete sensitivity budget grid')
    if any(len(b)!=len(common) for _,b in selected.groupby(['n_queries','structure'])):raise ValueError('Scoring coverage differs across budget/head')
    return dict(raw_auc=float(curve.to_numpy()@np.array([.1,.2,.4,.3])),baseline_auc=float(curve.iloc[0]),common_frames=common.tolist(),coverage=len(common)/3)

def main():
    execution=json.loads((BASE/'cohort_execution.json').read_text())
    if execution['final_test'] or len(execution['reports'])!=170:raise ValueError('Complete training only')
    out=BASE/'analysis/temporal_1s';out.mkdir(exist_ok=False);rows=[];evidence=[]
    for r in execution['reports']:
        folder=BASE/r['clip_id']/'suite/uncertainty_raw'
        query_path=folder/'queries.csv';queries=pd.read_csv(query_path)
        if len(queries)!=5 or queries.duplicated(['frame','structure']).any():raise ValueError('Wrong charged query trajectory')
        result=temporal_auc(pd.read_csv(folder/'per_frame.csv'),queries.frame.tolist())
        if result['common_frames']!=[99,124]:raise ValueError('Unexpected common window for five of six candidate pairs')
        rows.append(dict(clip_id=r['clip_id'],procedure_id=r['procedure_id'],raw_auc=result['raw_auc'],baseline_auc=result['baseline_auc']))
        evidence.append(dict(clip_id=r['clip_id'],common_frames=result['common_frames'],coverage=result['coverage'],query_sha256=hashlib.sha256(query_path.read_bytes()).hexdigest()))
    clips=pd.DataFrame(rows);source=clips.groupby('procedure_id')[['raw_auc','baseline_auc']].mean()
    contrast=paired_bootstrap(source.raw_auc.to_numpy(),source.baseline_auc.to_numpy(),seed=20261017)
    clips.to_csv(out/'per_clip_auc.csv',index=False);source.to_csv(out/'per_source_auc.csv')
    report=dict(secondary=True,primary_endpoint_unchanged=True,clips=170,source_groups=len(source),common_scoring_frames=[99,124],coverage=2/3,scored_rows=170*4*2*2,new_model_calls=0,query_trajectories_unchanged=True,baseline_and_every_budget_use_same_pool=True,mean_raw_auc=float(source.raw_auc.mean()),mean_baseline_auc=float(source.baseline_auc.mean()),raw_minus_baseline=contrast,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),final_test_open=False)
    (out/'evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf8');(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
