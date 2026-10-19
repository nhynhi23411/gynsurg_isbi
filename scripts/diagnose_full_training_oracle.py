"""Post-hoc binary-output oracle, not a deployable gate or global upper bound."""
from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import paired_bootstrap
BASE=Path('C:/Users/pc/ISBI_GynSurg_runs/train_expansion_20261017_v1')

def oracle_auc(frame):
    keys=['structure','frame_index']
    base=frame[frame.n_queries==0][keys+['dice']].rename(columns={'dice':'baseline_dice'})
    data=frame.merge(base,on=keys,validate='many_to_one')
    if len(data)!=24 or data.duplicated(['n_queries']+keys).any():raise ValueError('Incomplete diagnostic grid')
    data['oracle_dice']=np.maximum(data.dice,data.baseline_dice)
    curve=data.groupby('n_queries')[['dice','baseline_dice','oracle_dice']].mean().reindex([0,1,2,5])
    if curve.isna().any().any():raise ValueError('Missing diagnostic budgets')
    weights=np.array([.1,.2,.4,.3])
    return dict(raw_auc=float(weights@curve.dice),baseline_auc=float(weights@curve.baseline_dice),oracle_auc=float(weights@curve.oracle_dice))

def main():
    execution=json.loads((BASE/'cohort_execution.json').read_text())
    if execution['final_test'] or len(execution['reports'])!=170:raise ValueError('Complete training only')
    out=BASE/'analysis/restricted_oracle';out.mkdir(exist_ok=False);rows=[]
    for r in execution['reports']:
        frame=pd.read_csv(BASE/r['clip_id']/'suite/uncertainty_raw/per_frame.csv')
        rows.append(dict(clip_id=r['clip_id'],procedure_id=r['procedure_id'],**oracle_auc(frame)))
    clips=pd.DataFrame(rows);clips['headroom']=clips.oracle_auc-clips.raw_auc
    source=clips.groupby('procedure_id')[['raw_auc','baseline_auc','oracle_auc','headroom']].mean()
    if len(source)!=9:raise ValueError('Wrong diagnostic source count')
    primary=json.loads((BASE/'analysis/report.json').read_text())
    if not np.isclose(source.raw_auc.mean(),primary['mean_raw_dice_auc'],rtol=0,atol=1e-12):raise ValueError('Raw AUC not reproduced')
    sizes=clips.groupby('procedure_id').size()
    clips['macro_contribution']=clips.headroom/clips.procedure_id.map(sizes)/9
    if not np.isclose(clips.macro_contribution.sum(),source.headroom.mean(),rtol=0,atol=1e-12):raise ValueError('Contribution weighting mismatch')
    largest=clips.loc[clips.macro_contribution.idxmax()];total=float(clips.macro_contribution.sum())
    contrast=paired_bootstrap(source.oracle_auc.to_numpy(),source.raw_auc.to_numpy(),seed=20261017)
    report=dict(posthoc_diagnostic=True,attainable_method=False,global_upper_bound=False,binary_choice_between_existing_outputs_only=True,clips=170,source_groups=9,independent_validation=False,model_rerun=False,gate_trained=False,query_trajectory_unchanged=True,primary_unchanged=True,mean_raw_auc=float(source.raw_auc.mean()),mean_baseline_auc=float(source.baseline_auc.mean()),mean_restricted_oracle_auc=float(source.oracle_auc.mean()),oracle_minus_raw_descriptive=contrast,largest_contribution_clip=largest.clip_id,largest_clip_share= float(largest.macro_contribution/total) if total else None,final_test_open=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),interpretation='Gold chooses per scoring frame/structure between frozen baseline and raw SAM on the fixed charged trajectory. Positive headroom is guaranteed by max construction and is not evidence of learnability or efficacy.')
    clips.to_csv(out/'per_clip_oracle_auc.csv',index=False);source.to_csv(out/'per_source_oracle_auc.csv')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
