"""Recipe-aware analysis: do not merge raw/fallback runs just by policy name."""
from pathlib import Path
import argparse,json,sys,subprocess
import numpy as np,pandas as pd
from scipy.integrate import trapezoid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import paired_bootstrap

def main():
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);args=a.parse_args()
    base=ROOT/'results'/args.run;record=json.loads((base/'cohort_execution.json').read_text())
    if record['final_test'] or record['stage'] not in ['engineering','development']:raise ValueError('Development-only analysis')
    expected={'059','086'} if record['stage']=='engineering' else {'001','038','059','086','105','116','138','146','160'}
    if {r['clip_id'] for r in record['reports']}!=expected:raise ValueError('Incomplete declared cohort')
    out=base/'analysis';out.mkdir(exist_ok=False)
    cohort=pd.DataFrame([dict(clip_id=r['clip_id'],procedure_id=r['procedure_id']) for r in record['reports']]);cohort.to_csv(out/'cohort.csv',index=False)
    registry=ROOT/'configs/reliability_scaled_a10_v1/experiment_registry.json'
    subprocess.run([sys.executable,str(ROOT/'scripts/validate_cohort_results.py'),'--results-root',str(base),'--cohort',str(out/'cohort.csv'),'--registry',str(registry),'--output',str(out/'cohort_acceptance.json')],check=True)
    auc=[]
    for r in record['reports']:
        for recipe in json.loads(registry.read_text()):
            df=pd.read_csv(base/r['clip_id']/'suite'/recipe['run_id']/'per_clip.csv')
            curve=df.groupby('actual_density')[['dice','iou']].mean().sort_index();x=curve.index.to_numpy()
            auc.append(dict(clip_id=r['clip_id'],procedure_id=r['procedure_id'],recipe=recipe['run_id'],dice_auc=float(trapezoid(curve.dice,x)/(x[-1]-x[0])),iou_auc=float(trapezoid(curve.iou,x)/(x[-1]-x[0]))))
    pd.DataFrame(auc).to_csv(out/'recipe_auc.csv',index=False)
    table=pd.DataFrame(auc).groupby(['procedure_id','recipe']).dice_auc.mean().unstack();contrasts=[]
    for lhs,rhs in [('scaled_fallback','uncertainty_raw'),('scaled_raw','error_raw'),('uncertainty_fallback','uncertainty_raw'),('scaled_fallback','scaled_raw'),('legacy_full_raw','uncertainty_raw')]:
        values=(table[lhs]-table[rhs]).to_numpy()
        c=dict(proposed=lhs,baseline=rhs,mean_delta=float(values.mean()),n_procedures=len(values))
        if record['stage']=='development':c['exploratory_bootstrap']=paired_bootstrap(table[lhs].to_numpy(),table[rhs].to_numpy())
        contrasts.append(c)
    result=dict(stage=record['stage'],exploratory=True,confirmatory=False,final_test_open=False,contrasts=contrasts,note='Engineering means are diagnostic only; development CIs are exploratory. No test or automatic proceed/unlock decision.')
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
