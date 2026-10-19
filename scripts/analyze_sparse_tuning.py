"""Select on development only, report degeneracy and preserve all grid results."""
from pathlib import Path
import argparse,json,csv,sys,subprocess
import pandas as pd,numpy as np
from scipy.integrate import trapezoid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import paired_bootstrap
def main():
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);a.add_argument('--baseline-run',default='official_sparse_feasibility_20261016T144620Z');args=a.parse_args()
    base=ROOT/'results'/args.run;record=json.loads((base/'cohort_execution.json').read_text());assert record['stage']=='tuning' and record['final_test'] is False
    expected=list(csv.DictReader((ROOT/'data/manifests/sequence_official_sparse_v1/development_feasibility_cohort.csv').open(encoding='utf8')))
    if {r['clip_id'] for r in record['reports']}!={r['clip_id'] for r in expected}:raise ValueError('Incomplete tuning cohort')
    cohort=base/'analysis_cohort.csv'
    with cohort.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=['clip_id','procedure_id']);w.writeheader();w.writerows({k:r[k] for k in ['clip_id','procedure_id']} for r in expected)
    for r in record['reports']:
        original=json.loads((ROOT/'results'/args.baseline_run/r['clip_id']/'suite/suite_provenance.json').read_text())
        if r['input_sha256']!=original['input_sha256']:raise ValueError('Tuning and baseline inputs differ')
    acceptance=base/'cohort_acceptance.json'
    if not acceptance.exists():subprocess.run([sys.executable,str(ROOT/'scripts/validate_cohort_results.py'),'--results-root',str(base),'--cohort',str(cohort),'--registry',str(ROOT/'configs/official_sparse_tuning_v1/experiment_registry.json'),'--output',str(acceptance)],check=True,cwd=ROOT)
    out=base/'analysis'
    if out.exists():raise FileExistsError(out)
    out.mkdir();reference=pd.read_csv(ROOT/'results'/args.baseline_run/'analysis/per_procedure_all_policy_auc.csv');reference=reference[reference.policy=='uncertainty'].set_index('procedure_id').dice_auc
    registry=json.loads((ROOT/'configs/official_sparse_tuning_v1/experiment_registry.json').read_text());grid=[];all_case=[]
    for recipe in registry:
        cfg=json.loads((ROOT/recipe['config']).read_text());values={}
        for r in expected:
            df=pd.read_csv(base/r['clip_id']/'suite'/recipe['run_id']/'per_clip.csv');curve=df.groupby('density').agg(dice=('dice','mean'),actual_density=('actual_density','first')).sort_values('actual_density');x=curve.actual_density.to_numpy()
            values[r['procedure_id']]=float(trapezoid(curve.dice,x)/(x[-1]-x[0]));all_case.append(dict(run_id=recipe['run_id'],procedure_id=r['procedure_id'],dice_auc=values[r['procedure_id']]))
        ids=sorted(values);stats=paired_bootstrap([values[p] for p in ids],reference.reindex(ids).to_numpy())
        grid.append(dict(run_id=recipe['run_id'],alpha=cfg['alpha'],beta=cfg['beta'],procedures=len(values),mean_dice_auc=float(np.mean(list(values.values()))),delta_vs_uncertainty=stats['delta'],ci95_low=stats['ci95'][0],ci95_high=stats['ci95'][1]))
    ranked=sorted(grid,key=lambda r:(-r['mean_dice_auc'],r['alpha'],r['beta']));best=ranked[0];nonzero=next(r for r in ranked if r['alpha']>0 and r['beta']>0)
    pd.DataFrame(ranked).to_csv(out/'all_grid_results.csv',index=False);pd.DataFrame(all_case).to_csv(out/'all_grid_per_procedure.csv',index=False)
    folder=ROOT/'configs/official_sparse_selected_v1';folder.mkdir(exist_ok=True);validation=[]
    for label,row in [('best_development',best),('interaction_nonzero',nonzero)]:
        cfg=json.loads((ROOT/f"configs/official_sparse_tuning_v1/{row['run_id']}.json").read_text());path=folder/(label+'_seed20261016.json');path.write_text(json.dumps(cfg,indent=2));validation.append(dict(run_id=label+'_seed20261016',config=path.relative_to(ROOT).as_posix()))
    path=folder/'uncertainty_seed20261016.json';path.write_text((ROOT/'configs/official_sparse_v1/uncertainty_seed20261016.json').read_text());validation.append(dict(run_id='uncertainty_seed20261016',config=path.relative_to(ROOT).as_posix()))
    (folder/'validation_registry.json').write_text(json.dumps(validation,indent=2))
    selection=dict(scope='Development selection, not confirmatory inference',source_run=args.run,baseline_run=args.baseline_run,best_overall=best,best_both_nonzero=nonzero,
       interaction_supported_by_global_best=best['alpha']>0 and best['beta']>0,grid_selection_adjusted_ci=False,
       global_best_development_delta_positive=best['delta_vs_uncertainty']>0,
       proceed_gate_met_by_development=False,final_test_unlocked=False,
       note='Selected-grid CIs are descriptive, not selection-adjusted. Validate fixed selected recipes on separate official-val sources; never tune on final test.')
    (out/'selection.json').write_text(json.dumps(selection,indent=2));(folder/'selection_evidence.json').write_text(json.dumps(selection,indent=2));print(json.dumps(selection,indent=2))
if __name__=='__main__':main()
