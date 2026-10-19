"""Evaluate development-selected recipes on separate official-val sources."""
from pathlib import Path
import argparse,csv,json,subprocess,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);args=a.parse_args();base=ROOT/'results'/args.run
    record=json.loads((base/'cohort_execution.json').read_text());assert record['stage']=='validation' and record['final_test'] is False
    if sorted(r['clip_id'] for r in record['reports'])!=['171','190','211']:raise ValueError('Incomplete validation source cohort')
    cohort=base/'analysis_cohort.csv'
    with cohort.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=['clip_id','procedure_id']);w.writeheader();w.writerows({k:r[k] for k in ['clip_id','procedure_id']} for r in record['reports'])
    acceptance=base/'cohort_acceptance.json'
    if not acceptance.exists():subprocess.run([sys.executable,str(ROOT/'scripts/validate_cohort_results.py'),'--results-root',str(base),'--cohort',str(cohort),'--registry',str(ROOT/'configs/official_sparse_selected_v1/validation_registry.json'),'--output',str(acceptance)],check=True,cwd=ROOT)
    out=base/'analysis'
    if out.exists():raise FileExistsError(out)
    out.mkdir();comparisons=[]
    baseline=pd.concat([pd.read_csv(base/cid/'suite/uncertainty_seed20261016/per_clip.csv') for cid in ['171','190','211']]);baseline.to_csv(out/'uncertainty.csv',index=False)
    for name in ['best_development','interaction_nonzero']:
        data=pd.concat([pd.read_csv(base/cid/'suite'/(name+'_seed20261016')/'per_clip.csv') for cid in ['171','190','211']]);data.to_csv(out/(name+'.csv'),index=False)
        folder=out/name
        result=subprocess.run([sys.executable,'-m','src.analyze','--kind','surgery','--proposed',str(out/(name+'.csv')),'--baseline',str(out/'uncertainty.csv'),'--expected-seeds','20261016','--output',str(folder)],cwd=ROOT,capture_output=True,text=True)
        (out/(name+'_stdout.txt')).write_text(result.stdout);(out/(name+'_stderr.txt')).write_text(result.stderr)
        if result.returncode:raise RuntimeError(result.stderr[-4000:])
        comparisons.append(dict(recipe=name,**json.loads((folder/'statistics.json').read_text())))
    report=dict(scope='Three official-validation source procedures, one fixed clip each; not final test',comparisons=comparisons,
       selected_on_development=True,validation_used_to_retune=False,final_test_open=False,
       limitation='Small feasibility validation, not a sample-size justification or full 57-clip evaluation')
    (out/'validation_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
