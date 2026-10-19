"""Reproduce development curves and secondary summaries; never calls them test."""
from pathlib import Path
import argparse,json,sys,subprocess
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import paired_bootstrap
def main():
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);args=a.parse_args();base=ROOT/'results'/args.run
    out=base/'analysis'
    if out.exists():raise FileExistsError(out)
    record=json.loads((base/'cohort_execution.json').read_text())
    if record.get('final_test') is not False:raise ValueError('This analyzer is development-only')
    import csv
    cohort=base/'analysis_cohort.csv'
    expected=['058'] if record['stage']=='pilot' else [r['clip_id'] for r in csv.DictReader((ROOT/'data/manifests/sequence_official_sparse_v1/development_feasibility_cohort.csv').open(encoding='utf8'))]
    if sorted(r['clip_id'] for r in record['reports'])!=sorted(expected):raise ValueError('Incomplete declared development cohort')
    with cohort.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['clip_id','procedure_id']);w.writeheader();w.writerows({k:r[k] for k in ['clip_id','procedure_id']} for r in record['reports'])
    acceptance=base/'cohort_acceptance.json'
    if not acceptance.exists():
        subprocess.run([sys.executable,str(ROOT/'scripts/validate_cohort_results.py'),'--results-root',str(base),'--cohort',str(cohort),'--registry',str(ROOT/'configs/official_sparse_v1/experiment_registry.json'),'--output',str(acceptance)],check=True,cwd=ROOT)
    out.mkdir()
    files=sorted(base.glob('*/suite/*/per_clip.csv'));framefiles=sorted(base.glob('*/suite/*/per_frame.csv'))
    if not files or len(files)!=len(framefiles):raise ValueError('Missing result tables')
    metrics=pd.concat([pd.read_csv(p,dtype={'clip_id':str,'procedure_id':str}) for p in files],ignore_index=True)
    frames=pd.concat([pd.read_csv(p,dtype={'clip_id':str,'procedure_id':str}) for p in framefiles],ignore_index=True)
    for policy,name in [('full','proposed'),('uncertainty','baseline')]:metrics[metrics.policy==policy].to_csv(out/(name+'.csv'),index=False)
    cmd=[sys.executable,'-m','src.analyze','--kind','surgery','--proposed',str(out/'proposed.csv'),'--baseline',str(out/'baseline.csv'),'--expected-seeds','20261016','--output',str(out/'primary_development')]
    completed=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    (out/'analyzer_stdout.txt').write_text(completed.stdout);(out/'analyzer_stderr.txt').write_text(completed.stderr)
    if completed.returncode:raise RuntimeError('Development analyzer failed: '+completed.stderr[-3000:])
    # Keep source-level curves with equal structure then clip weighting; seeds averaged.
    clip=metrics.groupby(['policy','procedure_id','clip_id','seed','density']).agg(dice=('dice','mean'),iou=('iou','mean'),actual_density=('actual_density','first')).reset_index()
    procedure=clip.groupby(['policy','procedure_id','seed','density']).agg(dice=('dice','mean'),iou=('iou','mean'),actual_density=('actual_density','mean')).reset_index()
    curve=procedure.groupby(['policy','procedure_id','density']).agg(dice=('dice','mean'),iou=('iou','mean'),actual_density=('actual_density','mean')).reset_index()
    curve.to_csv(out/'per_procedure_curves.csv',index=False)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4.5))
    for policy,block in curve.groupby('policy'):
        avg=block.groupby('density').agg(dice=('dice','mean'),actual_density=('actual_density','mean'))
        ax.plot(avg.actual_density,avg.dice,marker='o',label=policy.replace('_',' '),linewidth=2 if policy in ['full','uncertainty'] else 1)
    ax.set(xlabel='Full-mask corrections per 100 raw frames',ylabel='Procedure macro Dice',title='Development feasibility — not final test')
    ax.legend(fontsize=8,ncol=2);fig.tight_layout()
    for suffix in ['png','pdf','svg']:fig.savefig(out/('all_policy_development_curves.'+suffix),dpi=180)
    plt.close(fig)
    auc=[]
    from scipy.integrate import trapezoid
    for (policy,pid),block in curve.groupby(['policy','procedure_id']):
        block=block.sort_values('actual_density');x=block.actual_density.to_numpy()
        auc.append(dict(policy=policy,procedure_id=pid,dice_auc=float(trapezoid(block.dice,x)/(x[-1]-x[0])),iou_auc=float(trapezoid(block.iou,x)/(x[-1]-x[0]))))
    pd.DataFrame(auc).to_csv(out/'per_procedure_all_policy_auc.csv',index=False)
    # Prespecified reference geometry, computed only in evaluation. No clinical strata.
    frames['distance_bin']=pd.cut(frames.nearest_same_structure_query_s,[0,1,2,4,np.inf],right=True,include_lowest=True).astype(str)
    frames.loc[frames.nearest_same_structure_query_s.isna(),'distance_bin']='no_same_structure_query'
    frames['gold_proximity_stratum']=np.where(frames.gold_proximity_normalized.isna(),'undefined',np.where(frames.gold_proximity_normalized<=.03,'distance_le_0.03','distance_gt_0.03'))
    summaries=[]
    for field in ['distance_bin','gold_proximity_stratum']:
        perclip=frames.groupby(['policy','procedure_id','clip_id','seed','density','structure',field]).agg(dice=('dice','mean'),iou=('iou','mean'),n_frames=('frame_index','size')).reset_index()
        perclip=perclip.groupby(['policy','procedure_id','clip_id','seed','density',field]).agg(dice=('dice','mean'),iou=('iou','mean'),n_frames=('n_frames','sum')).reset_index()
        proc=perclip.groupby(['policy','procedure_id','seed','density',field]).agg(dice=('dice','mean'),iou=('iou','mean'),n_frames=('n_frames','sum')).reset_index()
        proc=proc.groupby(['policy','procedure_id','density',field]).agg(dice=('dice','mean'),iou=('iou','mean'),n_frames=('n_frames','mean')).reset_index()
        proc.rename(columns={field:'stratum'}).assign(analysis=field).to_csv(out/(field+'_per_procedure.csv'),index=False)
        summaries.append(field)
    present=frames[frames.gold_present.astype(str).str.lower()=='true']
    present.groupby(['policy','procedure_id','clip_id','seed','density','structure']).agg(dice=('dice','mean'),iou=('iou','mean'),n_present_frames=('frame_index','size')).reset_index().to_csv(out/'present_only_clip_metrics.csv',index=False)
    report=dict(scope='Exploratory development only; not held-out final test',clips=int(metrics.clip_id.nunique()),procedures=int(metrics.procedure_id.nunique()),policies=sorted(metrics.policy.unique()),metric_rows=len(metrics),frame_rows=len(frames),
      primary_development=json.loads((out/'primary_development/statistics.json').read_text()),secondary=summaries,
      clinician_difficulty_labels_available=False,failure_threshold_frozen=False,clinical_benefit_measured=False,
      warning='Development estimates inform go/no-go; do not report them as independent final-test evidence. Stratum procedure counts may differ. A one-procedure bootstrap interval is degenerate and does not estimate population uncertainty.')
    (out/'development_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
