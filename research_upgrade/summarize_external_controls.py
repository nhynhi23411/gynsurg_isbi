"""Source-equal descriptive external analysis; no method fitting or selection."""
from pathlib import Path
import csv,datetime,json
import numpy as np
from research_upgrade.external_scope import AUC_WEIGHTS
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/research_upgrade_external_controls_20261020_v1'

def source_statistics(rows,bootstrap=10000):
    sources=sorted({r['source_video'] for r in rows});policies=['uncertainty','uniform','random']
    source_rows=[]
    for s in sources:
        for p in policies:
            for n in range(4):
                cells=[r for r in rows if r['source_video']==s and r['policy']==p and r['n_queries']==n]
                if not cells or any(r['dice'] is None or r['iou'] is None for r in cells):raise ValueError('Unavailable endpoint; no silent cohort exclusion')
                source_rows.append(dict(source_video=s,policy=p,n_queries=n,windows=len(cells),
                    dice=float(np.mean([r['dice'] for r in cells])),iou=float(np.mean([r['iou'] for r in cells]))))
    curves=[];auc={}
    for p in policies:
        for n in range(4):
            cells=[r for r in source_rows if r['policy']==p and r['n_queries']==n]
            curves.append(dict(policy=p,n_queries=n,actual_density=1.25*n,
                dice=float(np.mean([r['dice'] for r in cells])),iou=float(np.mean([r['iou'] for r in cells]))))
        auc[p]=np.array([sum(AUC_WEIGHTS[r['n_queries']]*r['dice'] for r in source_rows if r['source_video']==s and r['policy']==p) for s in sources])
    rng=np.random.default_rng(20261020);draw=rng.integers(len(sources),size=(bootstrap,len(sources)))
    contrasts=[]
    for other in ['uniform','random']:
        delta=auc['uncertainty']-auc[other];ci=np.quantile(delta[draw].mean(axis=1),[.025,.975])
        contrasts.append(dict(contrast='uncertainty_minus_'+other,delta_auc=float(delta.mean()),
            lower95=float(ci[0]),upper95=float(ci[1]),positive_sources=int((delta>0).sum()),sources=len(sources)))
    auc_rows=[dict(source_video=s,policy=p,dice_auc=float(auc[p][i])) for i,s in enumerate(sources) for p in policies]
    return dict(source_curves=source_rows,source_auc=auc_rows,mean_curves=curves,contrasts=contrasts,
        mean_auc={p:float(v.mean()) for p,v in auc.items()},sources=len(sources))

def collect_rows(folder):
    rows=[];sensitivity=[];missing=[];window_ids=[]
    for i in range(101):
        path=folder/f'{i:03d}';meta=json.loads((path/'execution.json').read_text());x=json.loads((path/'controls.json').read_text())
        window_ids.append(i);labels={(r['state'],r['frame']):r for r in x['evaluation']}
        for policy in ['uncertainty','uniform','random']:
            for n in range(4):
                seeds=[r for r in x['curves'] if r['policy']==policy and r['n_queries']==n]
                expected=10 if policy=='random' else 1
                if len(seeds)!=expected:raise ValueError('Incomplete selector seeds')
                row=dict(window_index=i,source_video=meta['source_video'],policy=policy,n_queries=n,
                    dice=float(np.mean([r['dice'] for r in seeds])) if all(r['dice'] is not None for r in seeds) else None,
                    iou=float(np.mean([r['iou'] for r in seeds])) if all(r['iou'] is not None for r in seeds) else None,
                    scored_frames=seeds[0]['scored_frames'],missing_frames=seeds[0]['missing_frames'])
                rows.append(row)
                if row['missing_frames']:missing.append(row)
                present_seed=[]
                for trace in [r for r in x['traces'] if r['policy']==policy]:
                    cells=[labels[trace['paths'][n],t] for t in [47,63,79]]
                    present=[r['dice'] for r in cells if r['reference_present'] and r['dice'] is not None]
                    if present:present_seed.append(float(np.mean(present)))
                sensitivity.append(dict(window_index=i,source_video=meta['source_video'],policy=policy,n_queries=n,
                    present_only_dice=float(np.mean(present_seed)) if present_seed else None,
                    reference_empty_frames=sum(r['foreground_pixels']==0 and r['valid_pixels']>0 for r in x['scoring_reference_summary'])))
    return rows,sensitivity,missing

def write_csv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    if not all((OUT/f'{i:03d}/execution.json').exists() for i in range(101)):raise ValueError('Complete external cohort required; no interim efficacy analysis')
    rows,sensitivity,missing=collect_rows(OUT);summary=source_statistics(rows)
    transitions=[];fractions=[]
    for p in ['uncertainty','uniform','random']:
        curves={i:[next(r['dice'] for r in rows if r['window_index']==i and r['policy']==p and r['n_queries']==n) for n in range(4)] for i in range(101)}
        for i,curve in curves.items():
            for a,b in [(1,2),(2,3)]:
                delta=curve[b]-curve[a];transitions.append(dict(window_index=i,policy=p,transition=f'{a}_to_{b}',delta_dice=delta,harmful=delta<0,decrease_gt_002=delta<-.02))
            gain=curve[3]-curve[0]
            fractions.append(dict(window_index=i,policy=p,total_gain=gain,positive_denominator=gain>0,
                first_mask_fraction=(curve[1]-curve[0])/gain if gain>0 else None))
    summary.update(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),windows=101,
        exploratory=True,learned_candidate=False,final_test=False,bootstrap_draws=10000,bootstrap_seed=20261020,
        missing_scoring_frame_rows=missing,scope='Release-annotated grasper/L-hook union; broader baseline head; void excluded',
        total_worker_hours=sum(json.loads((OUT/f'{i:03d}/execution.json').read_text())['total_worker_seconds'] for i in range(101))/3600,
        note='Counterfactual bank compute, not deployed policy latency or billed GPU-hours')
    dest=OUT/'analysis';dest.mkdir(exist_ok=True)
    for name,data in [('window_curves',rows),('source_curves',summary['source_curves']),('source_auc',summary['source_auc']),('mean_curves',summary['mean_curves']),('contrasts',summary['contrasts']),('present_sensitivity',sensitivity),('budget_transitions',transitions),('first_mask_fractions',fractions)]:write_csv(dest/(name+'.csv'),data)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({k:summary[k] for k in ['windows','sources','mean_auc','contrasts']},indent=2))

if __name__=='__main__':main()
