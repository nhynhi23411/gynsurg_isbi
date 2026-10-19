"""Accept complete all-training raw outputs and aggregate by verified source."""
from pathlib import Path
import hashlib,json,sys,subprocess,csv
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import paired_bootstrap
P=ROOT/'protocols/20261017_train_expansion_v1'
BASE=Path('C:/Users/pc/ISBI_GynSurg_runs/train_expansion_20261017_v1')

def source_summary(clip_auc):
    """Clips equal within source; sources equal, irrespective of clip count."""
    return clip_auc.groupby('procedure_id')[['raw_dice_auc','baseline_dice_auc','raw_iou_auc','baseline_iou_auc']].mean()

def main():
    freeze=json.loads((P/'FREEZE.json').read_text())
    for name,h in freeze['sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise ValueError('Freeze changed: '+name)
    inputs=json.loads((P/'inputs.json').read_text());execution=json.loads((BASE/'cohort_execution.json').read_text())
    if execution['final_test'] or len(execution['reports'])!=170 or {r['clip_id'] for r in execution['reports']}!={r['clip_id'] for r in inputs}:raise ValueError('Incomplete/wrong cohort')
    out=BASE/'analysis';out.mkdir(exist_ok=False)
    cohort=out/'cohort.csv'
    with cohort.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['clip_id','procedure_id']);w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in inputs)
    registry=out/'registry.json';registry.write_text(json.dumps([dict(run_id='uncertainty_raw',config='configs/reliability_scaled_a10_v1/uncertainty_raw.json')]))
    subprocess.run([sys.executable,str(ROOT/'scripts/validate_cohort_results.py'),'--results-root',str(BASE),'--cohort',str(cohort),'--registry',str(registry),'--output',str(out/'cohort_acceptance.json')],check=True)
    rows=[];curves=[];freeze_sha=hashlib.sha256((P/'FREEZE.json').read_bytes()).hexdigest()
    for expected in inputs:
        cid=expected['clip_id'];folder=BASE/cid
        record=json.loads((folder/'execution.json').read_text());receipt=next(r for r in execution['reports'] if r['clip_id']==cid)
        if record!=receipt or record['official_split']!='train' or record['actual_gpu']!='NVIDIA A10' or record['procedure_id']!=expected['procedure_id'] or record['freeze_sha256']!=freeze_sha:raise ValueError('Execution evidence mismatch')
        suite=json.loads((folder/'suite/suite_provenance.json').read_text())
        if suite['gpu']!='NVIDIA A10' or suite['input_sha256']!=record['input_sha256']:raise ValueError('Suite provenance mismatch')
        if suite['sam_checkpoint_sha256']!=freeze['sha256']['materials/checkpoints/sam2.1_hiera_tiny.pt']:raise ValueError('Checkpoint changed')
        for name,h in suite['source_hashes'].items():
            if freeze['sha256'][name]!=h:raise ValueError('Source hashes differ')
        df=pd.read_csv(folder/'suite/uncertainty_raw/per_frame.csv')
        if len(df)!=24 or set(df.frame_index)!={74,99,124} or set(df.n_queries)!={0,1,2,5} or set(df.structure)!={0,1}:raise ValueError('Wrong scoring/budget grid')
        if not np.allclose(df.actual_density,df.n_queries*.8,rtol=0,atol=1e-12):raise ValueError('Density mismatch')
        curve=df.groupby('n_queries')[['dice','iou']].mean().reindex([0,1,2,5])
        weights=np.array([.1,.2,.4,.3])
        rows.append(dict(clip_id=cid,procedure_id=record['procedure_id'],reused=record['reused'],raw_dice_auc=float(weights@curve.dice),baseline_dice_auc=float(curve.dice.iloc[0]),raw_iou_auc=float(weights@curve.iou),baseline_iou_auc=float(curve.iou.iloc[0])))
        for count,r in curve.iterrows():curves.append(dict(clip_id=cid,procedure_id=record['procedure_id'],n_queries=int(count),actual_density=count*.8,dice=float(r.dice),iou=float(r.iou)))
    clips=pd.DataFrame(rows);sources=source_summary(clips);sources['delta']=sources.raw_dice_auc-sources.baseline_dice_auc
    if len(sources)!=9 or int(clips.reused.sum())!=9:raise ValueError('Wrong source/reuse counts')
    contrast=paired_bootstrap(sources.raw_dice_auc.to_numpy(),sources.baseline_dice_auc.to_numpy(),seed=20261017)
    new_sources=source_summary(clips[~clips.reused])
    sensitivity=paired_bootstrap(new_sources.raw_dice_auc.to_numpy(),new_sources.baseline_dice_auc.to_numpy(),seed=20261017)
    clips.to_csv(out/'per_clip_auc.csv',index=False);sources.to_csv(out/'per_source_auc.csv');pd.DataFrame(curves).to_csv(out/'per_clip_curves.csv',index=False)
    pd.DataFrame(curves).groupby(['procedure_id','n_queries','actual_density'])[['dice','iou']].mean().to_csv(out/'per_source_curves.csv')
    summary=dict(complete=True,exploratory=True,independent_validation=False,clips=170,new_inference_clips=161,reused_clips=9,source_groups=9,final_test_open=False,mean_raw_dice_auc=float(sources.raw_dice_auc.mean()),mean_baseline_dice_auc=float(sources.baseline_dice_auc.mean()),raw_minus_budget_zero=contrast,exclude_prior_representatives_sensitivity=sensitivity,positive_sources=int((sources.delta>1e-12).sum()),default_policy='uncertainty_raw',efficacy_of_proposed_scheduler=False,freeze_sha256=freeze_sha)
    (out/'report.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    lines=['# Mở rộng baseline: toàn bộ 170 clip training','','Hoàn tất 161 clip inference mới và tái sử dụng chín kết quả raw A10 đã kiểm chứng; tổng170 clip thuộc chín nguồn đã quan sát. Không phải170 ca độc lập hoặc chín nguồn mới. Không thay policy, train model, tune gate, mở validation/final test hoặc tạo nhãn mới.','',f"Source-macro raw Dice AUC **{summary['mean_raw_dice_auc']:.6f}**, budget-zero baseline **{summary['mean_baseline_dice_auc']:.6f}**. Raw−baseline **{contrast['delta']:+.6f}**, CI mô tả source-bootstrap95% [{contrast['ci95'][0]:+.6f}, {contrast['ci95'][1]:+.6f}], n=9 nguồn.",'','| Nguồn | Số clip | Baseline AUC | Raw AUC | Delta |','|---|---:|---:|---:|---:|']
    for pid,r in sources.iterrows():lines.append(f"| {pid} | {int((clips.procedure_id==pid).sum())} | {r.baseline_dice_auc:.6f} | {r.raw_dice_auc:.6f} | {r.delta:+.6f} |")
    lines+=['',f"Sensitivity bỏ chín clip đại diện cũ, giữ161 clip: source-macro delta **{sensitivity['delta']:+.6f}**, CI mô tả [{sensitivity['ci95'][0]:+.6f}, {sensitivity['ci95'][1]:+.6f}]. Vẫn cùng nguồn nên không gọi là validation độc lập.",'','Macro cấu trúc/frame → clip → nguồn, sau đó trung bình ngang trọng số nguồn. CI resample nguồn, không resample frame/clip như ca độc lập. Budget-zero baseline là đường hằng của dự đoán frozen chưa correction; contrast đo lợi ích charged correction so với không correction, chưa chứng minh scheduling mới hơn uncertainty. Empty-reference convention giữ nguyên. Không có clinical review, annotation-time study hay clinical utility claim.','',f'Outputs lớn giữ private Modal volume `isbi-gynsurg-results-v1/train_expansion_20261017_v1`. CSV/JSON/logs local: `{BASE.as_posix()}`. Freeze protocol/input/code trước dispatch: `protocols/20261017_train_expansion_v1/FREEZE.json`. Final test khóa.']
    (ROOT/'REPORT_TRAIN_EXPANSION_20261017.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    files=[p for p in BASE.rglob('*') if p.is_file() and p.suffix in ['.csv','.json','.txt'] and p!=out/'audit.json']
    (out/'audit.json').write_text(json.dumps(dict(passed=True,clips=170,source_groups=9,frame_rows=4080,aggregate_rows=1360,final_test_open=False,artifact_sha256={p.relative_to(BASE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
