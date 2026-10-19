"""Consolidate accepted frozen prototype runs, including negative results."""
from pathlib import Path
import argparse,datetime,hashlib,json
import numpy as np,pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def main():
    a=argparse.ArgumentParser();a.add_argument('--engineering',required=True);a.add_argument('--development');args=a.parse_args()
    runs=[args.engineering]+([args.development] if args.development else []);lines=['# Thực thi prototype reliability + relative diversity — 17/10/2026','','Phiên bản thực thi `20261017_reliability_scaled_a10_v1`: chỉ đổi contract sang actual NVIDIA A10 sau khi A10G contract bị chặn trước inference. Thuật toán và tham số không đổi. Các kết quả dưới đây là exploratory trên dữ liệu đã tham gia quá trình thiết kế; không có confirmatory/test claim.'];records=[];hashes={}
    if args.development:
        dev=json.loads((ROOT/'results'/args.development/'analysis/report.json').read_text())
        c=next(c for c in dev['contrasts'] if c['proposed']=='scaled_fallback' and c['baseline']=='uncertainty_raw');ci=c['exploratory_bootstrap']['ci95']
        lines+=['',f"**Kết luận development:** prototype minus uncertainty/raw = {c['mean_delta']:.6f}, exploratory CI95% [{ci[0]:.6f}, {ci[1]:.6f}], n=9 nguồn. Hai ca engineering có gain nhưng tín hiệu không giữ được trên cohort rộng hơn. Không promote fallback này thành mặc định; không chỉnh tham số để che kết quả âm và không mở final test."]
    for run_id in runs:
        base=ROOT/'results'/run_id;report=json.loads((base/'analysis/report.json').read_text());accepted=json.loads((base/'analysis/cohort_acceptance.json').read_text());execution=json.loads((base/'cohort_execution.json').read_text())
        if not accepted['complete'] or execution['final_test'] or report['final_test_open']:raise ValueError('Incomplete/invalid scope')
        expected=12 if execution['stage']=='engineering' else 54
        if len(accepted['checks'])!=expected:raise ValueError('Wrong recipe/case count')
        auc=pd.read_csv(base/'analysis/recipe_auc.csv',dtype={'clip_id':str});means=auc.groupby('recipe')[['dice_auc','iou_auc']].mean()
        lines+=['',f"## {execution['stage']}: {len(execution['reports'])} nguồn, {expected} case-runs",'', '| Recipe | Mean Dice AUC | Mean IoU AUC |','|---|---:|---:|']
        for recipe,row in means.iterrows():lines.append(f'| {recipe} | {row.dice_auc:.6f} | {row.iou_auc:.6f} |')
        plot=base/'analysis/prototype_comparison.png'
        if plot.exists():lines+=['',f'![Exploratory prototype comparison](<{plot.as_posix()}>)','', 'Overview curves không có CI ribbon; paired endpoint CIs được ghi riêng trong bảng/JSON.']
        lines+=['','| Contrast | Δ Dice AUC | CI95% exploratory |','|---|---:|---|']
        for c in report['contrasts']:
            ci=c.get('exploratory_bootstrap',{}).get('ci95');interval=f'[{ci[0]:.6f}, {ci[1]:.6f}]' if ci else 'Không tính CI population cho engineering n=2'
            lines.append(f"| {c['proposed']} − {c['baseline']} | {c['mean_delta']:.6f} | {interval} |")
        harm=[];weight=[];trajectory=[]
        for r in execution['reports']:
            cid=r['clip_id'];suite=base/cid/'suite';prov=json.loads((suite/'suite_provenance.json').read_text())
            if prov['gpu']!='NVIDIA A10':raise ValueError('Mixed actual GPU hardware')
            for folder in sorted(suite.iterdir()):
                if not folder.is_dir():continue
                frame=pd.read_csv(folder/'per_frame.csv');zero=frame[frame.density==0][['structure','frame_index','dice']].rename(columns={'dice':'dice_zero'})
                nonzero=frame[frame.density>0].merge(zero,on=['structure','frame_index'],validate='many_to_one');nonzero['change_from_zero']=nonzero.dice-nonzero.dice_zero
                harm.extend(nonzero.assign(recipe=folder.name,clip_id=cid).to_dict('records'))
                q=pd.read_csv(folder/'queries.csv');trajectory.append(dict(clip_id=cid,recipe=folder.name,queries=q[['frame','structure']].to_numpy().tolist()))
                state=pd.read_csv(folder/'debug/state_metrics.csv');scores=pd.read_csv(folder/'debug/candidate_scores.csv')
                if len(state)!=72 or len(scores)!=20:raise ValueError('Incomplete debug state/candidate trace')
                for density,count in [(0,0),(1,1),(2,2),(4,5)]:
                    b=state[(state.pool=='score')&(state.query_count==count)].merge(frame[frame.density==density],left_on=['frame','structure'],right_on=['frame_index','structure'],suffixes=('_state','_run'),validate='one_to_one')
                    if len(b)!=6 or not np.allclose(b.dice_state,b.dice_run,rtol=0,atol=1e-12):raise ValueError('Debug/run metric mismatch')
                path=folder/'debug/reliability_weights.csv'
                if path.exists():
                    w=pd.read_csv(path);uncharged=w[~w.charged_prompt_override]
                    if not w.weight.between(0,1).all():raise ValueError('Trust weight out of bounds')
                    for k,b in uncharged.groupby('structure'):weight.append(dict(clip_id=cid,recipe=folder.name,structure=int(k),median_weight=float(b.weight.median()),q10=float(b.weight.quantile(.1)),q90=float(b.weight.quantile(.9))))
        pd.DataFrame(harm).to_csv(base/'analysis/correction_change_from_zero.csv',index=False)
        pd.DataFrame(weight).to_csv(base/'analysis/reliability_summary.csv',index=False)
        (base/'analysis/query_trajectories.json').write_text(json.dumps(trajectory,indent=2),encoding='utf8')
        primary=next(c for c in report['contrasts'] if c['proposed']=='scaled_fallback' and c['baseline']=='uncertainty_raw')
        lines+=['',f"Primary prototype delta = {primary['mean_delta']:.6f}. {'Có mean gain trong sample này; chưa chứng minh khái quát hóa.' if primary['mean_delta']>0 else 'Chưa có mean gain so với uncertainty/raw trong sample này.'}",'',f"[Analysis JSON](<{(base/'analysis/report.json').as_posix()}>) · [AUC theo recipe/nguồn](<{(base/'analysis/recipe_auc.csv').as_posix()}>) · [Propagation harm diagnostics](<{(base/'analysis/correction_change_from_zero.csv').as_posix()}>) · [Reliability summaries](<{(base/'analysis/reliability_summary.csv').as_posix()}>)"]
        records.append(dict(run_id=run_id,stage=execution['stage'],case_runs=expected,primary_delta=primary['mean_delta'],gpu='NVIDIA A10',debug_states_validated=True))
        sensitivity=base/'sensitivity_1s/analysis/report.json'
        if sensitivity.exists():
            sr=json.loads(sensitivity.read_text());contrast=next(c for c in sr['contrasts'] if c['proposed']=='scaled_fallback' and c['baseline']=='uncertainty_raw');ci=contrast['exploratory_bootstrap']['ci95']
            lines+=['',f"Sensitivity ±1s, scoring chung [99,124] (coverage 2/3): primary delta {contrast['mean_delta']:.6f}, exploratory CI95% [{ci[0]:.6f}, {ci[1]:.6f}]. Queries/budgets/propagation giữ nguyên; chỉ rescore outputs và không rerun model. [Sensitivity JSON](<{sensitivity.as_posix()}>)"]
        for p in base.rglob('*'):
            if p.is_file() and p.suffix in ['.csv','.json','.txt']:
                with p.open('rb') as f:hashes[p.relative_to(ROOT).as_posix()]=hashlib.file_digest(f,'sha256').hexdigest()
    if args.development:
        devbase=ROOT/'results'/args.development;diagnostic=devbase/'analysis/failure_diagnosis.json'
        if diagnostic.exists():
            diagnosis=json.loads(diagnostic.read_text());deltas=pd.read_csv(devbase/'analysis/source_deltas_ranked.csv',dtype={'clip_id':str})
            lines+=['','## Vì sao fallback thất bại?', '', '| Clip | Prototype − uncertainty AUC |','|---|---:|']
            for _,r in deltas.iterrows():lines.append(f"| {r.clip_id} | {r.prototype_delta:.6f} |")
            lines+=['','Hai mode thất bại khác nhau, đối chiếu theo nhãn official (không phải clinical adjudication):','','- **038:** uterus hiện diện trong cả ba scoring references. Baseline Dice=0; raw SAM sau correction đầu đạt mean Dice≈0.930, nhưng fallback vẫn Dice=0 ở mọi budget. Median trust weight trên scoring uterus≈0.023; baseline entropy≈0.051 và disagreement≈0.940. Agreement gate giữ baseline tự tin nhưng sai, chặn correction tốt.','- **138:** uterus reference rỗng ở cả ba scoring frames. Raw SAM loại foreground sai và đạt Dice=1 sau correction; fallback giữ foreground ở hai trong ba frames, mean Dice=0.333. Median trust weight≈0.045. Gate cũng giữ lỗi false positive của baseline.', '',f"Sensitivity present-only vẫn âm: mean prototype delta {diagnosis['present_only_mean_prototype_delta']:.6f} trên {diagnosis['present_only_n_sources']} nguồn. Vì vậy không thể giải thích toàn bộ thất bại chỉ bằng convention empty-mask; không thay endpoint chính bằng sensitivity.",'','Relative diversity/raw gần error/raw và chưa cho gain riêng; phần giảm lớn đến từ fallback. Đây là failure mode của heuristic agreement-as-trust, không phải lỗi runtime hoặc lý do để sửa scoring labels.']
            for cid in ['038','138']:
                for recipe in ['uncertainty_raw','scaled_fallback']:
                    image=devbase/cid/'suite'/recipe/'debug/query_02_score_124.png'
                    lines+=['',f'### Ca giảm mạnh {cid} / {recipe}','',f'![{cid} {recipe}](<{image.as_posix()}>)']
            lines+=['','Các ca trên chọn hậu nghiệm theo delta thấp nhất; bảng chính vẫn giữ đủ cả chín nguồn. Không dùng hình để kết luận hiện diện/absence lâm sàng ngoài phạm vi annotation được phát hành.']
    lines+=['','## Engineering before/after panels','','Các panels dưới là query2, scoring frame124 của ca059/086; chọn trước khi xem outcome v2 để đối chiếu hai cơ chế. Orange=uterus, cyan=tools. Không thay cho aggregate endpoints.']
    base=ROOT/'results'/args.engineering
    for cid in ['059','086']:
        for recipe in ['uncertainty_raw','scaled_raw','scaled_fallback']:
            image=base/cid/'suite'/recipe/'debug/query_02_score_124.png'
            with Image.open(image) as img:img.verify()
            lines+=['',f'### {cid} / {recipe}','',f'![{cid} {recipe} before after](<{image.as_posix()}>)']
    lines+=['','## Giới hạn','','41 tests pass trước GPU execution. Scoring và budget giữ nguyên; mọi prompt đều tính phí. Raw SAM outputs được cache trước fallback; weights chỉ dùng predictions và exact override chỉ dùng charged masks. Không có gold-based rejection hoặc parameter tuning. Không claim fallback luôn tốt hơn hoặc trust weights đã calibration.','','A10G contract và A10 contract từng bị chặn bởi cấp phát sai hardware; controller giữ ca đã hoàn tất và retry chỉ allocation mismatch. Các kết quả được nhận đều actual NVIDIA A10. Việc retry không dựa vào Dice. Snapshot/arrays đầy đủ giữ private Modal; development chỉ tải query1/query2-score124 panels để tiết kiệm ổ D, CSV/JSON đủ đã tải.','','Thiết kế được hình thành sau khi xem outcomes v1. Chín development sources và ba validation sources cũ đã được quan sát; không gọi chúng là unseen confirmation. Primary delta hoặc exploratory bootstrap CI không đủ để unlock test. Final test vẫn khóa; chưa train foundation model, chưa tạo nhãn mới hoặc nộp bài.']
    target=ROOT/'REPORT_RELIABILITY_EXECUTION_20261017.md';target.write_text('\n'.join(lines)+'\n',encoding='utf8')
    receipt=dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,runs=records,artifact_hashes=hashes,final_test_open=False,parameters_retuned=False,scope='Computational exploratory prototype execution; not clinical or confirmatory efficacy')
    (ROOT/'materials/verification/reliability_execution_acceptance.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps(dict(passed=True,runs=records,output=str(target)),indent=2))

if __name__=='__main__':main()
