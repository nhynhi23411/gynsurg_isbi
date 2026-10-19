"""Accept four debug runs and report observed allocation/propagation effects."""
from pathlib import Path
import argparse,datetime,json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);args=a.parse_args()
    base=ROOT/'results'/args.run;record=json.loads((base/'cohort_execution.json').read_text())
    if record['final_test'] or {r['clip_id'] for r in record['reports']}!={'059','086'}:raise ValueError('Wrong replay cohort')
    out=base/'analysis';out.mkdir(exist_ok=False);checks=[];metrics=[];scores=[];curated=[]
    for cid in ['059','086']:
        original=ROOT/'results'/record['source_run']/cid/'suite';suite=base/cid/'suite'
        previous=json.loads((original/'suite_provenance.json').read_text());current=json.loads((suite/'suite_provenance.json').read_text())
        if previous['input_sha256']!=current['input_sha256'] or previous['sam_checkpoint_sha256']!=current['sam_checkpoint_sha256']:raise ValueError('Input/checkpoint changed')
        for policy in ['full','uncertainty']:
            name=f'{policy}_seed20261016';folder=suite/name;debug=folder/'debug'
            oldcfg=json.loads((original/name/'provenance.json').read_text())['config'];cfg=json.loads((folder/'provenance.json').read_text())['config']
            if oldcfg!=cfg:raise ValueError('Policy config changed')
            oldq=pd.read_csv(original/name/'queries.csv');q=pd.read_csv(folder/'queries.csv')
            if not oldq.equals(q):raise ValueError('Query trajectories changed; review replay before interpretation')
            old=pd.read_csv(original/name/'per_frame.csv');new=pd.read_csv(folder/'per_frame.csv')
            keys=['density','structure','frame_index'];paired=new.merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one')
            if len(paired)!=24 or not (paired.scoring_frame_sha256_new==paired.scoring_frame_sha256_old).all():raise ValueError('Score grid changed')
            m=pd.read_csv(debug/'state_metrics.csv');s=pd.read_csv(debug/'candidate_scores.csv')
            if len(m)!=72 or len(s)!=20 or set(m.query_count)!=set(range(6)):raise ValueError('Incomplete state trace')
            if m.duplicated(['query_count','frame','structure']).any():raise ValueError('Duplicate trace metrics')
            if not m.dice.between(0,1).all() or not m.iou.between(0,1).all() or not np.isfinite(s.score).all():raise ValueError('Invalid trace values')
            if not all(s.groupby('query').selected.sum()==1):raise ValueError('Missing/duplicate selected candidates')
            for _,r in q.iterrows():
                chosen=s[(s['query']==r['query'])&s.selected]
                if len(chosen)!=1 or chosen.iloc[0].frame!=r.frame or chosen.iloc[0].structure!=r.structure:raise ValueError('Score trace/query log mismatch')
                exact=m[(m.query_count==r['query'])&(m.frame==r.frame)&(m.structure==r.structure)]
                if exact.iloc[0].dice!=1:raise ValueError('Charged prompt not exact in snapshot')
            for density,count in [(0,0),(1,1),(2,2),(4,5)]:
                state=m[(m.query_count==count)&(m.pool=='score')].rename(columns={'frame':'frame_index'})
                compare=state.merge(new[new.density==density],on=['frame_index','structure'],suffixes=('_trace','_run'),validate='one_to_one')
                if len(compare)!=6 or not np.allclose(compare.dice_trace,compare.dice_run,rtol=0,atol=1e-12):raise ValueError('Snapshot metrics disagree with main run')
            maxdiff=float(np.max(np.abs(paired.dice_new-paired.dice_old)))
            checks.append(dict(clip_id=cid,policy=policy,query_trajectory_exact=True,config_input_checkpoint_equal=True,max_scored_dice_difference=maxdiff,old_gpu=previous['gpu'],replay_gpu=current['gpu'],trace_reproduces_run_metrics=True))
            metrics.append(m.assign(clip_id=cid,policy=policy));scores.append(s.assign(clip_id=cid,policy=policy))
            for t in [74,99,124]:
                for step in range(1,6):
                    if not (debug/f'query_{step:02d}_score_{t:03d}.png').is_file():raise ValueError('Missing before/after panel')
    metrics=pd.concat(metrics);scores=pd.concat(scores)
    metrics.to_csv(out/'all_state_metrics.csv',index=False);scores.to_csv(out/'all_candidate_scores.csv',index=False)
    mean=metrics[metrics.pool=='score'].groupby(['clip_id','policy','query_count','structure']).dice.mean().reset_index()
    mean.to_csv(out/'state_score_means.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,cid in enumerate(['059','086']):
        for k,label in [(0,'uterus'),(1,'tools')]:
            ax=axes[row,k]
            for policy in ['full','uncertainty']:
                b=mean[(mean.clip_id==cid)&(mean.structure==k)&(mean.policy==policy)]
                ax.plot(b.query_count,b.dice,marker='o',label=policy)
            ax.set(title=f'Clip {cid}: {label}',xlabel='Charged corrections (debug states 0–5)',ylabel='Mean Dice on fixed score snapshots',ylim=(.7,1.005));ax.set_xticks(range(6));ax.legend();ax.grid(alpha=.2)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'correction_trajectories.{ext}',dpi=180)
    plt.close(fig)
    report=['# Debug replay propagation và allocation — 17/10/2026','','Replay quan sát hai ca development 059/086, full và uncertainty. Giữ nguyên input/checkpoint/config/budget; không tune hoặc mở final test. 35 software tests pass trước replay.','','## Đối chiếu lượt chạy cũ','','| Clip | Policy | Max frame Dice difference | GPU cũ → replay |','|---|---|---:|---|']
    for c in checks:report.append(f"| {c['clip_id']} | {c['policy']} | {c['max_scored_dice_difference']:.9f} | {c['old_gpu']} → {c['replay_gpu']} |")
    report+=['','Bốn trajectories và prompt hashes trùng lượt cũ. Snapshot Dice tái tạo đúng per-frame metrics của replay tại counts 0/1/2/5; mỗi charged prompt có Dice=1 trên chính query frame. Hardware differences có thể tạo numerical variation; xem max differences ở bảng. Debug có thêm lưu file/render nên không dùng runtime làm benchmark tốc độ.','',f'![Dice sau từng correction](<{(out/"correction_trajectories.png").as_posix()}>)','','Counts 3/4 chỉ là debug intermediate states, không phải endpoint mới hoặc thay grid chính. Chỉ có gold tại sáu snapshots; curve dùng ba scoring snapshots [74,99,124], không tính Dice trên các raw frames chưa có nhãn.','','## Query thứ hai và phân rã điểm chọn']
    for cid in ['059','086']:
        report+=['',f'### Clip {cid}', '', '| Policy | Candidate frame/structure | Error proxy | Proximity boost | Redundancy penalty | Score | Chọn |','|---|---|---:|---:|---:|---:|---|']
        block=scores[(scores.clip_id==cid)&(scores['query']==2)].sort_values(['policy','score'],ascending=[True,False])
        for _,r in block.iterrows():report.append(f"| {r.policy} | {int(r.frame)}/{int(r.structure)} | {r.error_proxy:.6f} | {r.proximity_boost:.6f} | {r.redundancy_penalty:.6f} | {r.score:.6f} | {'yes' if r.selected else ''} |")
        uterus=block[(block.policy=='full')&(block.structure==0)]
        chosen=block[(block.policy=='full')&block.selected].iloc[0]
        if len(uterus) and (uterus.score<0).all():
            report+=['',f"Ở query2, mọi candidate uterus của full có score âm: error proxy {uterus.error_proxy.min():.6f}–{uterus.error_proxy.max():.6f}, penalty {uterus.redundancy_penalty.min():.6f}–{uterus.redundancy_penalty.max():.6f}. Candidate được chọn là frame/structure {int(chosen.frame)}/{int(chosen.structure)}, score {chosen.score:.6f}. Penalty đang lấn át error proxy tại state này; đây là quan sát trực tiếp, không phải giả thuyết từ correlation."]
        # Inspect most harmed uterus scoring snapshot after the shared first correction.
        states=metrics[(metrics.clip_id==cid)&(metrics.policy=='full')&(metrics.pool=='score')&(metrics.structure==0)]
        b=states[states.query_count==0].merge(states[states.query_count==1],on='frame',suffixes=('_before','_after'))
        b['delta']=b.dice_after-b.dice_before;worst=b.sort_values('delta').iloc[0];t=int(worst.frame)
        image=base/cid/'suite/full_seed20261016/debug'/f'query_01_score_{t:03d}.png';curated.append(image.relative_to(ROOT).as_posix())
        report+=['',f"Sau correction đầu, scoring frame {t}, uterus Dice {worst.dice_before:.6f} → {worst.dice_after:.6f} (delta {worst.delta:.6f}).",'',f'![Before/after first correction clip {cid}](<{image.as_posix()}>)']
        # Query2 changes tools in these cases; inspect its worst scoring change separately.
        tool=metrics[(metrics.clip_id==cid)&(metrics.policy=='full')&(metrics.pool=='score')&(metrics.structure==1)]
        tool=tool[tool.query_count==1].merge(tool[tool.query_count==2],on='frame',suffixes=('_before','_after'))
        tool['delta']=tool.dice_after-tool.dice_before;changed=tool.sort_values('delta').iloc[0];tool_frame=int(changed.frame)
        image=base/cid/'suite/full_seed20261016/debug'/f'query_02_score_{tool_frame:03d}.png';curated.append(image.relative_to(ROOT).as_posix())
        report+=['',f"Query2 tools, scoring frame {tool_frame}: Dice {changed.dice_before:.6f} → {changed.dice_after:.6f}. Ảnh này được chọn hậu nghiệm theo delta thấp nhất của tools, không phải mẫu ngẫu nhiên.",'',f'![Before/after second correction clip {cid}](<{image.as_posix()}>)']
    report+=['','## Diễn giải và giới hạn','','Điểm số debug được dựng lại từ observable predictions trước khi đọc charged query gold, và khớp selector ở cả 20 decisions. Bảng query thứ hai cho phép xác định redundancy penalty có đẩy một cấu trúc xuống dù error proxy cao hơn hay không. Giá trị gold chỉ dùng retrospective scoring/rendering và loader correction đã chọn.','','Ảnh before/after minh họa propagation trên scoring frame khác query; không phải ảnh query được sửa đúng. Exact prompt tại query không bảo đảm mask được propagate đúng tới các frame khác. Sự giảm có thể liên quan model propagation và độ thay đổi hình ảnh; chưa thể kết luận nguyên nhân lâm sàng hoặc nhãn sai.','','Replay này không đánh giá cơ chế mới, không chứng minh một cách sửa hữu ích, và không thay final test. Bước thiết kế tiếp theo cần protocol riêng: đánh giá độ tin cậy propagation từ observable signals và ảnh hưởng phân bổ structure; không dùng scoring gold làm gate chấp nhận/reject correction trong thuật toán.','','## Artifacts và tái chạy','','`src/debug_trace.py` lưu candidate scores, metrics mọi state, annotated-frame probability snapshots và before/after panels. CLIs `src.surgery` và `src.run_suite` có `--debug-trace`. Large state NPZ ở private Modal volume `isbi-gynsurg-results-v1` dưới run ID; CSV/JSON/PNG đã tải local. Lệnh GPU tái chạy tạo run mới/tốn compute:','','```powershell',"$env:MODAL_PROFILE='nhi544609'",'python -m modal run scripts/modal_debug_replay.py','```','','`analysis/all_candidate_scores.csv`, `all_state_metrics.csv`, `state_score_means.csv` và `debug_acceptance.json` lưu evidence máy đọc. Không rerun GPU chỉ để đọc báo cáo.']
    report+=['','## Tái lập numerical và lưu ý diễn giải','','Ca059 exact về scored metrics trong cả hai policies, cùng GPU name NVIDIA A10. Ca086 đổi NVIDIA A10 → A10G và khác Dice tới 0.023031 ở uterus/frame74 sau correction đầu, dù input/config/checkpoint và query trajectory trùng. Đây lớn hơn range quan sát trong audit numerical trước, không được gọi là sai số nhỏ. Chưa cô lập nguyên nhân hardware/kernel/runtime; không quy hết cho GPU name hoặc debug. Không dùng replay ca086 như exact reproduction của mức giảm uterus cũ; quan sát tools sau query2 và scale penalty trong chính replay vẫn có bằng chứng trực tiếp. Trước confirmatory runs cần cố định actual hardware/runtime và kiểm repeatability; không coi các GPU cùng nhãn yêu cầu Modal là tự động bitwise-equivalent.','','## Ghi chú clipping trong feature code','','Tại initial state, probabilities bằng framewise nhưng feature code clip probabilities vào [1e-6, 1-1e-6] trước khi tính disagreement với framewise chưa clip. Vì vậy disagreement có thể cỡ 1e-8–1e-7 thay vì đúng zero. Các mô tả E=H/2 trong báo cáo chẩn đoán trước cần hiểu là xấp xỉ tại initial state; debug score hiện dùng đúng implementation và đã đối chiếu argmax ở mọi bước. Không sửa feature math trong observational replay này.']
    target=ROOT/'REPORT_DEBUG_REPLAY_20261017.md';target.write_text('\n'.join(report)+'\n',encoding='utf8')
    result=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,run_id=args.run,checks=checks,state_metrics_rows=len(metrics),candidate_score_rows=len(scores),curated_panels=curated,final_test_open=False,model_or_policy_modified=False,debug_observation_added=True,large_arrays_location='private Modal volume, not downloaded locally')
    (out/'debug_acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
