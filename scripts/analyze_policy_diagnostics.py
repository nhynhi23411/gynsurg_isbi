"""Post-hoc development diagnosis from saved metrics and CPU-extracted predictions."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def correlation(x,y):
    return float(spearmanr(x,y).statistic) if len(x)>2 and len(set(x))>1 and len(set(y))>1 else None

def main():
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);args=a.parse_args()
    base=ROOT/'results'/args.run;raw=json.loads((base/'diagnostics.json').read_text(encoding='utf8'))
    source=ROOT/'results'/raw['source_run'];out=base/'analysis';out.mkdir(exist_ok=False)
    accepted=json.loads((source/'cohort_acceptance.json').read_text());assert accepted['complete']
    aggregate=[];frames=[];initial=[];inputs=[]
    for record in raw['records']:
        cid=record['clip_id'];suite=source/cid/'suite'
        provenance=json.loads((suite/'suite_provenance.json').read_text())
        # The extraction must be tied to the same immutable sequence as the GPU run.
        assert provenance['input_sha256']==record['input_sha256']
        inputs.append(dict(clip_id=cid,input_sha256=record['input_sha256']))
        for name in ['full','uncertainty','error','error_proximity','error_diversity']:
            folder=suite/f'{name}_seed20261016'
            aggregate.append(pd.read_csv(folder/'per_clip.csv',dtype={'clip_id':str}))
            frames.append(pd.read_csv(folder/'per_frame.csv',dtype={'clip_id':str}))
        q=[r for r in record['rows'] if r['pool']=='query']
        for item in record['initial']:
            # At zero queries, disagreement and redundancy vanish. E=H/2 for error policies.
            def score(r):
                return r['entropy'] if item['policy']=='uncertainty' else .5*r['entropy']*(1+r['proximity'] if item['policy'] in ['full','error_proximity'] else 1)
            expected=max(q,key=lambda r:(score(r),-r['frame'],-r['structure']))
            assert (expected['frame'],expected['structure'])==(item['frame'],item['structure'])
            initial.append(dict(clip_id=cid,**{k:v for k,v in item.items() if k!='recorded_queries'},trajectory=json.dumps(item['recorded_queries'])))
    assert len(inputs)==9 and len({x['clip_id'] for x in inputs})==9
    data=pd.concat(aggregate);frame=pd.concat(frames)
    assert len(data)==360 and len(frame)==1080
    key=['procedure_id','clip_id','density','structure']
    baseline=data[data.policy=='uncertainty'];full=data[data.policy=='full']
    paired=full.merge(baseline,on=key,suffixes=('_full','_uncertainty'),validate='one_to_one')
    assert len(paired)==72
    assert (paired.scoring_frame_sha256_full==paired.scoring_frame_sha256_uncertainty).all()
    paired['delta']=paired.dice_full-paired.dice_uncertainty
    paired.to_csv(out/'paired_clip_structure_budget.csv',index=False)
    fp=frame[frame.policy=='full'].merge(frame[frame.policy=='uncertainty'],on=key+['frame_index'],suffixes=('_full','_uncertainty'),validate='one_to_one')
    fp['delta']=fp.dice_full-fp.dice_uncertainty
    fp.sort_values('delta').to_csv(out/'paired_frames_ranked.csv',index=False)
    auc=[]
    for keys,block in data.groupby(['policy','procedure_id','clip_id','structure']):
        block=block.sort_values('actual_density');x=block.actual_density.to_numpy()
        auc.append(dict(zip(['policy','procedure_id','clip_id','structure'],keys),auc=float(trapezoid(block.dice,x)/(x[-1]-x[0]))))
    table=pd.DataFrame(auc).pivot(index=['procedure_id','clip_id','structure'],columns='policy',values='auc').reset_index()
    table['full_minus_uncertainty']=table['full']-table['uncertainty']
    table.to_csv(out/'structure_auc.csv',index=False)
    case=table.groupby(['procedure_id','clip_id']).mean(numeric_only=True).drop(columns='structure').sort_values('full_minus_uncertainty')
    case.to_csv(out/'case_auc_ranked.csv')
    prior=pd.read_csv(source/'analysis/per_procedure_all_policy_auc.csv')
    for policy in ['full','uncertainty','error','error_proximity','error_diversity']:
        expected=prior[prior.policy==policy].set_index('procedure_id').dice_auc.sort_index()
        actual=table.groupby('procedure_id')[policy].mean().sort_index()
        assert np.allclose(expected,actual,rtol=0,atol=1e-12)
    zero=data[data.density==0][['policy','procedure_id','clip_id','structure','dice']].rename(columns={'dice':'zero_dice'})
    harm=data[data.density>0].merge(zero,on=['policy','procedure_id','clip_id','structure'],validate='many_to_one')
    harm['change_from_zero']=harm.dice-harm.zero_dice
    harm.to_csv(out/'correction_change_from_zero.csv',index=False)
    contrasts=[];rng=np.random.default_rng(20261016)
    for lhs,rhs in [('full','uncertainty'),('error','uncertainty'),('error_proximity','error'),('error_diversity','error'),('full','error_diversity'),('full','error_proximity')]:
        values=(table[lhs]-table[rhs]).groupby(table.procedure_id).mean().to_numpy()
        ci=np.quantile(values[rng.integers(0,len(values),(2000,len(values)))].mean(axis=1),[.025,.975])
        contrasts.append(dict(contrast=lhs+' minus '+rhs,mean_delta=float(values.mean()),ci95=list(map(float,ci)),n=9,negative_sources=int((values<0).sum()),positive_sources=int((values>0).sum())))
    rows=pd.DataFrame([r for record in raw['records'] for r in record['rows']]);rows['error']=1-rows.dice
    rows.to_csv(out/'initial_snapshot_features.csv',index=False)
    initial=pd.DataFrame(initial);initial.to_csv(out/'initial_decisions_and_trajectories.csv',index=False)
    qrows=rows[rows.pool=='query']
    symmetric=bool((qrows.pivot(index=['clip_id','frame'],columns='structure',values='proximity').diff(axis=1)[1].abs()<1e-12).all())
    present_summary=[]
    for k,block in qrows[qrows.gold_present].groupby('structure'):
        present_summary.append(dict(structure=int(k),n=len(block),proximity_error_rho=correlation(block.proximity,block.error),entropy_error_rho=correlation(block.entropy,block.error)))
    corrs=[]
    for (pid,pool),block in rows.groupby(['procedure_id','pool']):
        corrs.append(dict(procedure_id=pid,pool=pool,n=len(block),proximity_error_rho=correlation(block.proximity,block.error),entropy_error_rho=correlation(block.entropy,block.error),proximity_unique=block.proximity.nunique()))
    pd.DataFrame(corrs).to_csv(out/'within_procedure_correlations.csv',index=False)
    summaries=[]
    for pool,block in rows.groupby('pool'):
        summaries.append(dict(pool=pool,n=len(block),proximity_zero=int((block.proximity==0).sum()),proximity_ge_0p99=int((block.proximity>=.99).sum()),median_proximity=float(block.proximity.median()),pooled_proximity_error_rho=correlation(block.proximity,block.error),pooled_entropy_error_rho=correlation(block.entropy,block.error)))
    decisions=initial.pivot(index='clip_id',columns='policy',values=['frame','structure'])
    changed=[]
    for cid in decisions.index:
        if (decisions.loc[cid,('frame','full')],decisions.loc[cid,('structure','full')])!=(decisions.loc[cid,('frame','uncertainty')],decisions.loc[cid,('structure','uncertainty')]):changed.append(cid)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for k,label in [(0,'uterus'),(1,'tools')]:
        b=rows[(rows.pool=='query')&(rows.structure==k)]
        axes[0].scatter(b.proximity,b.error,label=label,alpha=.7)
        b=paired[paired.structure==k].groupby('actual_density_full').delta.mean()
        axes[1].plot(b.index,b.values,marker='o',label=label)
    axes[0].set(xlabel='Predicted proximity P (initial query candidates)',ylabel='Baseline Dice error (1 - Dice)',title='Retrospective diagnosis; 9 development sources')
    axes[1].axhline(0,color='black',linewidth=.8);axes[1].set(xlabel='Corrections / 100 raw frames',ylabel='Mean paired Dice: full minus uncertainty',title='Same scoring frames and annotation budget')
    for ax in axes:ax.legend();ax.grid(alpha=.15)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'diagnostic_overview.{ext}',dpi=180)
    plt.close(fig)
    result=dict(scope=raw['scope'],posthoc=True,input_hashes_match=True,first_decisions_reconstructed=True,prior_auc_reproduced=True,source_count=9,contrasts=contrasts,feature_summaries=summaries,query_proximity_symmetric_between_structures=symmetric,present_only_query_correlations=present_summary,first_decision_changed_clips=changed,worst_sources=case.head(3).reset_index().to_dict('records'),limits=['Six annotated snapshots per clip; one development clip per source','Correlations descriptive, clustered/tied and not causal or confirmatory','No saved intermediate propagation states; later-step feature attribution unavailable','Baseline overlays are not propagated final predictions','Final test unopened; no new labels or parameter optimization'])
    (out/'diagnostic_report.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    def fmt(x):return 'undefined' if x is None else f'{x:.4f}'
    lines=['# Phân tích lỗi policy và proximity — GynSurg', '', '**Phân tích hậu nghiệm trên 9 nguồn development; không mở final test, không sửa policy hoặc chọn tham số mới.**', '', '## Kết quả theo thành phần', '', '| Contrast | Δ normalized Dice AUC | CI95% mô tả | Nguồn âm/dương |','|---|---:|---|---|']
    for c in contrasts:lines.append(f"| {c['contrast']} | {c['mean_delta']:.6f} | [{c['ci95'][0]:.6f}, {c['ci95'][1]:.6f}] | {c['negative_sources']}/{c['positive_sources']} |")
    lines+=['','Các contrasts ablation giữ cùng engine, dữ liệu và budget. Đây là so sánh policy trajectories, không chứng minh nguyên nhân lâm sàng. CIs hậu nghiệm chưa hiệu chỉnh multiple comparisons.','', '## Ca và cấu trúc gây giảm', '', '| Clip | Uterus AUC delta | Tools AUC delta | Macro delta |','|---|---:|---:|---:|']
    for (pid,cid),r in case.iterrows():
        b=table[table.clip_id==cid].set_index('structure');lines.append(f"| {cid} | {b.loc[0,'full_minus_uncertainty']:.6f} | {b.loc[1,'full_minus_uncertainty']:.6f} | {r.full_minus_uncertainty:.6f} |")
    lines+=['','## Proximity có dự báo lỗi không?', '', '| Pool | Số frame–structure | P=0 | P≥0.99 | Median P | P–error pooled rho | Entropy–error pooled rho |','|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['pool']} | {r['n']} | {r['proximity_zero']} | {r['proximity_ge_0p99']} | {r['median_proximity']:.4f} | {fmt(r['pooled_proximity_error_rho'])} | {fmt(r['pooled_entropy_error_rho'])} |")
    lines+=['',f"Proximity giữa hai structures bằng nhau ở mọi query frame: **{symmetric}**. Với đúng một organ head và một tools-union head, minimum distance có tính đối xứng: P chỉ reweight theo frame, không tự phân biệt nên sửa uterus hay tools ở cùng frame. Đây là giới hạn cấu trúc của feature hiện tại.",'','Present-only query correlations (vẫn descriptive, phụ thuộc theo nguồn):']
    for r in present_summary:lines.append(f"- Structure {r['structure']}: n={r['n']}, proximity/error rho={fmt(r['proximity_error_rho'])}, entropy/error rho={fmt(r['entropy_error_rho'])}.")
    lines+=['', 'Spearman tính trên initial predictions và lỗi so với nhãn official; query gold chỉ dùng retrospective evaluation. Pooled correlations trộn cấu trúc/nguồn và các quan sát phụ thuộc; không xem là statistical significance. CSV within_procedure_correlations giữ cả undefined khi proximity không biến thiên. Không suy cơ chế từ correlation đơn lẻ.', '',f"First decision full khác uncertainty ở {len(changed)}/9 clips: {', '.join(changed) or 'không có'}. Đã tái tạo và đối chiếu first decision của năm policies từ initial probabilities. Tại bước này disagreement=0 và redundancy=0, nên error chỉ bằng entropy/2; khác biệt full/uncertainty chỉ có thể đến proximity hoặc tie. Các bước sau không đủ saved probability states để phân rã score; query logs chỉ cho thứ tự lựa chọn.",'','## Hình và dữ liệu kiểm tra', '', '![Diagnostic overview](analysis/diagnostic_overview.png)', '', 'Ba nguồn có macro delta thấp nhất; hiển thị baseline ở scoring frame 99 để xem độ khó ban đầu. Màu cam=uterus, cyan=tools. Đây không phải mask sau propagation; không dùng hình baseline để khẳng định propagation thất bại.']
    for cid in case.head(3).reset_index().clip_id:lines += ['',f'### Clip {cid}', '',f'![Baseline và official reference clip {cid}](panels/{cid}_099.png)']
    lines+=['','## Cơ chế có bằng chứng cụ thể', '', 'Ở hai ca giảm mạnh nhất, full và error_diversity có cùng kết quả và cùng query trajectory; error/error_proximity/uncertainty cũng trùng nhau. Sự khác biệt trong các ca này gắn với việc bật diversity trong trajectory, không phải thêm proximity. Không suy điều này thành causal claim cho mọi ca.']
    for cid in ['059','086']:
        b=paired[(paired.clip_id==cid)&(paired.density==2)].set_index('structure')
        paths=initial[(initial.clip_id==cid)&(initial.policy.isin(['full','uncertainty']))]
        lines+=['',f"Clip {cid}, budget hai corrections: uterus delta={b.loc[0,'delta']:.6f}, tools delta={b.loc[1,'delta']:.6f}."]
        for _,r in paths.iterrows():lines.append(f"- {r.policy}: hai queries đầu {json.loads(r.trajectory)[:2]} (frame, structure; 0=uterus, 1=tools).")
    b=data[(data.clip_id=='059')&(data.policy=='full')&(data.structure==0)].set_index('density')
    lines+=['',f"Ca059 cho thấy correction không bảo đảm tăng Dice trên frame khác: uterus baseline={b.loc[0,'dice']:.6f}, sau một correction={b.loc[1,'dice']:.6f}. Budget hai của full giữ nguyên uterus trong khi uncertainty sửa thêm uterus; tools được lợi nhưng không bù phần giảm uterus. Đây là propagation transfer có hại trên scoring snapshots, không phải chứng cứ nhãn query sai.",'', '## Giới hạn và bước tiếp theo','','Không đủ bằng chứng để tuyên bố proximity là error predictor hữu ích từ sample này. Ưu tiên tiếp theo là phân tích độ tin cậy của propagation và phân bổ giữa structures; chỉ thiết kế cơ chế mới nếu có giả thuyết kiểm được. Không tự giảm beta/đổi sigma để tối ưu các scoring snapshots vừa xem. Nếu nghiên cứu tiếp, lập amendment riêng trước vòng development mới và kiểm trên tập validation chưa dùng để thiết kế; không dùng final test để tìm gain.','','Không có intermediate probability tensors cho các budget trước cuối, nên chưa thể tạo overlay propagation ở mọi budget từ artifacts hiện có. Lượt này không rerun SAM/GPU; xuất baseline overlays, query trajectories và per-frame paired metrics đúng bằng chứng đang có.','','Các CSV trong analysis gồm paired_frames_ranked, paired_clip_structure_budget, structure_auc, case_auc_ranked, correction_change_from_zero, initial_snapshot_features, initial_decisions_and_trajectories và within_procedure_correlations. Input NPZ hashes khớp suite provenance của cả chín nguồn; AUC tái tính khớp báo cáo cũ với sai số <1e-12.']
    (base/'REPORT_POLICY_DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
