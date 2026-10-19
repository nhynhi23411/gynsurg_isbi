"""Restricted oracle headroom and descriptive signals; never fit a gate."""
from pathlib import Path
import argparse,json,sys
import numpy as np,pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import rankdata,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import paired_bootstrap

def rank_auc(scores,positive):
    scores=np.asarray(scores,float);positive=np.asarray(positive,bool);n=int(positive.sum());m=len(positive)-n
    if not n or not m:return None
    return float((rankdata(scores)[positive].sum()-n*(n+1)/2)/(n*m))

def main():
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);args=a.parse_args();base=ROOT/'results'/args.run
    obs=pd.read_csv(base/'observations.csv',dtype={'clip_id':str});provenance=json.loads((base/'provenance.json').read_text())
    if not provenance['cpu_only'] or provenance['model_rerun'] or provenance['final_test_open']:raise ValueError('Wrong diagnostic scope')
    if len(obs)!=324 or obs.duplicated(['clip_id','query_count','frame','structure']).any():raise ValueError('Incomplete/duplicate diagnostic grid')
    if set(obs.clip_id)!={'001','038','059','086','105','116','138','146','160'}:raise ValueError('Wrong development cohort')
    if not np.allclose(obs.restricted_oracle_dice,np.maximum(obs.raw_dice,obs.baseline_dice),rtol=0,atol=1e-12):raise ValueError('Oracle formula mismatch')
    if not np.allclose(obs[obs.query_count==0].raw_dice,obs[obs.query_count==0].baseline_dice,rtol=0,atol=1e-12):raise ValueError('Budget zero differs')
    out=base/'analysis';out.mkdir(exist_ok=False);primary=obs[obs.query_count.isin([0,1,2,5])].copy();primary['actual_density']=primary.query_count*.8
    auc=[]
    for (pid,cid),block in primary.groupby(['procedure_id','clip_id']):
        curve=block.groupby('actual_density')[['baseline_dice','raw_dice','restricted_oracle_dice']].mean().sort_index();x=curve.index.to_numpy()
        auc.append(dict(procedure_id=pid,clip_id=cid,**{name+'_auc':float(trapezoid(curve[name],x)/(x[-1]-x[0])) for name in curve}))
    auc=pd.DataFrame(auc);auc['headroom']=auc.restricted_oracle_dice_auc-auc.raw_dice_auc;auc.to_csv(out/'per_source_oracle_auc.csv',index=False)
    # Exact trapezoidal decomposition: one clip/source, six frame-structure cells.
    primary['auc_weight']=primary.query_count.map({0:.1,1:.2,2:.4,5:.3})
    primary['headroom_contribution']=(primary.restricted_oracle_dice-primary.raw_dice)*primary.auc_weight/6
    contribution=primary.groupby(['procedure_id','clip_id','frame','structure','gold_present']).headroom_contribution.sum().reset_index().sort_values('headroom_contribution',ascending=False)
    contribution.to_csv(out/'headroom_by_frame_structure.csv',index=False)
    if not np.isclose(contribution.headroom_contribution.sum()/9,auc.headroom.mean(),atol=1e-12):raise ValueError('Headroom decomposition mismatch')
    old=pd.read_csv(ROOT/'results'/provenance['source_run']/'analysis/recipe_auc.csv',dtype={'clip_id':str});old=old[old.recipe=='uncertainty_raw'].set_index('clip_id').dice_auc
    if not np.allclose(auc.set_index('clip_id').raw_dice_auc.sort_index(),old.sort_index(),rtol=0,atol=1e-12):raise ValueError('Raw AUC not reproduced')
    eligible=primary[(primary.query_count>0)&(primary.structure_prompted.astype(str).str.lower()=='true')].copy()
    eligible['effect']=np.where(eligible.gain>.01,'helpful',np.where(eligible.gain<-.01,'harmful','neutral'))
    eligible.to_csv(out/'prompted_observations.csv',index=False)
    features=['sam_entropy','baseline_entropy','entropy_advantage','roi_disagreement','log_area_ratio','baseline_mask_iou','state_change_iou','sparse_temporal_iou','nearest_query_s']
    directions={'sam_entropy':-1,'entropy_advantage':1,'nearest_query_s':-1,'sparse_temporal_iou':1}
    association=[]
    def summarize(block,scope,label):
        selected=block[block.effect!='neutral'];positive=selected.effect=='helpful'
        for feature in features:
            rho=float(spearmanr(block[feature],block.gain).statistic) if block[feature].nunique()>1 and block.gain.nunique()>1 else None
            association.append(dict(scope=scope,label=label,feature=feature,n=len(block),n_helpful=int((block.effect=='helpful').sum()),n_harmful=int((block.effect=='harmful').sum()),spearman_gain=rho,score_direction=directions.get(feature),helpful_vs_harmful_auc=rank_auc(selected[feature]*directions[feature],positive) if feature in directions else None))
    summarize(eligible,'pooled','all')
    for pid,b in eligible.groupby('procedure_id'):summarize(b,'source',pid)
    for n,b in eligible.groupby('query_count'):summarize(b,'budget',str(n))
    assoc=pd.DataFrame(association);assoc.to_csv(out/'descriptive_signal_associations.csv',index=False)
    summary=paired_bootstrap(auc.restricted_oracle_dice_auc.to_numpy(),auc.raw_dice_auc.to_numpy())
    result=dict(posthoc=True,restricted_binary_choice_only=True,attainable_method=False,raw_auc_reproduced=True,mean_baseline_auc=float(auc.baseline_dice_auc.mean()),mean_raw_auc=float(auc.raw_dice_auc.mean()),mean_restricted_oracle_auc=float(auc.restricted_oracle_dice_auc.mean()),oracle_minus_raw_descriptive=summary,opportunities=len(eligible),helpful=int((eligible.effect=='helpful').sum()),harmful=int((eligible.effect=='harmful').sum()),neutral=int((eligible.effect=='neutral').sum()),model_rerun=False,gate_trained=False,final_test_open=False)
    total=float(auc.headroom.sum());largest=auc.loc[auc.headroom.idxmax()]
    empty=contribution[contribution.gold_present.astype(str).str.lower()=='false'].headroom_contribution.sum()
    result.update(largest_headroom_clip=largest.clip_id,largest_clip_share=float(largest.headroom/total) if total else None,mean_headroom_without_largest_clip=float(auc[auc.clip_id!=largest.clip_id].headroom.mean()),empty_reference_share=float(empty/total) if total else None)
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained');x=np.arange(len(auc))
    for j,(column,label) in enumerate([('baseline_dice_auc','Frozen baseline'),('raw_dice_auc','Raw uncertainty trajectory'),('restricted_oracle_dice_auc','Gold-assisted binary oracle')]):axes[0].bar(x+(j-1)*.25,auc[column],width=.24,label=label)
    axes[0].set_xticks(x,auc.clip_id);axes[0].set(xlabel='Previously observed source clip',ylabel='Normalized Dice AUC',ylim=(0,1.03));axes[0].legend(fontsize=8)
    for effect,b in eligible.groupby('effect'):axes[1].scatter(b.nearest_query_s,b.gain,label=effect,alpha=.5,s=18)
    axes[1].axhline(0,color='black',linewidth=.7);axes[1].set(xlabel='Nearest same-structure query (seconds)',ylabel='Raw SAM Dice − frozen baseline Dice',title='Dependent frame/structure/budget observations');axes[1].legend(fontsize=8)
    fig.suptitle('Post-hoc diagnosis only: oracle is not an executable policy')
    for ext in ['png','pdf','svg']:fig.savefig(out/f'oracle_headroom.{ext}',dpi=180)
    pooled=assoc[assoc.scope=='pooled'];lines=['# Oracle chẩn đoán và tín hiệu lợi/hại propagation','','**Gold-assisted oracle chỉ chọn tốt hơn giữa baseline và raw SAM trên trajectory đã chạy. Đây không phải thuật toán triển khai, trần cho mọi phương pháp, hay kết quả efficacy mới.** Không rerun model/GPU, train gate, chỉnh policy hoặc mở final test.','',f"Mean normalized Dice AUC: baseline={result['mean_baseline_auc']:.6f}; raw SAM={result['mean_raw_auc']:.6f}; restricted oracle={result['mean_restricted_oracle_auc']:.6f}. Oracle−raw={summary['delta']:.6f}, descriptive source-bootstrap CI95% [{summary['ci95'][0]:.6f}, {summary['ci95'][1]:.6f}], n=9 nguồn đã quan sát.",'',f"Có {len(eligible)} frame/structure/budget observations của cấu trúc đã query: helpful={result['helpful']}, harmful={result['harmful']}, neutral={result['neutral']}, ngưỡng ±0.01 Dice chỉ để chẩn đoán. Không coi các hàng lặp là samples độc lập.",'', '| Tín hiệu | Spearman với gain, pooled | AUROC helpful/harmful, hướng cố định |','|---|---:|---:|']
    for _,r in pooled.iterrows():lines.append(f"| {r.feature} | {r.spearman_gain:.4f} | {r.helpful_vs_harmful_auc:.4f} |" if pd.notna(r.helpful_vs_harmful_auc) else f"| {r.feature} | {r.spearman_gain:.4f} | Không định hướng/không ước tính |")
    lines+=['','## Headroom có phân bố rộng không?', '',f"Ca {largest.clip_id} chiếm {100*result['largest_clip_share']:.1f}% tổng headroom. Bỏ riêng ca lớn nhất, mean headroom của tám nguồn còn lại là {result['mean_headroom_without_largest_clip']:.6f}. Phần headroom từ reference rỗng chiếm {100*result['empty_reference_share']:.1f}%; giữ nguyên convention chính, không bỏ các frame đó để đổi kết luận. Oracle CI dương không chứng minh gate học được: max(raw, baseline) vốn luôn ≥ raw theo xây dựng.",'','## Tín hiệu đáng kiểm tra tiếp', '', 'Khoảng cách tới query cùng cấu trúc có pooled descriptive AUROC≈0.791. Cần xem tính nhất quán từng nguồn/budget thay vì coi đây là validation.']
    lines+=['','| Nguồn | Helpful/harmful | AUROC negative query distance |','|---|---|---:|']
    for _,r in assoc[(assoc.scope=='source')&(assoc.feature=='nearest_query_s')].iterrows():lines.append(f"| {r.label} | {r.n_helpful}/{r.n_harmful} | {r.helpful_vs_harmful_auc:.4f} |" if pd.notna(r.helpful_vs_harmful_auc) else f"| {r.label} | {r.n_helpful}/{r.n_harmful} | Undefined: một class |")
    lines+=['','Ở sáu nguồn có cả helpful/harmful, query-distance AUROC có thể ước tính; ba nguồn chỉ có helpful nên không đánh giá specificity. SAM entropy có pooled AUROC≈0.771 nhưng dấu không ổn định ở mọi nguồn (ca086 có AUROC0 trên rất ít observations). Sparse temporal IoU pooled≈0.577, chưa cho tín hiệu mạnh. Không chọn threshold hoặc train gate trong lượt này.']
    lines+=['','AUROC chỉ mô tả trên dữ liệu đã xem, neutral bị loại. Không chọn hướng dấu/threshold theo outcome. Các CSV giữ cả per-source và per-budget để xem confounding; một pooled AUROC tốt không chứng minh dự báo ngoài nguồn. Không có frame-level p-values. Temporal IoU dùng snapshots cách nhau khoảng 1s, không phải adjacent-frame SAM stability; prior snapshot có thể là charged mask, đã ghi cờ trong observations. Counts3/4 được lưu nhưng signal assessment/AUC ở counts0/1/2/5 của grid gốc.','',f'![Restricted oracle headroom](<{(out/"oracle_headroom.png").as_posix()}>)','','## Diễn giải giới hạn','','Khoảng cách oracle−raw là mức có thể thu lại nếu biết đúng lúc nên giữ baseline hay SAM trong đúng hai outputs này. Oracle được quyền xem scoring gold nên không được so như một deployable method, không dùng để chọn query, không thay endpoint thật. Probability blending hoặc query trajectories khác có thể vượt oracle hai lựa chọn này; không gọi nó là global upper bound.','','Có headroom chưa đồng nghĩa có observable signal đủ tốt. Các entropy/area/disagreement/stability là proxies, không có calibration hoặc out-of-source prediction test trong lượt này. Nếu muốn học gate, cần protocol group-based riêng và validation chưa tham gia thiết kế; không dùng final test để lựa chọn cơ chế.','','Artifacts: observations.csv; analysis/per_source_oracle_auc.csv; prompted_observations.csv; descriptive_signal_associations.csv; report.json. Input và 54 saved-state NPZ hashes được lưu trong provenance.json. Raw AUC tái tính khớp source run với tolerance1e-12.']
    (ROOT/'REPORT_ORACLE_DIAGNOSTICS_20261017.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2));print(pooled[['feature','spearman_gain','helpful_vs_harmful_auc']].to_string(index=False))

if __name__=='__main__':main()
