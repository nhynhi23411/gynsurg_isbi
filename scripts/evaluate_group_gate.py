"""Frozen, CPU-only nested source-group development evaluation."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from group_gate import candidates, keep_raw, macro_auc, select_from_training
PROTOCOL = ROOT/'protocols/20261017_group_gate_v1'
INPUT = ROOT/'results/oracle_diagnostics_20261017T075644Z'
OUT = ROOT/'results/group_gate_development_20261017_v1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    freeze = json.loads((PROTOCOL/'FREEZE.json').read_text())
    for path, expected in freeze['sha256'].items():
        if sha(ROOT/path) != expected:
            raise ValueError(f'Frozen artifact changed: {path}')
    review = json.loads((ROOT/'configs/review_status.json').read_text())
    if review.get('approved_for_test', False):
        raise ValueError('Final test must remain locked')
    provenance = json.loads((INPUT/'provenance.json').read_text())
    if provenance['model_rerun'] or provenance['final_test_open']:
        raise ValueError('Wrong input scope')
    obs = pd.read_csv(INPUT/'observations.csv', dtype={'clip_id':str})
    allowed = {'001','038','059','086','105','116','138','146','160'}
    if len(obs) != 324 or set(obs.clip_id) != allowed or obs.duplicated(['clip_id','query_count','frame','structure']).any():
        raise ValueError('Wrong or incomplete input cohort')
    obs = obs[obs.query_count.isin([0,1,2,5])].copy().reset_index(drop=True)
    if obs.groupby('procedure_id').clip_id.nunique().ne(1).any() or obs.groupby('clip_id').procedure_id.nunique().ne(1).any() or obs.procedure_id.nunique()!=9:
        raise ValueError('Expected nine distinct source groups')
    if not np.allclose(obs[obs.query_count==0].raw_dice,obs[obs.query_count==0].baseline_dice,rtol=0,atol=1e-12):
        raise ValueError('Baseline zero mismatch')
    prompted = obs.structure_prompted.astype(str).str.lower().eq('true').to_numpy()
    if not np.allclose(obs.loc[~prompted,'raw_dice'], obs.loc[~prompted,'baseline_dice'],rtol=0,atol=1e-12):
        raise ValueError('Unprompted output differs from frozen baseline')
    rules = candidates(); by_name = {c['name']:c for c in rules}
    scores = {}; masks = {}
    for rule in rules:
        keep = keep_raw(prompted,obs.nearest_query_s,obs.sam_entropy,rule)
        dice = np.where(keep,obs.raw_dice,obs.baseline_dice)
        masks[rule['name']] = keep
        scores[rule['name']] = {pid:macro_auc(b,dice[b.index]) for pid,b in obs.groupby('procedure_id')}
    scores = pd.DataFrame(scores)
    old = pd.read_csv(INPUT/'analysis/per_source_oracle_auc.csv').set_index('procedure_id')
    if not np.allclose(scores.raw.sort_index(),old.raw_dice_auc.sort_index(),rtol=0,atol=1e-12):
        raise ValueError('Raw endpoint not reproduced')
    decisions = []; folds = []; per_source = []
    for pid,b in obs.groupby('procedure_id'):
        selected, means, inner = select_from_training(scores,pid)
        if any(pid in f['training_sources'] or pid==f['validation_source'] for f in inner):
            raise ValueError('Outer source leaked into selection')
        keep = masks[selected][b.index]
        gated = np.where(keep,b.raw_dice,b.baseline_dice)
        record = b.copy();record['selected_rule']=selected;record['keep_raw']=keep;record['gated_dice']=gated
        decisions.append(record)
        per_source.append(dict(procedure_id=pid,clip_id=b.clip_id.iloc[0],selected_rule=selected,raw_auc=float(scores.loc[pid,'raw']),gated_auc=macro_auc(b,gated),fallback_rows=int((~keep).sum()),prompted_rows=int(prompted[b.index].sum())))
        folds.append(dict(outer_held_out=pid,selection_sources=[str(x) for x in scores.index if x!=pid],inner_folds=inner,inner_mean_candidate_auc=means,selected_rule=selected))
    per = pd.DataFrame(per_source);per['delta']=per.gated_auc-per.raw_auc
    rng = np.random.default_rng(20261017);delta=per.delta.to_numpy()
    ci = np.quantile(delta[rng.integers(0,9,size=(2000,9))].mean(axis=1),[.025,.975]).tolist()
    without = float(per[per.clip_id!='160'].delta.mean());positive=int((per.delta>1e-12).sum())
    passes = bool(ci[0]>0 and positive>=6 and without>0)
    means = scores.mean();proposal=next(name for name in scores.columns if means[name]>=means.max()-1e-12)
    summary = dict(exploratory=True,prior_design_exposure=True,independent_validation=False,group_count=9,mean_raw_auc=float(per.raw_auc.mean()),mean_oof_gate_auc=float(per.gated_auc.mean()),mean_delta=float(delta.mean()),descriptive_ci95=ci,positive_sources=positive,mean_delta_excluding_160=without,development_signal_criterion_passed=passes,all_development_proposal=by_name[proposal],proposal_independent_performance=None,selected_frequencies=per.selected_rule.value_counts().to_dict(),model_rerun=False,gpu_used=False,final_test_open=False,default_policy='uncertainty_raw',decision='eligible_for_separate_validation_protocol' if passes else 'do_not_promote_gate',input_sha256=sha(INPUT/'observations.csv'),freeze_sha256=sha(PROTOCOL/'FREEZE.json'))
    OUT.mkdir(exist_ok=False)
    scores.to_csv(OUT/'all_candidate_source_auc.csv',index_label='procedure_id')
    per.to_csv(OUT/'per_source_oof_auc.csv',index=False)
    pd.concat(decisions).to_csv(OUT/'oof_decisions.csv',index=False)
    (OUT/'folds.json').write_text(json.dumps(folds,indent=2),encoding='utf8')
    (OUT/'report.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    lines=['# Gate theo nhóm nguồn: kết quả development','','Đây là tái sử dụng chín nguồn đã xem khi thiết kế feature/grid. Outer leave-one-source-out giữ toàn bộ frame/structure/budget của một nguồn ngoài chọn ngưỡng; kết quả không phải validation độc lập. Chỉ CPU, không rerun SAM, không đổi query trajectory, không mở final test.','',f"Raw AUC **{summary['mean_raw_auc']:.6f}**; OOF gate **{summary['mean_oof_gate_auc']:.6f}**; delta **{summary['mean_delta']:+.6f}**, descriptive source-bootstrap CI95% [{ci[0]:+.6f}, {ci[1]:+.6f}].",'',f"Nguồn cải thiện: **{positive}/9**. Bỏ clip160 theo sensitivity định trước: mean delta **{without:+.6f}**.",'','| Clip | Gate chọn trong 8 nguồn khác | Raw AUC | OOF gate AUC | Delta | Fallback rows |','|---|---|---:|---:|---:|---:|']
    for _,r in per.iterrows():
        lines.append(f'| {r.clip_id} | {r.selected_rule} | {r.raw_auc:.6f} | {r.gated_auc:.6f} | {r.delta:+.6f} | {r.fallback_rows} |')
    lines += ['',f"Tiêu chí development định trước: CI dưới >0, ≥6/9 nguồn tăng, và delta bỏ160 >0: **{'đạt' if passes else 'không đạt'}**. Giữ default uncertainty_raw; chưa tích hợp gate vào pipeline chính.",'',f"Quy tắc đề xuất nếu dùng tất cả development để chọn: `{proposal}`. Đây là lựa chọn để xem xét, không có performance độc lập. Không dùng điểm all-development làm kết quả efficacy.",'','## Cách đánh giá','','26 candidates cố định: raw; distance-only ở [0.5,1,2,3,5] giây; distance+SAM entropy với h=[0.05,0.1,0.2,0.4]. Chọn mean normalized macro Dice AUC trên tám inner source holdouts; không fit coefficients, nên inner fitting split chỉ ghi provenance/exclusion. Tie ưu tiên raw rồi thứ tự cố định. Dice và gold chỉ vào evaluator, không vào API gate. Giữ nguyên count0, structures chưa query, scoring [74,99,124], budgets [0,1,2,5] và empty-reference convention.','', '## Giới hạn và bước tiếp theo','','Các ngưỡng được chọn bằng development labels; outer outcome không tham gia chọn ngưỡng fold của chính nó. Tuy nhiên feature/grid đã được gợi ý từ phân tích cả chín nguồn trước đây, nên OOF vẫn có design exposure. CI chỉ mô tả, không điều chỉnh uncertainty của quá trình chọn phương pháp và không chứng minh lợi ích ngoài nguồn. Chín clips đại diện chín nguồn không thay toàn bộ training cohort. Không suy annotation time hay lợi ích lâm sàng.','', 'Không mở final test, không tiếp tục thay grid để tìm gain. Nếu tiêu chí không đạt, báo kết quả âm và giữ raw. Nếu đạt, cần protocol validation riêng, audit exposure và freeze candidate trước một lượt validation; ba nguồn validation cũ đã được dùng cho policy khác nên không được gọi pristine overall.','',f'Artifacts: `{OUT.relative_to(ROOT).as_posix()}` gồm OOF decisions, per-source AUC, đủ 26 candidates ×9 sources, inner/outer memberships và report.json. Protocol và code/input hashes ở `protocols/20261017_group_gate_v1/FREEZE.json`.']
    (ROOT/'REPORT_GROUP_GATE_20261017.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
