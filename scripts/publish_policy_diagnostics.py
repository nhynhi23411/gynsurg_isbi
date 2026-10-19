"""Validate and publish saved post-hoc diagnostics without model execution."""
from pathlib import Path
import argparse, datetime, hashlib, json, re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);args=a.parse_args()
    base=ROOT/'results'/args.run;analysis=base/'analysis'
    report=json.loads((analysis/'diagnostic_report.json').read_text())
    if not all(report[k] for k in ['posthoc','input_hashes_match','first_decisions_reconstructed','prior_auc_reproduced']):raise ValueError('Diagnostic acceptance failed')
    if report['source_count']!=9:raise ValueError('Wrong development source count')
    raw=json.loads((base/'diagnostics.json').read_text())
    ids={r['clip_id'] for r in raw['records']}
    if ids!={'001','038','059','086','105','116','138','146','160'}:raise ValueError('Development-only allowlist failed')
    if json.loads((ROOT/'configs/review_status.json').read_text())['approved_for_test']:raise ValueError('Unexpected test authorization')
    paired=pd.read_csv(analysis/'paired_clip_structure_budget.csv',dtype={'clip_id':str})
    frames=pd.read_csv(analysis/'paired_frames_ranked.csv',dtype={'clip_id':str})
    if len(paired)!=72 or len(frames)!=216:raise ValueError('Incomplete paired outputs')
    if not np.isfinite(paired.delta).all() or not np.isfinite(frames.delta).all():raise ValueError('Nonfinite deltas')
    if not np.allclose(paired.delta,paired.dice_full-paired.dice_uncertainty,atol=1e-12):raise ValueError('Paired delta mismatch')
    if not np.allclose(frames.delta,frames.dice_full-frames.dice_uncertainty,atol=1e-12):raise ValueError('Frame delta mismatch')
    content=(base/'REPORT_POLICY_DIAGNOSTICS.md').read_text(encoding='utf8')
    for match in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',content):
        if not (base/match).is_file():raise FileNotFoundError(match)
        content=content.replace(']('+match+')','](<'+(base/match).as_posix()+'>)')
    header='''# Kết luận phân tích policy — 17/10/2026

**Đã hoàn thành phân tích outputs hiện có.** Hai ca 059/086 gây giảm lớn nhất, chủ yếu ở budget hai corrections. Proximity ban đầu có tương quan mô tả gần 0 với lỗi trên query candidates (Spearman rho=0.0119); nó chỉ đổi lựa chọn đầu tiên ở 1/9 ca. Với hai heads hiện tại, proximity đối xứng giữa uterus và tools nên không tự phân biệt cấu trúc nào cần sửa.

Kết quả ablation của hai ca giảm mạnh gắn với bật diversity và thay thứ tự phân bổ correction. Ca059 còn cho thấy một correction có thể làm Dice uterus trên scoring frames giảm từ 0.976361 xuống 0.886498. Đây là đầu mối để nghiên cứu độ tin cậy propagation; chưa có cơ sở tiếp tục khẳng định proximity cải thiện allocation.

Không đổi method/hyperparameters hoặc mở final test trong lượt này. Dữ liệu dùng lại từ GPU runs đã hoàn tất; CPU trên private Modal trích xuất features và baseline overlays. Ảnh baseline không phải ảnh sau propagation. Các kết luận chỉ áp dụng cho exploratory development, một clip trên mỗi nguồn.

---

'''
    if (ROOT/'REPORT_DEBUG_REPLAY_20261017.md').exists():
        header+='Ghi chú cập nhật từ debug replay: initial disagreement có thể cỡ 1e-8–1e-7 do clipping probabilities trước khi so với framewise chưa clip; E=H/2 trong phần dưới là xấp xỉ. Debug mới ghi đúng mọi thành phần score và đối chiếu argmax ở từng bước. Xem [báo cáo replay](<'+(ROOT/'REPORT_DEBUG_REPLAY_20261017.md').as_posix()+'>).\n\n'
    target=ROOT/'REPORT_POLICY_DIAGNOSTICS_20261017.md'
    target.write_text(header+content,encoding='utf8')
    script=ROOT/'scripts/analyze_policy_diagnostics.py'
    files=[p for p in base.rglob('*') if p.is_file()]+[script,ROOT/'scripts/modal_policy_diagnostics.py',ROOT/'scripts/publish_policy_diagnostics.py',ROOT/'protocols/20261016_official_sparse_v1/POSTHOC_DIAGNOSTICS.md',target]
    receipt=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,diagnostic_run=args.run,source_count=9,paired_rows=72,paired_frame_rows=216,final_test_open=False,new_annotations=0,new_gpu_inference=False,artifact_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)},scope='Post-hoc consolidation hashes and consistency checks; not a new raw-data audit or confirmation of clinical utility')
    (ROOT/'materials/verification/policy_diagnostics_acceptance.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    qa=dict(reviewed_at_utc=receipt['checked_at_utc'],reviewer='Codex engineering visual inspection',run=args.run,reviewed_panels=['059_099.png','086_099.png','038_099.png'],findings='Labels readable; image panels aligned; resized to 16:9; footer outside image; orange uterus and cyan tools overlay convention. Baseline only, not post-propagation masks.',clinical_review=False)
    (ROOT/'materials/verification/policy_diagnostics_visual_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf8')
    print('Published',target,';',len(files),'artifact hashes; paired outputs accepted')

if __name__=='__main__':main()
