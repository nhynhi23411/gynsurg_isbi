"""Refresh this project's execution evidence without resetting the other project."""
from pathlib import Path
import ast,csv,datetime,hashlib,json,shutil,re
ROOT=Path(__file__).resolve().parents[1]
def put(p,t):p.write_text(t.strip()+'\n',encoding='utf8')
def main():
    if (ROOT/'data/manifests/official_sparse_gate_status.json').exists():raise RuntimeError('Legacy finalizer: use scripts/finalize_official_sparse.py for the amended protocol')
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    technical=[]
    for p in sorted((ROOT/'results').glob('technical_smoke_*/technical_smoke.json')):
        r=json.loads(p.read_text());r['peak_gpu_memory_scope']='SAM2 parent process only; exporter subprocess excluded';put(p,json.dumps(r,indent=2));technical.append((p,r))
    previews=[(p,json.loads(p.read_text())) for p in sorted((ROOT/'results').glob('development_frame_smoke_*/development_frame_smoke.json'))]
    assert technical and previews
    matched=[(p,r) for p,r in technical if any('selected-union' in x for x in r['joint_export']['structure_names'])][-1]
    preview=previews[-1];transfer=json.loads((ROOT/'data/manifests/task3_transfer_integrity.json').read_text());alignment=json.loads((ROOT/'data/manifests/task3_frame_alignment.summary.json').read_text());gates=json.loads((ROOT/'data/manifests/execution_gate_status.json').read_text());annotations=json.loads((ROOT/'clinical_annotation/tail_reference_v1/preparation.json').read_text())
    logs=(ROOT/'materials/verification/execution_unit_tests.txt').read_text(encoding='utf8');test_count=int(re.search(r'Ran (\d+) tests',logs).group(1));assert test_count>=9 and 'OK' in logs
    nas=json.loads((ROOT/'materials/verification/provider_remote_access_check.json').read_text());approval=json.loads((ROOT/'configs/review_status.json').read_text());assert approval['approved_for_training'] is True and approval['approved_for_test'] is False
    py=list((ROOT/'src').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))
    for p in py:ast.parse(p.read_text(encoding='utf8'),filename=str(p))
    json_count=0
    for p in (ROOT/'configs').rglob('*.json'):json.loads(p.read_text(encoding='utf8'));json_count+=1
    recovery_path=ROOT/'materials/verification/task1_recovery_status.json'
    recovery=json.loads(recovery_path.read_text()) if recovery_path.exists() else None
    identity_path=ROOT/'data/manifests/source_identity_recovery_v2/summary.json'
    identity=json.loads(identity_path.read_text()) if identity_path.exists() else None
    clip_benchmarks=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('full_clip_technical_*/full_clip_technical.json'))]
    report=dict(updated_at_utc=now,status=('Task1_mirror_recovery_in_progress_clinical_labels_pending' if recovery else 'awaiting_working_Task1_source_or_verified_mapping_and_clinical_labels'),task1_recovery=recovery,primary_correction_evaluation_complete=False,final_test_open=False,training_performed=False,clinical_preview_unique_frames=3,synthetic_unit_tests_passed=test_count,python_syntax_files=len(py),config_json_files=json_count,task3_transfer=transfer,task3_alignment=alignment,annotation_package=annotations,gates=gates,matched_gpu_smoke_path=str(matched[0].relative_to(ROOT)),latest_development_preview_path=str(preview[0].parent.relative_to(ROOT)/'development_preview.png'),modal_compute_profile='nhi544609',modal_data_source_profile='acc_v3',NAS_access_results=nas,remaining_credit_balances='not exposed by billing summary API; no balance invented')
    report['full_clip_engineering_benchmarks']=clip_benchmarks
    report['source_identity_verification']=identity
    if recovery and recovery['all_task1_videos_downloaded']:
        report['status']='Task1_download_complete_identity_verification_and_clinical_labels_pending'
    report['paper_protocol_compilations']=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('paper_protocol_draft_*/compilation_report.json'))]
    put(ROOT/'materials/verification/execution_report.json',json.dumps(report,ensure_ascii=False,indent=2))
    put(ROOT/'EXECUTION_STATUS.md',f'''# Thực hiện ISBI_GynSurg — chưa hoàn tất nghiên cứu

Cập nhật UTC: {now}. Người dùng đã cho phép thực hiện toàn bộ project và dùng các account Modal. Authorization và snapshot plan gốc nằm trong `protocols/20261016_execution_v1/`. Không cần xin lại quyền chạy; gate dữ liệu vẫn độc lập với authorization.

## Đã thực hiện thật

| Hạng mục | Kết quả và bằng chứng |
|---|---|
| Modal | Kiểm billing/inventory của 6 profile đã cấu hình. `huongntt-c2kc-ph` bị paused; không unpause. Đã dùng A10G tại `nhi544609`, dữ liệu cũ tại `acc_v3`. Billing summary không cho remaining credit balance. |
| Task 3 | Khôi phục đủ 1.800 ảnh + 1.800 mask, 590.595.868 bytes; hash local khớp remote toàn bộ. Mask RGB có ba kênh giống nhau, palette 0..180 bước 20. |
| Frame alignment | 1.800/1.800 ảnh khớp frame video ở kiểm tra pixel; normalized RGB MAE tối đa {alignment['maximum_expected_rgb_mae']:.6f}. Không có frame bị flag review theo threshold đã định. |
| Reference preparation | Có 3.600 binary masks dẫn xuất cho uterus/annotated-tools union; source scope review còn pending. Raw không sửa. |
| Checkpoint | Manifest 5.955 training rows khớp SHA256; không AutoLaparo row và không ảnh trùng SHA256 với Task 3. Lịch sử AutoLaparo benchmark vẫn phải disclose. |
| GPU | SAM2 hai chiều và real-checkpoint TorchScript export pass trên A10G. Bản matched gộp GynSurg classes 1/5/6; clinical scope review chưa hoàn tất. Optional SAM2 postprocessing tắt tường minh ở mọi config. |
| Preview | Đã inference trên 3 frame thuộc official train pool, có ảnh preview. Chưa tính clinical accuracy hoặc policy comparison. |
| Full-clip technical benchmark | Đã pass 250 raw frames trên A10G, working 720×405, prompts là predicted masks (không gold). Decode/framewise/save ~17.09s; hai replay ~42.76/85.37s; peak Torch allocation 945.860.608 bytes. Không phải total nvidia-smi VRAM hoặc correction-policy result. |
| Paper protocol | Có draft method/data/evaluation cụ thể, compile/render thành 3 trang. Results pending; authorship và compliance của study chưa hoàn tất. |
| Software | {test_count} synthetic unit tests pass; {len(py)} file Python parse được; {json_count} config JSON hợp lệ. Có environment lock local/GPU, source/config/checkpoint hashes. |
| Clinical annotation kit | 1.200 ảnh native 1920×1080 và 2.400 task trống, có source-video/image hashes. Không có pseudo-gold hoặc nhãn bác sĩ mới. |

## Điểm chưa thể vượt qua

1. Chưa có verified clip→procedure mapping. NAS chính thức timeout; mirror đã tải/kiểm SHA256 {recovery['downloaded_sha256_verified_videos'] if recovery else 0}/21 video (104.049.666.773 bytes), lập chỉ mục {recovery['indexed_videos'] if recovery else 0}/21 trên Modal. Refine đã hoàn tất; {identity['pixel_pass_clips'] if identity else 0}/300 clip pass kiểm tra pixel/pool; còn 18 clip cần giải quyết, 2 clip lệch official pool và counts chưa khớp. Xem `data/manifests/source_identity_recovery_v2/summary.json`. Chưa áp dụng mapping; không gán procedure theo counts hoặc candidate gần nhất.
2. Chưa có 2.400 tail-mask annotations và clinical scope/completeness review. Không tự gán absent, dùng prediction làm gold hoặc lấy clip ID thay procedure ID.

Supplementary chỉ cho biết 16 video đóng góp clip, với 9/3/4 video ở ba pool; riêng official test có 4 source procedures. Counts không tự xác minh từng clip. Mục tiêu 8–10 held-out procedures trong plan chưa đạt với riêng pool test này.

Primary AUC, procedure-cluster CI, ablation results, final test và manuscript Results **chưa có**. `check_execution_gates.py --require-ready` hiện trả exit 2 đúng thiết kế. Không đổi endpoint hoặc bỏ gate để tạo kết quả.

## File để tiếp tục

- [Preview development]({report['latest_development_preview_path']}) — ba frame, không phải accuracy estimate.
- [Báo cáo máy đọc](materials/verification/execution_report.json).
- [Đầu vào còn thiếu](NEEDS_INPUT.md) và [draft vận hành](protocols/20261016_execution_v1/OPERATIONAL_DRAFT.md).
- [Bộ gán nhãn](clinical_annotation/tail_reference_v1/README_VI.md), `annotation_tasks.csv` và `image_provenance.csv`.
- `python scripts/annotation_workspace.py` mở công cụ gán nhãn trên `http://127.0.0.1:8768`; lưu working versions, không tự duyệt reference. Bộ kiểm thử hiện có {test_count} tests pass.
- `data/manifests/procedure_mapping_tasks_v1.csv`; `data/manifests/sequence_review_v1/` có 300 bộ draft.
- `materials/verification/provider_access_check.json` và `provider_remote_access_check.json` ghi network failure thật. Draft liên hệ nhà cung cấp chưa gửi.

Tiếp theo: giải quyết identity failures và xác minh mapping từ video đã tải; hoàn tất clinician labels/scope và backlog code trong AUDIT_REPORT_20261016.md; validate input; chạy reviewed clinical pilot; development/ablation; freeze protocol rồi mới final test. Quyền dùng nghiên cứu phi thương mại đã đối chiếu nguồn; không tự tạo ethics waiver hoặc clinician sign-off.
''')
    p=ROOT/'DATA_STATUS.md';t=p.read_text(encoding='utf8').split('## Phase thực thi ngày 16/10/2026')[0];t+='''\n## Phase thực thi ngày 16/10/2026

Đã bổ sung đầy đủ AutoLaparo Task 3 từ volume Modal: 3.600 files, 590.595.868 bytes; hash local/remote khớp. Cả 1.800 image–mask pairs có 1920×1080, RGB mask ba kênh giống nhau. Frame IDs một-based 1/25/50/75/100/125 đã đối chiếu với video; tất cả pass kiểm tra pixel. Có 3.600 binary masks dẫn xuất ngoài raw, cần clinical scope review.

Đã chuẩn bị 1.200 tail-frame images và 2.400 blank annotation tasks. Đã tải đủ 21/21 video Task 1 từ mirror, SHA256 khớp revision đã pin, 104.049.666.773 bytes; lưu private Modal volume thay vì ổ D. Lập chỉ mục đủ 21 video và refine candidates; 282/300 clip pass pixel/pool, còn 18 chưa đạt và counts chưa khớp. Mapping chưa được áp dụng. Đã kiểm kê lại NoiSoi nhưng chưa tìm thấy bộ tail masks phù hợp. Trạng thái và toàn bộ thiếu sót hiện hành ở AUDIT_REPORT_20261016.md; các ghi chú preparation trước đó được giữ để truy vết.
''';put(p,t)
    source=ROOT/'SOURCE_CHECKSUMS.csv';archive=ROOT/'protocols/20261016_execution_v1/SOURCE_CHECKSUMS_PREPARATION.csv'
    if source.exists() and not archive.exists():shutil.copy2(source,archive)
    files=[]
    for name in ['src','scripts','configs','tests','paper']:
        files.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in str(p))
    files.extend(ROOT/name for name in ['PLAN_RULES.md','RUNBOOK.md','AGENTS.md','README.md','EXECUTION_STATUS.md','NEEDS_INPUT.md'])
    files.extend(p for p in [ROOT/'AUDIT_REPORT_20261016.md'] if p.exists())
    files.extend(p for p in (ROOT/'clinical_annotation').rglob('*') if p.is_file() and p.suffix in ['.md','.html'])
    files.extend(p for p in (ROOT/'protocols/20261016_execution_v1').glob('*.md'))
    with source.open('w',newline='',encoding='utf8') as f:w=csv.writer(f);w.writerow(['path','bytes','sha256']);w.writerows((str(p.relative_to(ROOT)),p.stat().st_size,hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in sorted(files))
    print('Execution evidence refreshed:',len(files),'source files; primary evaluation gate remains closed')
if __name__=='__main__':main()
