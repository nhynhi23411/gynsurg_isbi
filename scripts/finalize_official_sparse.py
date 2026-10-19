"""Refresh actual evidence across original and official-sparse protocol versions."""
from pathlib import Path
import ast,csv,datetime,hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
def put(p,t):p.write_text(t.strip()+'\n',encoding='utf8')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    approval=read(ROOT/'configs/review_status.json');assert approval['approved_for_training'] and not approval['approved_for_test']
    logs=(ROOT/'materials/verification/execution_unit_tests.txt').read_text(encoding='utf8');tests=int(re.search(r'Ran (\d+) tests',logs).group(1));assert tests>=9 and '\nOK' in logs
    sources=list((ROOT/'src').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))
    for p in sources:ast.parse(p.read_text(encoding='utf8'),filename=str(p))
    configs=list((ROOT/'configs').rglob('*.json'))
    for p in configs:read(p)
    identity_path=next((ROOT/f'data/manifests/source_identity_recovery_{v}/summary.json' for v in ['v5','v3','v2'] if (ROOT/f'data/manifests/source_identity_recovery_{v}/summary.json').exists()),None)
    identity=read(identity_path) if identity_path else None
    recovery=read(ROOT/'materials/verification/task1_recovery_status.json')
    recovery['mapping_verified']=bool(identity and identity['mapping_verified']);put(ROOT/'materials/verification/task1_recovery_status.json',json.dumps(recovery,indent=2))
    sparse_path=ROOT/'data/manifests/official_sparse_gate_status.json';sparse=read(sparse_path) if sparse_path.exists() else None
    executions=[read(p) for p in sorted((ROOT/'results').glob('official_sparse_*/cohort_execution.json'))]
    analysis=[read(p) for p in sorted((ROOT/'results').glob('official_sparse_*/analysis/development_report.json'))]
    tuning=[read(p) for p in sorted((ROOT/'results').glob('official_sparse_tuning_*/analysis/selection.json'))]
    validation=[read(p) for p in sorted((ROOT/'results').glob('official_sparse_validation_*/analysis/validation_report.json'))]
    report=dict(updated_at_utc=now,status='official_sparse_development_in_progress' if sparse else 'source_verification_in_progress',
       primary_correction_evaluation_complete=False,final_test_open=False,training_performed=False,
       synthetic_unit_tests_passed=tests,python_syntax_files=len(sources),config_json_files=len(configs),
       task1_recovery=recovery,source_identity_verification=identity,
       task3_transfer=read(ROOT/'data/manifests/task3_transfer_integrity.json'),task3_alignment=read(ROOT/'data/manifests/task3_frame_alignment.summary.json'),
       original_tail_protocol_gates=read(ROOT/'data/manifests/execution_gate_status.json'),official_sparse_protocol_gates=sparse,
       official_sparse_executions=executions,development_analyses=analysis,tuning_selections=tuning,validation_analyses=validation,
       temporal_sensitivity_analyses=[read(p) for p in sorted((ROOT/'results').glob('official_sparse_*/sensitivity_1s/analysis/development_report.json'))],
       repeatability_checks=[read(p) for p in sorted((ROOT/'results').glob('official_sparse_repeatability_*/repeatability_comparison.json'))],
       debug_replays=[read(p) for p in sorted((ROOT/'results').glob('debug_replay_*/analysis/debug_acceptance.json'))],
       reliability_prototype_runs=[read(p) for p in sorted((ROOT/'results').glob('reliability_scaled_a10_*/analysis/report.json'))],
       oracle_diagnostics=[read(p) for p in sorted((ROOT/'results').glob('oracle_diagnostics_*/analysis/report.json'))],
       grouped_gate_development=[read(p) for p in sorted((ROOT/'results').glob('group_gate_development_*/report.json'))],
       train_expansion_progress=[read(p) for p in sorted((ROOT/'results').glob('train_expansion_*/cohort_progress.json'))],
       train_expansion_results=[read(p) for p in sorted((ROOT/'results').glob('train_expansion_*/analysis/report.json'))],
       train_expansion_final_audits=[{k:v for k,v in read(p).items() if k!='artifact_sha256'} for p in sorted((ROOT/'results').glob('train_expansion_*/final_recovery_audit.json'))],
       full_development_decisions=[read(p) for p in sorted((ROOT/'decisions').glob('*_full_development_execution.json'))],
       train_expansion_recovery_preflights=[read(p) for p in sorted((ROOT/'results').glob('train_expansion_*/recovery_20261019/runtime_preflight.json'))],
       train_expansion_pilot_acceptance=[read(p) for p in sorted((ROOT/'results').glob('train_expansion_*/recovery_20261019/pilot_passed.json'))],
       full_clip_engineering_benchmarks=[read(p) for p in sorted((ROOT/'results').glob('full_clip_technical_*/full_clip_technical.json'))],
       paper_protocol_compilations=[read(p) for p in sorted((ROOT/'results').glob('paper_protocol_draft_*/compilation_report.json'))],
       modal_compute_profile='nhi544609',modal_data_source_profile='acc_v3',remaining_credit_balances='API does not expose remaining balance; no balance inferred')
    if analysis:report['status']='official_sparse_development_results_available_final_test_locked'
    if validation:report['status']='code_and_feasibility_validated_no_policy_advantage_demonstrated_test_locked'
    if report['reliability_prototype_runs']:report['status']='exploratory_reliability_prototype_results_available_final_test_locked'
    if report['train_expansion_progress']:report['status']='training_baseline_expansion_receipts_incomplete_final_test_locked'
    if report['train_expansion_results'] and report['train_expansion_results'][-1].get('complete'):report['status']='all_training_baseline_analysis_available_final_audit_pending_test_locked'
    if report['train_expansion_final_audits'] and report['train_expansion_final_audits'][-1].get('passed'):report['status']='all_official_training_baseline_audited_complete_final_test_locked'
    put(ROOT/'materials/verification/execution_report.json',json.dumps(report,ensure_ascii=False,indent=2))
    mapping='Đã xác minh và áp dụng đủ 300/300 clip → 16 source procedures (v5); counts/pools/durations khớp, visual QA 16 source sheets + 5 outlier sheets. Source timestamps là correspondence cục bộ, không khẳng định exact cut time.' if recovery['mapping_verified'] else f"Chưa áp dụng mapping; {identity['pixel_pass_clips'] if identity else 0}/300 candidates pass."
    proto_status='Chưa có GPU results của reliability prototype.' if not report['reliability_prototype_runs'] else 'Reliability prototype có '+str(len(report['reliability_prototype_runs']))+' cohort analyses thật; đọc REPORT_RELIABILITY_EXECUTION_20261017.md. Đây là redirection exploratory, tách khỏi full-policy v1; final test vẫn khóa.'
    expansion_status='Chưa có expansion receipt.'
    if report['train_expansion_progress']:
        progress=report['train_expansion_progress'][-1]
        expansion_status=f"{progress['completed']}/170 clip receipts; stage={progress.get('stage','unknown')}; cùng9 nguồn training, không tăng số ca độc lập."
    if report['train_expansion_results'] and report['train_expansion_results'][-1].get('complete'):
        expansion_status='Đủ170 clip /9 nguồn training;161 inference mới +9 raw results tái sử dụng. Full cohort acceptance và source-macro analysis đã có; không phải final test.'
    if report['train_expansion_final_audits'] and report['train_expansion_final_audits'][-1].get('passed'):
        expansion_status+=' Final recovery audit passed; primary/secondary/oracle source aggregation và provenance đã kiểm.'
    put(ROOT/'EXECUTION_STATUS.md',f'''# Thực hiện GynSurg với nhãn chính thức — chưa mở final test

Cập nhật UTC: {now}. Người dùng đã cho phép execution và Modal; amendment không gán nhãn mới ghi ở `protocols/20261016_official_sparse_v1/PROTOCOL.md`. Plan gốc và snapshot được giữ nguyên.

| Hạng mục | Trạng thái hiện hành |
|---|---|
| Task 1 | Đủ 21/21 video, 104.049.666.773 bytes, SHA256 theo mirror revision, index đủ; private Modal volume |
| Source mapping | {mapping} |
| Task 2/3 | 300 clips; 1.800 official image-mask pairs; tất cả frame alignment pass |
| Official-only protocol | 125 raw frames; query [0,24,49], score [74,99,124]; hai targets; counts [0,1,2,5] |
| References | 3.600 target masks official đủ cho 300 sequences theo amended scope; không có nhãn mới |
| Scope review | Engineering official-release QA có evidence hash; clinical review vẫn false; không có inter-reader study |
| Software | {tests} tests pass; {len(sources)} Python files parse được; {len(configs)} config JSON hợp lệ; per-frame outputs/cohort validation/secondary summaries đã bổ sung |
| Development | {len(executions)} cohort runs hoàn tất; {len(analysis)} development analyses; {len(tuning)} grid selections; {len(validation)} validation analyses; xem report máy đọc |
| Raw baseline expansion | {expansion_status} |
| Final test | Khóa; chưa có final held-out results hoặc submission-ready paper |

Original 250-frame tail protocol vẫn thiếu 2.400 masks; requirement này không còn là dependency của official-only 125-frame benchmark. Không dùng prediction làm gold. S8 không có matched-supervision adaptation nên được ghi unavailable trong comparator decision; không gọi heuristic khác là SASVi reproduction.

Các outputs development là exploratory, không thay independent final test. Official test có bốn procedures; mục tiêu 8–10 chưa đạt với official split. Cần validation/tuning record, precision/repeatability checks, freeze ledger, test plan, institutional ethics/disclosures và authorship trước nộp.

Default full policy không hơn uncertainty trên chín ca development. Grid tốt nhất tắt proximity/diversity; ba ca validation cũng chưa cho ưu thế. Chưa đạt proceed criterion cho interaction-aware benefit nên giữ final test khóa. Theo steering mới, ưu tiên code; thông tin tác giả và các mục nộp bài được để pending.

{proto_status}

Lượt mở rộng170 clip ngày17/10 đã lỗi module; recovery ngày19/10 dùng launcher độc lập, CPU preflight và pilot002 đã có acceptance nếu được liệt kê trong machine report. Xem REPORT_RECOVERY_20261019.md và REPORT_TRAIN_EXPANSION_20261017.md để biết kết quả/tiến độ hiện hành. Không dùng trạng thái launcher cũ để suy rằng job hiện đang chạy.

Đọc `REPORT_OFFICIAL_SPARSE_EXECUTION_20261016.md` nếu đã có để xem kết quả hiện hành. `AUDIT_REPORT_20261016.md` là snapshot trước remediation, có ghi cập nhật ở đầu. `materials/verification/execution_report.json` tổng hợp evidence mới nhất.
''')
    data=ROOT/'DATA_STATUS.md';text=data.read_text(encoding='utf8').split('## Phase thực thi ngày 16/10/2026')[0].rstrip()
    text+='\n## Phase thực thi ngày 16/10/2026\n\n'+mapping+'\n\nTask 1 đủ 21 video/104,05 GB trên private Modal; Task 3 đủ 1.800 pairs. Protocol official-only dùng 125 frames và sáu snapshots có sẵn, không cần 2.400 tail masks. Engineering scope evidence và scientific gates riêng ở `data/manifests/official_sparse_gate_status.json`; clinical review không được khai đã có. Original tail annotation kit giữ nguyên để mở rộng sau. Xem EXECUTION_STATUS.md và report execution mới.\n'
    put(data,text)
    files=[]
    for name in ['src','scripts','configs','tests','paper']:
        files.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in str(p))
    files.extend(p for p in (ROOT/'protocols').rglob('*.md'));files.extend(p for p in ROOT.glob('*.md'))
    files.extend(p for p in (ROOT/'clinical_annotation').rglob('*') if p.is_file() and p.suffix in ['.md','.html'])
    with (ROOT/'SOURCE_CHECKSUMS.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['path','bytes','sha256']);w.writerows((p.relative_to(ROOT).as_posix(),p.stat().st_size,hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in sorted(set(files)))
    print('Refreshed',len(set(files)),'source hashes;',tests,'tests; final test remains locked')
if __name__=='__main__':main()
