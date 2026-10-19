"""Consolidate verified execution, negative findings and remaining scientific scope."""
from pathlib import Path
import json,re,datetime,sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import paired_bootstrap
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('C:/Users/pc/ISBI_GynSurg_runs/train_expansion_20261017_v1')

def read(path):return json.loads(path.read_text(encoding='utf8'))
def ci(delta,interval):return f'{delta:+.6f} [{interval[0]:+.6f}, {interval[1]:+.6f}]'
def link(path,label):return f'[{label}](<{path.as_posix()}>)'

def main():
    full=read(BASE/'analysis/report.json');audit=read(BASE/'final_recovery_audit.json')
    if not audit['passed'] or not full['complete'] or full['clips']!=170:raise ValueError('Complete audited expansion required')
    v1=read(ROOT/'results/official_sparse_feasibility_20261016T144620Z/analysis/development_report.json')['primary_development']
    val=read(ROOT/'results/official_sparse_validation_20261016T153640Z/analysis/validation_report.json')['comparisons']
    reliability=read(ROOT/'results/reliability_scaled_a10_development_20261017T031408Z/analysis/report.json')['contrasts'][0]
    gate=read(ROOT/'results/group_gate_development_20261017_v1/report.json')
    oracle=read(ROOT/'results/oracle_diagnostics_20261017T075644Z/analysis/report.json')
    logs=(ROOT/'materials/verification/execution_unit_tests.txt').read_text(encoding='utf8')
    tests=int(re.search(r'Ran (\d+) tests',logs).group(1))
    if '\nOK' not in logs or tests<52:raise ValueError('Software checks missing')
    if read(ROOT/'configs/review_status.json')['approved_for_test']:raise ValueError('Test must remain locked')
    primary=full['raw_minus_budget_zero'];sens=full['exclude_prior_representatives_sensitivity']
    present=read(BASE/'analysis/present_only/report.json');pc=present['raw_minus_baseline']
    temporal=read(BASE/'analysis/temporal_1s/report.json');tc=temporal['raw_minus_baseline']
    oracle_full=read(BASE/'analysis/restricted_oracle/report.json');oc=oracle_full['oracle_minus_raw_descriptive']
    receipts=read(BASE/'cohort_execution.json')['reports']
    new_seconds=[r['total_seconds'] for r in receipts if not r['reused']]
    runtime_total=sum(new_seconds);runtime_median=float(pd.Series(new_seconds).median())
    source_table=pd.read_csv(BASE/'analysis/per_source_auc.csv')
    iou=paired_bootstrap(source_table.raw_iou_auc.to_numpy(),source_table.baseline_iou_auc.to_numpy(),seed=20261017)
    lines=['# Báo cáo tổng hợp thực thi ISBI GynSurg — 19/10/2026','',
        '**Đã hoàn tất phần mở rộng baseline và nghiệm thu170/170 clip training trong scope official-only125 frames.** Tổng161 clip inference mới và9 raw results A10 đã kiểm chứng tái sử dụng; vẫn là9 nguồn đã quan sát, không phải170 ca độc lập. Không đồng nghĩa đã hoàn tất original dense-label protocol hoặc bài sẵn sàng nộp.','',
        '## Kết quả mới trên toàn bộ training','',
        f"Source-macro normalized Dice AUC: raw uncertainty/SAM **{full['mean_raw_dice_auc']:.6f}**; frozen budget-zero baseline **{full['mean_baseline_dice_auc']:.6f}**. Contrast raw−no-correction **{ci(primary['delta'],primary['ci95'])}**, descriptive source-bootstrap95%, n=9.",'',
        f"Secondary IoU AUC: raw **{source_table.raw_iou_auc.mean():.6f}**, baseline **{source_table.baseline_iou_auc.mean():.6f}**; delta **{ci(iou['delta'],iou['ci95'])}**, cùng source-level aggregation/resampling. Các CI là exploratory, không điều chỉnh multiplicity.",'',
        f"Sensitivity bỏ9 clip đại diện cũ, còn161 clip trong cùng9 nguồn: **{ci(sens['delta'],sens['ci95'])}**. Đây là sensitivity trong nguồn đã biết, không phải out-of-source validation.",'',
        f"Sensitivity reference-present only: delta **{ci(pc['delta'],pc['ci95'])}**, {present['clips_with_present_reference']} clips/{present['source_groups']} sources, {present['present_frame_structure_cells']}/{present['total_unique_frame_structure_cells']} frame-structure cells. Số clip không có bất kỳ target hiện diện: {len(present['clips_without_any_present_reference'])}; coverage CSV giữ đầy đủ IDs. Mean present frames trong head→heads trong clip→clips trong nguồn; không thay primary empty-reference convention hoặc chọn lại queries.",'',
        f"Sensitivity loại±1s quanh union của mọi query đã charge: common scoring pool **[99,124]** cho baseline và mọi budget, coverage **2/3**. Delta **{ci(tc['delta'],tc['ci95'])}**; {temporal['scored_rows']} frame rows sau lọc. Không rerun model hoặc dùng scoring set khác nhau cho hai phía so sánh.",'',
        f"Restricted binary oracle trên170 clips: raw AUC **{oracle_full['mean_raw_auc']:.6f}**, oracle **{oracle_full['mean_restricted_oracle_auc']:.6f}**, headroom **{ci(oc['delta'],oc['ci95'])}**. Clip{oracle_full['largest_contribution_clip']} đóng góp {100*oracle_full['largest_clip_share']:.1f}% headroom sau weighting clip→source. Gold chọn giữa baseline/raw ở từng frame/head rồi mới macro; max bảo đảm headroom không âm, không chứng minh gate khả học hoặc hiệu quả phương pháp mới.",'',
        'Contrast này đo charged corrections so với không correction, không phải lợi ích scheduling mới. Mỗi nguồn ngang trọng số sau khi trung bình clip trong nguồn; giữ nguyên empty-reference convention và tất cả failure cases.','',
        f"![Full training baseline]({(BASE/'analysis/train_expansion_summary.png').as_posix()})",'',
        '## Toàn bộ kết quả phương pháp đã thử','',
        '| So sánh | Nguồn | Delta normalized Dice AUC [CI95% mô tả] | Diễn giải |',
        '|---|---:|---|---|',
        f"| Full policy v1 − uncertainty |9|{ci(v1['delta'],v1['ci95'])}|Chưa cho thấy ưu thế|",
        f"| Best-development policy trên validation cũ − uncertainty |3|{ci(val[0]['delta'],val[0]['ci95'])}|Không có ưu thế; chỉ một clip/nguồn|",
        f"| Interaction-nonzero trên validation cũ − uncertainty |3|{ci(val[1]['delta'],val[1]['ci95'])}|Không có ưu thế|",
        f"| Scaled diversity + reliability fallback − raw uncertainty |9|{ci(reliability['mean_delta'],reliability['exploratory_bootstrap']['ci95'])}|Fallback làm giảm quality trung bình|",
        f"| Grouped output gate OOF − raw |9|{ci(gate['mean_delta'],gate['descriptive_ci95'])}|Không đạt tiêu chí development; feature/grid đã xem các nguồn này|",'',
        f"Restricted binary oracle, dùng gold để chọn tốt hơn giữa hai outputs đã có, cho raw AUC {oracle['mean_raw_auc']:.6f} và oracle {oracle['mean_restricted_oracle_auc']:.6f}. Đây không phải thuật toán triển khai hoặc global upper bound; {100*oracle['largest_clip_share']:.1f}% headroom tập trung ở clip{oracle['largest_headroom_clip']}. Không dùng oracle để chọn query hoặc báo efficacy.",'',
        'Không sửa grid/gate theo kết quả bất lợi, không chọn seed tốt nhất, không bỏ ca âm và không mở final test để tìm gain. Giữ uncertainty_raw làm baseline; chưa chứng minh interaction-aware scheduling hoặc gate mới vượt uncertainty.','',
        '## Dữ liệu và phạm vi','',
        '| Hạng mục | Bằng chứng/trạng thái |','|---|---|',
        '| GynSurg releases |4.992 instrument +725 anatomy +1.579 auxiliary pairs; archive CRC/SHA và palette checks đã có; head annotation scope giữ riêng|',
        '| AutoLaparo Task2 |Đủ300 clips,250 frames/clip ở25fps; raw local giữ nguyên|',
        '| AutoLaparo Task3 |Đủ1.800 official image-mask pairs; alignment cả1.800 pairs đã pass;3.600 binary target masks|',
        '| Source mapping |300/300 clips→16 source videos verified v5; pixel/temporal/count/pool evidence và21 QA sheets|',
        '| Task1 |21 mirror videos /104.049.666.773 bytes trên private Modal; mirror hashes đã kiểm, không gọi là provider-original hashes; metadata20-vs21 discrepancy đã ghi|',
        '| Official split |170 train/57 validation/73 test, grouped9/3/4 nguồn; lượt mới chỉ170 training|',
        '| Current computational scope |125 raw frames, query[0,24,49], score[74,99,124], two dataset-defined targets, counts[0,1,2,5], actual densities[0,.8,1.6,4]|',
        '| Original dense scope |250-frame tail protocol vẫn thiếu2.400 reference masks; không thay bằng pseudo-labels|',
        '| Human review |Chưa có clinical adjudication/inter-reader study hoặc đo thời gian annotation thật|','',
        '## Code, kiểm thử và recovery đã nghiệm thu','',
        f"**{tests} unit tests đạt**, full-cohort acceptance170 case-runs;4.080 frame rows và1.360 aggregate rows. Recovery kiểm immutable scientific freeze, checkpoint/config/source/input evidence, actual NVIDIA A10, đủ9 reused+161new receipts; source-bootstrap và macro aggregation đã tái tạo. Final audit: **passed**.",'',
        'Lỗi triển khai cũ đã xử lý bằng launcher độc lập, serialized controller Python3.11 khớp image3.11, single-use GPU workers, individual spawn/get và call-ID ledger. CPU import preflight và pilot002 end-to-end pass trước batch. Windows Store child-user-site correction chỉ áp dụng cho original analysis interpreter; không đổi GPU model hoặc source algorithm. Các log/freeze thất bại cũ giữ nguyên. Các lần cấp sai tên GPU bị chặn trước inference; retry chỉ vì allocation, không vì Dice thấp.','',
        'Sau khi đủ170 accepted receipts và GPU app đã dừng, một job CPU chỉ đọc đã đóng gói2.317 CSV/JSON/TXT để thay download tuần tự chậm. Archive và từng member đều kiểm SHA256 trước local handoff; log/ledger cũ giữ lại. Finisher copy artifacts đã xác minh và chạy analyzer gốc còn nguyên freeze trong original analysis environment. Transport/finisher có addendum và source freeze riêng; không có model call hoặc metric rewrite trong bước tải. Final audit đã kiểm archive, member hashes và freeze chain này.','',
        'Raw/NoiSoi bất biến. Frames/NPZ lớn nằm private Modal results volume; CSV/JSON/logs local ở C:, junction từ results trong project. Không có model training mới, annotation mới, submission hoặc final-test run trong lượt này.','',
        f"Runtime receipts161 new calls: tổng worker elapsed **{runtime_total/3600:.3f} giờ**, median **{runtime_median:.2f}s/call**; tối đa4 workers. Đây là tổng thời gian trong worker (có preparation/inference và overhead), không phải end-to-end wall time, kernel-only GPU time hoặc hóa đơn Modal. Reused calls không được cộng như inference mới; không suy chi phí chính xác từ elapsed time.",'',
        '## Checklist hoàn tất và phần chưa thể gọi là hoàn tất','',
        '| Yêu cầu | Kết luận |','|---|---|',
        '| Data/grouping cho official-only benchmark |Đã hoàn tất engineering verification|',
        '| Code chạy thật, matched charged budget, query-excluded scoring |Đã chạy và kiểm metrics/provenance|',
        '| Mở rộng raw baseline toàn bộ170 training clips |Đã hoàn tất và audit|',
        '| Thử nghiệm full/proximity/diversity/reliability/gate |Đã có kết quả thật; chưa chứng minh scheduling advantage|',
        '| Báo cáo MD, per-case/source CSV, plots và scripts tái tạo |Đã có, liên kết bên dưới|',
        '| Original dense references/clinical review |Chưa có; không thể tự khai hoàn tất hoặc biến prediction thành gold|',
        '| Matched external comparator S8/SASVi |Chưa có reproduction tương đương supervision/engine/budget; không khai đã vượt related methods|',
        '| Independent final evaluation của phương pháp mới |Chưa chạy; final test vẫn khóa do chưa đạt proceed criterion, official test chỉ4 nguồn|',
        '| Submission-ready paper |Chưa; template/protocol draft có, các mục tác giả/cơ quan/ethics theo yêu cầu trước vẫn deferred|','',
        'Phần mở rộng baseline170 training clips và các phân tích kèm theo trong lượt này đã được thực hiện và có receipts. Kết luận khoa học hiện tại là các phương pháp đề xuất đã thử chưa vượt uncertainty; hoàn thành compute không biến kết quả âm thành claim mới. Các đầu việc nghiên cứu còn lại được giữ riêng trong completion plan. Cần một hypothesis/protocol mới và evaluation phù hợp nếu tiếp tục nghiên cứu, thay vì dùng final test hoặc đổi endpoint để cứu kết quả.','',
        '## Artifacts để kiểm tra và tái tạo','',
        '- '+link(ROOT/'REPORT_TRAIN_EXPANSION_20261017.md','Full170 training result report'),
        '- '+link(ROOT/'PLAN_COMPLETION_20261019.md','Detailed completion plan and remaining research requirements'),
        '- '+link(ROOT/'REPRODUCE_FULL_TRAINING_20261019.md','Reproduction and private artifact handover guide'),
        '- '+link(ROOT/'CODE_REVIEW_20261019.md','Code invariant review and limitations'),
        '- '+link(BASE/'analysis/per_clip_auc.csv','Per-clip AUC CSV')+'; '+link(BASE/'analysis/per_source_auc.csv','Per-source AUC CSV'),
        '- '+link(BASE/'analysis/train_expansion_summary.pdf','Figure PDF')+'; '+link(BASE/'analysis/train_expansion_summary.svg','Figure SVG'),
        '- '+link(BASE/'final_recovery_audit.json','Final artifact hash seal')+'; '+link(ROOT/'SOURCE_CHECKSUMS.csv','Source checksum ledger'),
        '- '+link(ROOT/'protocols/20261017_train_expansion_v1/PROTOCOL.md','Scientific protocol')+'; '+link(ROOT/'protocols/20261019_train_expansion_recovery_v1/PROTOCOL.md','Recovery protocol'),
        '- '+link(ROOT/'scripts/run_train_expansion_recovery.py','Reproducible execution driver')+'; '+link(ROOT/'scripts/analyze_train_expansion.py','Frozen analysis script')+'; '+link(ROOT/'scripts/plot_train_expansion.py','Plot script'),
        '- '+link(ROOT/'REPORT_RELIABILITY_EXECUTION_20261017.md','Reliability negative result')+'; '+link(ROOT/'REPORT_GROUP_GATE_20261017.md','Grouped gate negative result')+'; '+link(ROOT/'REPORT_ORACLE_DIAGNOSTICS_20261017.md','Oracle diagnosis'),
        '- '+link(ROOT/'paper/main.tex','Existing ISBI protocol/implementation draft (not submission-ready)'),
    ]
    global_path=ROOT/'materials/verification/official_sparse_evidence_audit.json'
    progress_path=ROOT/'materials/verification/completion_work_items_20261019.json'
    if global_path.exists() and progress_path.exists():
        global_audit=read(global_path);work=read(progress_path)
        expansion=global_audit.get('full_training_expansion')
        if global_audit['passed'] and expansion and expansion['clips']==170:
            lines+=['','## Nghiệm thu bàn giao và tiến độ đầu việc','',
                f"Global saved-evidence audit: **passed**, kiểm **{global_audit['source_hashes_verified']} source hashes**, **{expansion['sealed_artifacts_verified']} sealed artifacts** và **{expansion['owned_apps_stopped']} owned apps đã dừng**. Checklist A–D: **{work['technical_scope_A_to_D']['done']} hoàn tất / {work['technical_scope_A_to_D']['pending']} còn chờ** trong phạm vi kỹ thuật của lượt này. Mục E còn **{work['remaining_research_requirements_E']['pending']} yêu cầu nghiên cứu/hoàn thiện bài** (một số phụ thuộc scope/claim được giữ). Không quy các counts không trọng số này thành phần trăm hoàn tất toàn bộ nghiên cứu.",'',
                '- '+link(global_path,'Integrated evidence audit')+'; '+link(progress_path,'Checklist counts and scope'),
                '- '+link(ROOT/'materials/conference/ISBI2027_rules_recheck_20261019.md','ISBI2027 requirements rechecked19October'),
            ]
    (ROOT/'REPORT_GYNSURG_FULL_EXECUTION_20261019.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    decision=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),official_sparse_engineering_execution_complete=True,train_expansion_accepted=170,source_groups=9,new_scheduler_advantage_demonstrated=False,original_dense_annotation_complete=False,clinical_review_complete=False,final_test_open=False,submission_ready=False,default_policy='uncertainty_raw',claim='Computational development evidence only; no demonstrated scheduling advantage')
    (ROOT/'decisions/20261019_full_development_execution.json').write_text(json.dumps(decision,indent=2),encoding='utf8')
    print('Wrote full execution report and scientific decision; no test unlock')

if __name__=='__main__':main()
