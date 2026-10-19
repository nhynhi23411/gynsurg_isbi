"""Consolidate only verified code/data/experiment outputs into the requested MD."""
from pathlib import Path
import json,csv,re,datetime
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def link(p,label):return f'[{label}](<{p.as_posix()}>)'
def main():
    feasibility=ROOT/'results/official_sparse_feasibility_20261016T144620Z'
    tuning=ROOT/'results/official_sparse_tuning_20261016T151226Z'
    validation=ROOT/'results/official_sparse_validation_20261016T153640Z'
    dev=read(feasibility/'analysis/development_report.json');sens=read(feasibility/'sensitivity_1s/analysis/development_report.json');selection=read(tuning/'analysis/selection.json');val=read(validation/'analysis/validation_report.json')
    averages=pd.read_csv(feasibility/'analysis/per_procedure_all_policy_auc.csv').groupby('policy')[['dice_auc','iou_auc']].mean()
    tests=int(re.search(r'Ran (\d+) tests',(ROOT/'materials/verification/execution_unit_tests.txt').read_text()).group(1))
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).isoformat()
    table='\n'.join(f"| {policy} | {row.dice_auc:.6f} | {row.iou_auc:.6f} |" for policy,row in averages.iterrows())
    best=selection['best_overall'];nonzero=selection['best_both_nonzero'];vp='\n'.join(f"| {r['recipe']} | {r['delta']:.6f} | [{r['ci95'][0]:.6f}, {r['ci95'][1]:.6f}] |" for r in val['comparisons'])
    report=f'''# Báo cáo code và thực thi GynSurg với nhãn chính thức

Cập nhật giờ Việt Nam: {now}. **Đã hoàn thành recovery nguồn, chuẩn bị dữ liệu official-only, phần mềm và các vòng feasibility/tuning/validation nêu dưới đây. Chưa chứng minh full policy tốt hơn uncertainty; final test chưa mở.** Theo yêu cầu mới, tập trung code và để thông tin tác giả/nộp bài pending.

## Công việc đã hoàn tất thật

| Hạng mục | Kết quả |
|---|---|
| Task 1 | Tải/kiểm SHA256/index đủ 21 video, 104.049.666.773 bytes trên private Modal volume |
| Source mapping | 300/300 clip → 16 nguồn verified v5; counts/pools/durations khớp; xem 16 source sheets và 5 prior-outlier sheets |
| Task 2/3 | 300 clips; 1.800 image-mask pairs; 1.800 frame alignments pass |
| Official-only input | 300 metadata/reference sets; 3.600 binary target masks; không thiếu reference bắt buộc của scope mới |
| Development feasibility | 9 clip, 9 nguồn, 16 recipes = 144 case-runs; 1.152 aggregate rows và 3.456 frame rows |
| Grid tuning | 12 alpha/beta recipes × 9 nguồn = 108 case-runs; kiểm complete cohort/config/budget/frame outputs |
| Separate validation | 3 nguồn official-val, 3 recipes = 9 case-runs; chọn configs trên development, không tune lại bằng validation |
| Temporal sensitivity | ±1s theo union query logs của tất cả policies/seeds; scoring chung [99,124], coverage 2/3; không rerun model vì query selection độc lập scoring |
| Independent repeat | Full/uncertainty trên clip058: query trajectories và scored Dice/IoU trùng giữa hai jobs; không lưu full probability tensors cho check này |
| Code checks | {tests} tests pass; có fixture AUC tính tay với density không đều, input/gold/frame guards, source identity, budget/scoring, cache và hardware contract |
| Nhãn mới/huấn luyện | 0 nhãn mới; không train foundation model; không sửa raw/NoiSoi |

## Protocol hiện hành

Dùng 125 raw frames đầu (25fps), query [0,24,49], scoring [74,99,124], uterus và hợp tool parts mà Task 3 annotate. Một correction là full mask của một target/frame; budget zero không có GT prompt miễn phí. Nominal densities [0,1,2,4] cho counts [0,1,2,5], actual densities [0,0.8,1.6,4]. Primary comparison giữ normalized procedure-macro Dice AUC, paired procedure bootstrap 2.000 lần.

Reference mode là `official_release_computational`: dùng human annotations do dataset phát hành, với engineering encoding/alignment/scope QA. **Không khai có clinical lead review, inter-reader adjudication hoặc annotation-time study của nhóm hiện tại.** Original 250-frame tail protocol giữ riêng và vẫn thiếu 2.400 masks; thiếu này không chặn scope125frame đã được ghi amendment.

Mapping v5 là source identity qua sáu ảnh khớp theo local temporal correspondence; không khẳng định exact clip cut timestamp. Task3→Task2 mask alignment đã audit riêng. Clip IDs không được coi là patient IDs; source grouping không chứng minh patient independence.

## Kết quả development

| Policy | Mean procedure Dice AUC | Mean procedure IoU AUC |
|---|---:|---:|
{table}

Default full minus uncertainty: **{dev['primary_development']['delta']:.6f}**, CI95% [{dev['primary_development']['ci95'][0]:.6f}, {dev['primary_development']['ci95'][1]:.6f}], n=9 nguồn. Đây là exploratory development trên một clip/nguồn, không phải final test hoặc power analysis. Không có bằng chứng ưu thế của full policy trong vòng này.

Sensitivity ±1s: delta **{sens['primary_development']['delta']:.6f}**, CI95% [{sens['primary_development']['ci95'][0]:.6f}, {sens['primary_development']['ci95'][1]:.6f}]. Queries, annotation budget và prediction trajectories giữ nguyên; chỉ rescore chung trên hai frame còn lại.

## Grid và validation riêng

Grid từ plan: alpha [0,0.5,1,2], beta [0,0.25,0.5], sigma=.03, tau=2s. Tốt nhất toàn grid: **alpha={best['alpha']}, beta={best['beta']}**, mean Dice AUC {best['mean_dice_auc']:.6f}, delta_vs_uncertainty {best['delta_vs_uncertainty']:.6f}. Cấu hình này tắt cả interaction và diversity; không được gọi là bằng chứng ủng hộ hai thành phần đó.

Tốt nhất trong các cấu hình cả hai weights >0: alpha={nonzero['alpha']}, beta={nonzero['beta']}, mean AUC {nonzero['mean_dice_auc']:.6f}, delta {nonzero['delta_vs_uncertainty']:.6f}. CIs của cấu hình được chọn trên development là descriptive, chưa selection-adjusted. Baseline đối chiếu grid đến từ feasibility jobs; xem numerical hardware audit bên dưới.

Các cấu hình này được chạy cố định trên ba nguồn validation khác (clip171/190/211), không đổi tham số theo outcome:

| Recipe so với uncertainty | Paired AUC delta | CI95%, n=3 |
|---|---:|---|
{vp}

Validation này là kiểm khả thi trên ba clip, không phải toàn bộ 57 validation clips. Kết quả vẫn chưa cho thấy ưu thế. **Proceed criterion cho interaction-aware benefit chưa đạt; không mở final test để tìm gain.** Không xóa kết quả âm hoặc đổi endpoint vì outcome này.

## Code đã bổ sung và kiểm tra

- `src/surgery.py`: lưu per-frame/class/budget Dice/IoU, presence, nearest same-structure query distance, reference-only proximity; query mask hashes; reject fractional IDs, nonbinary gold, invalid fps.
- `src/policy.py`: chỉ tính features trên eligible frames, bỏ proximity computation khi không sử dụng; test đối chiếu selection với full-frame reference; reject typo policy và invalid scales.
- `src/result_validation.py` và cohort validator: reject thiếu/trùng rows, thiếu seed/density/structure, nonfinite metrics, mismatched frames/hash/config/budget và incomplete declared cohorts; kiểm frame means tái tạo aggregate metrics.
- `src/reference_scope.py`: tách clinical review khỏi official-release benchmark scope; cần evidence hash đúng, không tự đặt clinical=true.
- `src/run_suite.py`: cùng engine/predictor, bounded CPU cache, registry/input/checkpoint/source/environment/GPU/CUDA provenance; cache runtime được ghi amortized.
- `src/runtime_contract.py`: ghi GPU thực tế; optional `required_gpu_name` từ chối sai model GPU. Configs hiện hành để unset; phải chốt hardware contract trước confirmatory runs.
- Scripts recovery/identity v3–v5 giữ reports phiên bản cũ, kiểm all-candidate alternatives và temporal correspondence; không gán source theo counts.
- Scripts feasibility, tuning, validation, sensitivity và fetch small artifacts đã chạy thật. Source hashes của từng job được giữ, không giả định mọi run dùng cùng snapshot mới nhất.

S8 chưa có validated matched-supervision adaptation. SASVi dùng automatic predicted prompts, khác charged human-mask protocol; đã ghi exclusion decision theo điều kiện của plan. Không viết một heuristic khác rồi gọi là SASVi reproduction.

## Tái lập và numerical limits

Repeat check ở clip058 đạt exact query/scored-metric equality cho hai policies. Check tương đương error-only và full(alpha=beta=0) trên chín clips cho query trajectories giống nhau; frame Dice khác tối đa **0,000770** giữa một số jobs. Trong sample này, khác biệt trùng với đổi GPU name NVIDIA A10↔A10G; cùng named GPU cho metrics exact ở các cặp đã kiểm. Chưa cô lập hoàn toàn nguyên nhân, không claim cross-hardware bitwise reproducibility.

{link(ROOT/'materials/verification/gpu_variant_numerical_audit.json','Audit số học/GPU')} lưu từng clip, actual hardware và input/checkpoint equality. PyTorch cũng không bảo đảm tái lập hoàn toàn giữa platforms/devices; xem [PyTorch 2.7 reproducibility](https://docs.pytorch.org/docs/2.7/notes/randomness.html). Mọi policies trong một clip suite chạy trên cùng device và engine; hardware/precision phải được ghi và khóa trước final study. Không dùng paired development CI để tuyên bố gain nhỏ hơn numerical floor.

## Vị trí dữ liệu và artifacts

- Task1: Modal profile `nhi544609`, volume `isbi-gynsurg-task1-recovery-v1`, revision `3b4e39d5eba8f23c2123ce9557950063da6e0ac6`; mỗi nguồn có video, receipt/hash, ffprobe và evidence. Mirror checksum không phải provider-original checksum; NAS vẫn timeout.
- Task2 local: `data/raw/local/AutoLaparo/autolaparo_task2/clips/clips/` (300 clips, 3.776.270.793 bytes).
- Task3 local: `data/raw/AutoLaparo_Task3/`; derived binary targets riêng ở `data/derived/task3_official_v1/`.
- NPZ/JPEG tensors lớn và full job logs: private Modal volume `isbi-gynsurg-results-v1`; các CSV/JSON/log cần review đã tải local. Không tải 104GB Task1 vào ổ D đang ít dung lượng.
- {link(ROOT/'data/manifests/procedure_mapping_verified_v5.csv','Mapping master đã verified')}.
- {link(ROOT/'data/manifests/official_sparse_gate_status.json','Data readiness của amended protocol')}.
- {link(ROOT/'materials/verification/execution_unit_tests.txt','Unit/integration test log')}.
- {link(feasibility/'analysis/development_report.json','Development report máy đọc')} và {link(feasibility/'analysis/all_policy_development_curves.png','Curve tất cả policies')}.
- {link(feasibility/'sensitivity_1s/analysis/development_report.json','Sensitivity report')}.
- {link(tuning/'analysis/all_grid_results.csv','Toàn bộ grid results')} và {link(tuning/'analysis/selection.json','Selection evidence')}.
- {link(validation/'analysis/validation_report.json','Validation report')}.

## Tiếp tục code sau khi review

Chạy từ project root bằng venv `C:\\Users\\pc\\.venvs\\isbi_gynsurg\\Scripts\\python.exe`; set `MODAL_PROFILE=nhi544609` trong process khi dùng Modal, không đổi default profile toàn máy.

```powershell
python -m unittest discover -s tests -v
python scripts/check_official_sparse_gates.py
python scripts/finalize_official_sparse.py
python scripts/audit_official_sparse_evidence.py
python -m modal run scripts/modal_official_sparse.py --stage pilot
python -m modal run scripts/modal_official_sparse.py --stage feasibility
python -m modal run scripts/modal_sparse_tuning.py
python -m modal run scripts/modal_official_sparse.py --stage validation
```

Các commands Modal tạo run ID mới và tốn compute; không cần chạy lại những outputs đã verified chỉ để xem báo cáo. Dùng `fetch_sparse_results.py` để resume tải small artifacts, không rerun GPU vì thiếu bản local. Các scripts analyze yêu cầu run ID thực và từ chối overwrite analysis directory đã có.

Final inference CLIs vẫn yêu cầu verified inputs và test authorization. Không đổi `approved_for_test` chỉ để bỏ lỗi gate. Với evidence hiện tại, bước khoa học tiếp theo là review/redirect mechanism hoặc bổ sung dữ liệu thích hợp; code đã cung cấp nền để thực hiện, nhưng không thể tạo lợi ích phương pháp từ kết quả không có gain. Tác giả/funding/ethics/publication được để pending theo yêu cầu ưu tiên code; chưa nộp bài.

## Chốt audit code và trạng thái compute

Lần kiểm cuối: {tests} tests pass; source/config ledger và original plan hash được kiểm bởi `scripts/audit_official_sparse_evidence.py`. Evidence hiện hành ở [audit JSON](<D:/Thầy Khánh/ISBI/ISBI_GynSurg/materials/verification/official_sparse_evidence_audit.json>); đây là kiểm consistency artifacts, không phải một đợt rehash toàn bộ raw data hoặc clinical review mới. Entry point protocol rõ ràng ở `configs/active_protocol.json`; [runbook](<D:/Thầy Khánh/ISBI/ISBI_GynSurg/RUNBOOK.md>) có lệnh kiểm và tái chạy.

Modal profile nhi544609 trả container list rỗng ở lần kiểm cuối. Snapshot billing cấp account ghi metered 5.40292415 USD, billed 0 USD và credit adjustment -4.60 USD; không phải cost riêng dự án hoặc remaining balance. Private volumes vẫn giữ dữ liệu. [Billing snapshot](<D:/Thầy Khánh/ISBI/ISBI_GynSurg/materials/verification/modal_billing_after_official_sparse.json>).

Template hiện tại compile thành công thành ba trang ở `results/paper_protocol_draft_20261016T155203Z`; đã xem cả ba trang và workflow figure. Đây vẫn là protocol/implementation draft, không có final results. Ưu tiên bàn giao code, CSV/JSON và báo cáo MD theo yêu cầu hiện tại.
'''
    diagnostic=ROOT/'REPORT_POLICY_DIAGNOSTICS_20261017.md'
    if diagnostic.exists():
        report+='\n## Phân tích cơ chế bổ sung ngày 17/10/2026\n\nĐã phân tích paired frame/structure/budget outputs và initial proximity trên chín nguồn development, dùng CPU và saved inputs; không có GPU inference mới. Xem [báo cáo chẩn đoán policy](<'+diagnostic.as_posix()+'>). Proximity chưa có tín hiệu dự báo lỗi hữu ích trong sample này; các ca giảm mạnh nhất gắn với diversity/query allocation. Không đổi policy hoặc mở final test.\n'
    debug=ROOT/'REPORT_DEBUG_REPLAY_20261017.md'
    if debug.exists():
        report+='\n## Replay debug ngày 17/10/2026\n\nĐã bổ sung chế độ --debug-trace và replay hai ca059/086 với full/uncertainty, dùng GPU riêng cho quan sát states. Xem [báo cáo debug replay](<'+debug.as_posix()+'>). Giữ nguyên config và budget; debug snapshots/candidate scores dùng để phân tích, không cung cấp gold cho selector. Lượt replay này tách khỏi vòng CPU diagnostics ở trên.\n'
    prototype=ROOT/'REPORT_RELIABILITY_SCALED_V1.md'
    if prototype.exists():
        report+='\n## Thiết kế prototype reliability + relative diversity\n\nĐã triển khai/freeze sáu recipes và kiểm CPU synthetic. Snapshot thiết kế trước execution ở [báo cáo prototype](<'+prototype.as_posix()+'>); kết quả GPU thật, nếu đã có, được liệt kê ở mục thực thi dưới đây. Không thay kết quả v1 hoặc mở final test.\n'
    execution=ROOT/'REPORT_RELIABILITY_EXECUTION_20261017.md'
    if execution.exists():
        report+='\n## Thực thi prototype trên GPU\n\nTrạng thái thiết kế ở mục trên là snapshot trước execution. Phiên bản hardware A10 đã có outputs thật; xem [báo cáo thực thi prototype](<'+execution.as_posix()+'>) và `materials/verification/reliability_execution_acceptance.json` để biết cohorts đã hoàn tất. Tất cả kết quả vẫn exploratory; không unlock final test.\n'
    oracle=ROOT/'REPORT_ORACLE_DIAGNOSTICS_20261017.md'
    if oracle.exists():
        report+='\n## Chẩn đoán dư địa cải thiện\n\nCPU-only diagnostic trên saved states đã đo restricted oracle và các tín hiệu observable. Oracle có quyền xem gold chỉ để phân tích; không phải algorithm thực tế. Xem [báo cáo oracle/signal](<'+oracle.as_posix()+'>). Không train gate hoặc mở final test.\n'
    gate=ROOT/'REPORT_GROUP_GATE_20261017.md'
    if gate.exists():
        report+='\n## Gate theo nhóm nguồn\n\nĐã kiểm chứng output gate bằng nested source-group development, tách nguồn đánh giá khỏi chọn ngưỡng. Kết quả không đạt tiêu chí phát triển; giữ raw default và final test khóa. Xem [báo cáo group gate](<'+gate.as_posix()+'>). Đây vẫn là dữ liệu đã xem khi thiết kế, không phải validation độc lập.\n'
    expansion=ROOT/'REPORT_TRAIN_EXPANSION_20261017.md'
    if expansion.exists():
        report+='\n## Mở rộng raw baseline trên training\n\nScope170 clip training thuộc cùng chín nguồn,161 inference mới và chín raw runs tái sử dụng. Đọc [báo cáo expansion](<'+expansion.as_posix()+'>) và progress receipt trước khi kết luận đã hoàn tất. Không dùng thêm clip như thêm ca độc lập, không mở validation/final test.\n'
    consolidated=ROOT/'REPORT_GYNSURG_FULL_EXECUTION_20261019.md'
    if consolidated.exists():
        report+='\n## Báo cáo tổng hợp đã nghiệm thu ngày19/10\n\nXem [báo cáo đầy đủ](<'+consolidated.as_posix()+'>) cho170/170 training clips, source-level primary/secondary analyses, restricted oracle, figures và final audit. Compute hoàn tất không chứng minh scheduling advantage; original dense labels/clinical review và independent final evaluation vẫn chưa hoàn tất. Kế hoạch đầu việc còn lại: [completion plan](<'+(ROOT/'PLAN_COMPLETION_20261019.md').as_posix()+'>).\n'
    out=ROOT/'REPORT_OFFICIAL_SPARSE_EXECUTION_20261016.md';out.write_text(report,encoding='utf8');print('Wrote',out)
if __name__=='__main__':main()
