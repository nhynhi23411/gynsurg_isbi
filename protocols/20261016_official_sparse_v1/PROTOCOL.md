# Computational benchmark dùng nhãn AutoLaparo chính thức

Phiên bản 1, 16/10/2026. Người dùng không có người gán nhãn và yêu cầu thực hiện mọi phần khả thi sau khi được giải thích phương án dùng Task 3. Đây là amendment có ghi nhận trước khi chạy correction comparisons, không sửa snapshot plan gốc và không mở final test.

## Đầu vào và phạm vi

Dùng chỉ sáu annotations đã phát hành của mỗi clip, native frames [0,24,49,74,99,124]. Chạy propagation trên 125 raw frames đầu ở 25 fps. Không dùng các tail images/masks mới, không dùng prediction làm reference, không tuyên bố dense accuracy hoặc toàn bộ clip 10 giây.

Targets: uterus (raw value 180) và hợp tất cả raw tool parts (20–160). Frozen model dùng uterus head và tổng xác suất GynSurg grasper/sealer-divider/hook (1/5/6), phù hợp tên các dụng cụ được source mô tả; đây là cross-dataset class harmonization và có domain mismatch. Không suy individual part IDs thành instrument types. Không coi “ngoài target được dataset annotate” là mọi vật thể ngoài scope vắng mặt.

Reference review mode: `official_release_computational`. Công bố sử dụng human annotations do nhóm dataset cung cấp, kiểm encoding/alignment/target union bằng engineering QA. Không khai có clinical lead của nhóm hiện tại, labels mới, double annotation, adjudication hoặc đánh giá clinical safety. Chỉ cho phép engine chạy khi có scope evidence hash và source grouping verified; không tự đổi `clinical_scope_reviewed=true`. Human annotation completeness ở đây chỉ là completeness theo defined release targets, không independent clinical completeness assessment.

## Query, scoring và budget

Query frames [0,24,49], scoring frames [74,99,124], × hai targets. Query nằm trước scoring; tất cả policies dùng cùng pools. Offline propagation được xem toàn bộ 125 raw frames; không claim online deployment. Một correction là full binary reference mask cho một frame/target. Budget zero chỉ framewise model, không có GT seed miễn phí.

Nominal densities [0,1,2,4] /100rawframes → integer correction counts [0,1,2,5] → actual densities [0,0.8,1.6,4]. Pool có sáu candidates, không cap tại grid này. Primary normalized AUC tích phân trên actual density. Với sensitivity ±1s và union query windows, scoring frames [99,124] luôn còn sau exclusion; ghi coverage và giữ tập này giống nhau giữa policies/seeds.

Primary contrast full vs uncertainty; secondary uniform/random/error/proximity-diversity ablations, IoU, absent-excluded sensitivity, gold-mask proximity strata và distance-to-query. Không có difficult-event clinical labels, vì vậy bỏ claim difficult-event analysis khỏi scope hiện tại. Geometry từ reference chỉ dùng evaluator, không đưa vào policy.

## Grouping và split

Primary exploratory development chỉ official train-source pool; không inference official test khi freeze chưa hoàn tất. Giữ official 170/57/73 split cho phương án benchmark hiện tại; tách tuning/pilot theo verified source khi development mở rộng. Không áp dụng draft repartition 10 test procedures ở phase này. Official test chỉ bốn source procedures nên đây là pilot-scale held-out evaluation, không đạt mục tiêu feasibility 8–10 ca; phải báo hạn chế precision.

Clip eligibility phải dựa pixel identity/QA trước xem segmentation metrics. Bản manifest ghi rõ all-300 recovery hay verified subset, mọi exclusion và nguồn thiếu. Không ép source assignment theo supplementary counts. Không lựa cohort theo Dice/gain. Chưa đưa clip không verified vào bootstrap.

## Method và compute decisions

Frozen checkpoint và SAM2.1 Tiny không train. Deterministic selectors dùng seed 20261016; lặp ba research seeds sẽ trùng selection nếu engine deterministic, nên không gọi chúng là ba independent replicates. Random policy dùng 100–109. Ghi torch/CUDA precision, không mặc định numerical determinism; repeatability spot check cần chạy trước freeze.

Defaults alpha=1,beta=.25,sigma=.03,tau=2s là engineering defaults, chưa validation-selected. Tuning alpha/beta phải trên development procedures đã xác minh, không test. Failure threshold chưa có: để null đến khi chọn bằng validation. BF16 là candidate precision từ technical benchmark, phải dùng cùng precision cho tất cả policies và ghi provenance.

## Comparator S8

SASVi nguyên bản dùng automatic predicted prompts/overseer, khác human full-mask budget. Không gọi heuristic tự viết là SASVi reproduction và không tính predicted prompts như annotation actions. Vòng đầu S8 được ghi `not included: incompatible supervision without a separately validated adaptation`; S0–S7 là matched comparisons. Nếu bổ sung S8 sau này phải version protocol/config trước test. Xem COMPARATOR_DECISION.md.

## Scientific gates

Trước development correction inference: verified source evidence cho clip, reference source/encoding/alignment/completeness theo release, engineering scope QA có hash, exposure/permission recorded, input validation và tests pass. Trước final test: chọn config/thresholds trên development, freeze split/endpoint/seeds/precision/exclusions bằng hash ledger, review study ethics/disclosures và giữ test lock đến khi đủ điều kiện. Academic access hoặc computational scope không tạo institutional ethics waiver.

Kết quả chỉ được gọi computational benchmark/pilot. Không có clinician time, clinical benefit, safety endpoint hoặc new reference validation. Không hứa accept hoặc kịp deadline.
