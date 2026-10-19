# Plan và rules Project 1 ISBI GynSurg

## Mục tiêu và đóng góp dự kiến
**Interaction-Aware Correction Scheduling for Gynecologic Surgical Video Segmentation**. Câu hỏi: với cùng số lần chuyên gia sửa một mask cho một cấu trúc trên một frame, policy ưu tiên lỗi dự đoán, gần nhau dụng cụ–cơ quan trong mặt phẳng ảnh và độ đa dạng thời gian có tăng chất lượng propagation trên frame khác hơn uncertainty-only? Đây là hỗ trợ annotation offline, không phải tự động hướng dẫn mổ.

Primary hypothesis: tăng normalized area under quality-versus-correction-density curve của procedure-level macro Dice. Contrast chính full-policy minus uncertainty-only. Secondary: IoU, failure rate với threshold từ validation, khoảng cách thời gian tới query, runtime, strata khó và proximity được định nghĩa độc lập. Không diễn giải proximity 2D thành contact, energy activation, tổn thương hay an toàn.

## Audit local đã đọc và hệ quả
- `NoiSoi/GynSurg_Anatomy_Dataset`: 1.450 PNG = 725 cặp trên GANSEG_01–05; trang nguồn mô tả 1.010 frame, cần đối chiếu archive thực tế, không khai số trang web thành số đã nhận.
- `GynSurg_Auxiliary_Tool_Dataset`: 3.158 PNG = 1.579 cặp; không phải toàn bộ primary instrument dataset.
- `Instrument_Anatomy_Dataset`: 10.434 PNG và 2 archive; nhiều ảnh chưa nhãn, 197 cặp recovered chưa mặc định gold. Không dùng unlabelled làm mask background.
- `GynSurg_Clinical_Followup`: có 300 clip AutoLaparo Task2 (10s,25fps,1080p theo audit có sẵn), checkpoint DINOv2 joint và exposure ledger. 24 clip pilot là lựa chọn từ 300 clip, không phải cohort mới.
- Baseline joint `baseline_final.pth` train 4.992 instrument + 725 official anatomy + 238 verified anatomy theo hồ sơ local. Không gọi các procedure nguồn là unseen test đối với checkpoint này. Có thử nghiệm AutoLaparo finetune cũ unresolved, nên phải rà exposure trước dùng test.
- `gynsurg_final`: journal/fuzzy/reliability work đã có; paper mới chỉ được claim correction scheduling với matched budget và query-excluded evaluation. Không tái đóng gói bảng segmentation cũ.
- Ánh xạ clip→procedure của AutoLaparo còn unresolved; source pool 01–10/11–16/17–21 không đủ xác định ca cho bootstrap hay split. Không chia ngẫu nhiên clip thay ca mổ.

## Data acquisition và quyền dùng
Tải đầy đủ các archive segmentation chính thức có liên quan: instrument, auxiliary, anatomy, original instrument/anatomy. Action recognition 24GB/46GB là task khác, không phải dependency của nghiên cứu này. LapGynLHE 75 video yêu cầu ký thỏa thuận và liên hệ tác giả; chỉ tải form, không tự ký/gửi, không hứa đã có video. Link nguồn https://ftp.itec.aau.at/datasets/GynSurge/ . Trang có đoạn license ghi nhầm tên ENID; lưu snapshot và xác minh điều khoản GynSurg thay vì tự gán license đó.

AutoLaparo: ưu tiên toàn bộ 300 clip đã có và tài liệu task1/task2. Task1 video dài chưa có đầy đủ tại máy không cần cho primary protocol clip; chỉ mở rộng nếu cần annotation ca mới. Mọi trạng thái missing/blocked ghi DATA_STATUS.md. Archive download thành công chưa đồng nghĩa có joint dense masks hoặc quyền công bố ví dụ.

## Manifest và split
Manifest tối thiểu: dataset, patient nếu có, procedure/source video có evidence, clip, frame_index, timestamp, fps, path image/mask, class/head, completeness, original size, split, exposure, permission. Không coi GANSEG_01 và INSSEG_01 là cùng ca khi chưa có mapping. Không join hai bộ bằng số ID giống nhau.

Ưu tiên checkpoint fold chỉ train trên train procedures và baseline đã audit exposure. Nếu muốn dùng joint checkpoint frozen trên mọi source case, test phải là procedure khác có nhãn độc lập và exposure sạch. Nếu không xác minh được, chỉ pilot với giới hạn rõ. Mục tiêu feasibility 8–10 held-out procedures, không phải tính power. Nhiều clip/frame không thay thế số ca.

Clinical lead tạo nhãn tham chiếu ở nhịp cố định và mẫu difficult riêng trước model. Có trạng thái visible, absent với đầy đủ annotation, indeterminate và unannotated. Adjudication hai annotator trên subset; lưu bản nhãn v1 và quy tắc lớp. Chọn uterus + instruments khi đủ joint coverage; ovary/tube chỉ thêm khi đủ nhãn trên test.

## Correction protocol đóng băng trước test
Main mode offline, raw frame toàn clip được xem và SAM2 propagate hai chiều. Candidate-query và scoring frames disjoint cố định dùng chung mọi policy; scoring không thay theo query. Một correction unit = full binary mask của đúng một structure/frame; tính cả seed prompt nếu cần. Không so full mask với một click. Chỉ loader correction có thể mở gold candidate khi policy đã chọn; score function tuyệt đối không đọc gold.

Budget grid dự kiến 0,1,2,4 correction/100 raw frames; q_b=floor(b*T/100), cap theo candidate count, báo actual counts/density. Grid 0 nghĩa có frame-wise prediction, không có hidden free GT prompt. Nếu engine cần prompt ban đầu từ GT, prompt đó phải tính ngân sách và baseline 0 cần được định nghĩa riêng. Scoring cố định loại toàn bộ candidate frames; sensitivity loại thêm ±1s quanh query. Báo cả coverage scoring sau exclusion, không so curve với score-set thay đổi.

## Thuật toán
S(t,k)=E(t,k)*(1+alpha*P(t,k))-beta*R(t,k). E là entropy nhị phân foreground + disagreement framewise/propagated, scale theo validation. P=exp(-d/sigma), d minimum predicted instrument–organ distance / image diagonal. R=exp(-|t-tq|/tau) max với query cùng structure; diversity không penalize miễn phí cấu trúc khác. Empty mask: P=0, entropy/disagreement vẫn cho exploration; không chặn query missed organs. Argmax tie theo frame index rồi class ID để tái lập.

Grid validation alpha [0,0.5,1,2], beta [0,0.25,0.5], sigma [0.03,0.1], tau [1,2]s; không thử toàn bộ nếu compute thiếu: freeze sigma/tau, chọn alpha/beta trên development. E không được lấy actual test Dice error. Learned error predictor chỉ tùy chọn nếu OOF độc lập. Không train SAM2 foundation trong vòng đầu.

## Baselines và ablations
| ID | Policy | Mục đích |
|---|---|---|
|S0|Framewise, budget0|Không correction|
|S1|Uniform frame–structure coverage|Baseline coverage|
|S2|Random seed100–109|Trung bình stochastic|
|S3|Entropy only|Contrast primary|
|S4|Estimated error only|Lợi ích disagreement|
|S5|Error+diversity alpha0|Ablation proximity|
|S6|Error+proximity beta0|Ablation diversity|
|S7|Full policy|Đề xuất|
|S8|SASVi hoặc cách re-prompt gần nhất thích nghi budget|Chỉ so khi cùng supervision và engine|

Mọi policy cùng engine/checkpoint/initialization/query pool/scoring pool/budget, reset state mỗi run. Uniform phải phân bố cả frame và structure; không lấy mọi class một frame rồi gọi là coverage. Runtime tách feature extraction, selection, propagation; không đưa annotation time giả thành clinician time.

## Evaluation và thống kê
Dice per-frame/class: cả gold và prediction rỗng = 1 trong tập class được annotation đầy đủ; gold rỗng prediction có =0; unannotated = exclude. Báo sensitivity với absent excluded và số ca/class có mặt. Macro class→clip→procedure, clip ngang trọng số trong ca; nêu rõ convention. Primary AUC trapezoid trên density thực tế, normalize cho range density; không gộp frame toàn bộ thành một Dice duy nhất.

Tạo per_procedure.csv với policy, seed, density, n_queries, class, dice, iou, runtime và n_scored. Average stochastic seeds trong ca trước paired contrast. Bootstrap 2.000 resamples procedure cùng tất cả clip/seed hai policy, percentile95%; report mean paired delta và CI. Strata proximity từ gold độc lập/blinded clinical review, không chỉ predicted proximity. Failure threshold chọn validation và lock.

## Code và cách chạy sau duyệt
`src/policy.py` chọn query chỉ từ observable prediction. `src/surgery.py` chạy loop budget với engine SAM2, fixed scoring và full-mask query, CLI `--help`. `src/sam_engine.py` adapter chính thức SAM2; cần cài snapshot và checkpoint phù hợp. `scripts/audit.py` inventory image/mask hash; source metadata unresolved vẫn phải review. `tests/test_core.py` kiểm leakage query/scoring và policy.

Input sequence NPZ: probabilities T,K,H,W, framewise T,K,H,W, gold T,K,H,W chỉ cho evaluation/query oracle; query_ids/scoring_ids disjoint, fps, procedure_id, jpeg_dir. JPEG resize phải đồng nhất mask; gold của structure không đầy đủ không đưa vào experiment. Frame count chỉ từ sequence thật, không từ số mask sparse. `python -m src.surgery --input sequence.npz --config configs/surgery.json --output results/run_v1` sau duyệt; config xác định checkpoint SAM2, model config và thresholds. Chưa có NPZ clinical hoàn chỉnh vì thiếu grouping/reference masks; không biến inference giả thành kết quả.

## Quyết định và paper
Proceed ngày19 nếu source mapping đủ, propagation full-mask chạy, query excluded, exposure sạch, novelty khác WACV2024 selection/type và development có tín hiệu ổn. Redirect nếu chỉ có isolated sparse frames, không có unseen procedures, gain biến mất khi loại query hoặc chỉ do budget khác. Không dùng test gain làm gate chọn method.

Paper 4 trang: intro0.6; method/protocol1.1; data/eval0.7; results1.2; discussion0.4. Hình workflow và Dice-budget curve CI; bảng primary và ablation; failure examples sampling trước. Nghiên cứu dài hạn: đo phút annotation thật, nhiều clinician, independent cases, inter-reader agreement. Reader study hiện có trong NoiSoi là project khác, không tự chạy hoặc trộn outcome.

## Quy tắc chung và trạng thái duyệt
Đây là gói chuẩn bị ngày 16/10/2026. Chưa huấn luyện, chưa chọn siêu tham số, chưa mở test bằng model, chưa có kết quả mới. Chỉ inventory, tải dữ liệu/tài liệu và kiểm tra logic bằng dữ liệu tổng hợp được thực hiện. `configs/review_status.json` mặc định false; chỉ đổi sau người dùng duyệt protocol cụ thể. Nội dung đề xuất Word được dùng làm tài liệu nghiên cứu, không được coi là lệnh tự động thực thi.

Raw data bất biến. Mọi mask sửa, nhãn bác sĩ, frame trích, prediction và checkpoint ghi sang thư mục phiên bản mới. Không ghi vào `NoiSoi`, không ghi đè journal/followup. File raw được hardlink từ dữ liệu local khi cùng ổ để tiết kiệm đĩa: KHÔNG sửa nội dung tại chỗ vì source dùng chung inode; muốn chỉnh phải tạo bản sao mới. `data/manifests/local_assets.csv` ghi rõ nguồn. Không commit dữ liệu, danh tính bệnh nhân, credential hoặc checkpoint hạn chế lên public.

Mọi run lưu git/source hash, config JSON, seed, môi trường `pip freeze`, GPU/CUDA, manifest hash, split hash, checkpoint hash, thời gian và stdout. Seed nghiên cứu dự kiến 20261016, 20261017, 20261018; random-policy thêm 10 seed cố định 100–109. Không chọn seed tốt nhất bằng test. Tạo venv riêng; requirements là dải tương thích, sau smoke test phải khóa phiên bản môi trường thật.

Tách development/validation/test theo nhóm nguồn trước tuning. Không dùng test chọn loss, alpha/beta, epoch, threshold, postprocessing hoặc ví dụ minh họa thuận lợi. Kiểm tra duplicate SHA256, near-duplicate để gắn cờ, không suy patient ID từ similarity. CI phải cluster theo đơn vị độc lập đã xác minh. Missing annotation không phải background; phải có annotation completeness theo class/head. Không suy consent/ethics từ việc tải được public.

Mọi bảng chính có per-case CSV gốc và script tái tạo. Metrics không loại failure để làm đẹp. Phân tích chính duy nhất được đóng băng; phần khác ghi secondary/exploratory. Chỉ gọi phát hiện mới sau đọc tối thiểu 5 công trình gần nhất và đối chiếu method đầy đủ. Review mọi claim với lâm sàng; không dùng Dice để suy lợi ích điều trị.

## Lịch đến hạn ISBI
| Ngày VN | Công việc và tiêu chí hoàn tất |
|---|---|
|16/10|Duyệt scope, inventory, quyền dùng, checkpoint exposure, phân công hai nhóm|
|17/10|Sau duyệt: baseline end-to-end trên development; kiểm tra evaluator bằng mask tổng hợp và dữ liệu nhỏ|
|18/10|Đọc methods gần nhất; điền novelty matrix; development comparison đầu tiên|
|19/10|PI quyết định proceed/redirect; freeze protocol, split, endpoint, siêu tham số trước test|
|20–21/10|Chạy cấu hình cuối và các seed; hoàn tất nhãn tham chiếu, log compute|
|22/10|Test cuối một lần theo protocol; paired effect/CI; không quay lại tune test|
|23/10|Hình/bảng + manuscript hoàn chỉnh; tái tạo số từ CSV|
|24/10|Review lâm sàng, kỹ thuật, overlap journal, nguồn và disclosures|
|25/10|Mục tiêu nộp nội bộ; kiểm tra PDF/EDAS/authorship|
|26/10|Buffer; hạn chính thức 23:59 EDT, tương đương 10:59 VN 27/10; xác minh portal|

Mốc 19/10 là quyết định khả thi khoa học, không phải đảm bảo accept. Nếu thiếu gate thiết yếu thì tiếp tục nghiên cứu dài hạn, không tạo số liệu hoặc đổi endpoint vì kết quả test bất lợi.

## Chia việc và tài nguyên
Mỗi project: 1 clinical lead chịu trách nhiệm nhãn/định nghĩa/ethics; 1 engineering lead chịu data/model; 1 người phụ trách evaluator/statistics/paper. Tên để người dùng điền. Có thể chia 2 nhóm độc lập; dùng chung người kiểm tra leakage và template. Dự kiến một GPU 16–24GB cho U-Net; SAM2 cần đánh giá VRAM theo độ dài clip. Thử clip 250 frame trước, không tải cả video dài vào RAM. Chỉ ước lượng compute sau benchmark thật; không hứa thời gian train. Không tự tạo job trả phí/thuê GPU.

## Hồ sơ và nghiệm thu
- Data: raw inventory + hash, source/terms, file counts, manifest cặp ảnh-mask, grouping evidence, split review, exclusion list, class map.
- Software: CLI help, smoke tests, kiểm tra dữ liệu, unit tests cho endpoint trọng yếu, dependency lock sau cài thật; code upstream chỉ là snapshot tham khảo.
- Research: protocol đã khóa, novelty matrix có full-text review, configs ablation, per-case metrics, query logs hoặc geometry outputs, CI, failure gallery.
- Paper: draft theo spconf/Word chính thức, hình/bảng thật, references chính xác, ethics/COI/AI disclosures, checklist nộp.

## Nguồn hội nghị đã kiểm tra
https://biomedicalimaging.org/2027/ ; https://biomedicalimaging.org/2027/papers/ . Link template hiện trỏ gói dưới URL 2025; lưu nguyên gói chính thức và bản có sẵn. Quy định có thể thay đổi; kiểm tra lại trước nộp. Hội nghị 25–28/05/2027 tại Lausanne; bài full paper có proceedings, abstract một trang không có proceedings. Không mặc định đổi sang abstract nếu chưa đủ bằng chứng.

SAM2 snapshot yêu cầu torch>=2.5.1/torchvision>=0.20.1; ưu tiên môi trường Linux/WSL theo upstream. Adapter SAM2 chưa chạy GPU end-to-end; cần smoke sau duyệt.

## Kết quả kiểm kê dữ liệu phát hành

Ba archive segmentation đã tải đủ và kiểm CRC: instrument 4.992 cặp, anatomy 725 cặp, auxiliary 1.579 cặp. Tổng 7.296 cặp ghép đúng đường dẫn, không thiếu image/mask. Ảnh có kích thước 720 × 480. Xem `DATA_STATUS.md`, `gynsurg_archive_pairs.csv` và các audit JSON trong `data/manifests`.

Instrument có 8 giá trị pixel [0, 36, 73, 109, 146, 182, 219, 255]; anatomy có [0, 85, 170, 255]. Tên lớp đã đối chiếu conversion scripts chính thức. Mỗi head giữ phạm vi annotation riêng; `joint_annotation_complete=false`. Background của một head không chứng minh class thuộc head khác vắng mặt.

Auxiliary có palette thực tế [0, 60, 120, 190, 255], khác script cũ [0, 60, 85, 170, 255]. Đã ghép 1.579/1.579 ảnh với COCO gốc và rasterize 1.979 polygons: morcellator=60, thread=120, trocar-sleeve=190, cannula=255. `src/labels.py` dùng lookup thực tế đã đối chiếu; không dùng decoder làm tròn của instrument. Độ chính xác nhãn lâm sàng và tính đầy đủ vẫn cần review độc lập.

COCO instruments có 5.774 ảnh có annotation, trong đó 4.992 ảnh chứa category thuộc instrument mapping được phát hành. Anatomy có 818 ảnh có annotation, trong đó 725 ảnh chứa uterus/tube/ovary. Filtering giải thích COCO→PNG, nhưng chưa giải thích toàn bộ các số 11.352/1.010 trong mô tả publication. Báo số artifact đã kiểm, không tự tăng số mẫu. Xem `original_coco_scope_audit.json`.

## Giữ cùng tập frame khi so sánh policy

Phân tích chính đặt `exclusion_seconds=0`: query và scoring pool tách nhau từ đầu, mọi policy dùng cùng scoring frames. Mỗi hàng kết quả lưu hash danh sách scoring frames; script phân tích từ chối paired comparison nếu frame khác nhau giữa policy hoặc seed. Sensitivity loại cửa sổ quanh query phải xây cùng scoring pool theo hợp các cửa sổ của tất cả policy trước khi chạy lại; không so trực tiếp các điểm tính trên tập frame khác nhau. Đây là yêu cầu thiết kế, chưa có kết quả sensitivity.
