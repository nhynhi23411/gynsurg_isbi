# Protocol kỹ thuật tái lập v1 — CHƯA DISPATCH

## Mục tiêu và phạm vi
Posthoc technical study, không thay đổi hoặc cứu primary gate đã thất bại. Không xem validation57/finaltest73. Không sửa frozen science, không fit/tune, không chọn subset/hardware sau xem kết quả. Dữ liệu và hiệu quả development đã biết trước registration này.

## Mẫu cố định theo loại lỗi, không theo lợi ích phương pháp
Sáu clip: 044/132/148 (toàn bộ query-divergent), 002 (metric/AUC failure có cùng query, ID thấp nhất), 004 (new A10G-pass, ID thấp nhất), 003 (new A10-pass, ID thấp nhất). Đây là panel chẩn đoán cố ý chọn, không dùng ước lượng efficacy hoặc tỷ lệ lỗi quần thể.

## Thiết kế dự kiến
6 clips × 2 actual hardware (A10/A10G) × 2 modes (fixed historical raw query/adaptive raw query) × 2 fresh-process repeats =48 executions. Không EVA/automatic/learned candidate. Cùng immutable private inputs/checkpoint/SAM2/scoring/budget, torch/cuda và BF16 như recorded execution. Chỉ hardware, query mode và repeat là các factor được khai báo. Không đồng thời thay sang FP32 hoặc đổi tolerance. Mode fixed chỉ replay lịch sử từ training và mọi mask vẫn tính phí; không nhìn scoring labels để chọn query.

Hardware phải được ghi từ thiết bị thật. Hai yêu cầu Modal A10 không đảm bảo có hai loại actual GPU. Nếu không phân bổ được cả hai loại theo cơ chế xác định trước, dừng cell còn thiếu và báo incomplete; không rerun successful scientific calls để chọn GPU thuận lợi. Không tuyên bố causal hardware nếu factors khác chưa được kiểm soát.

Pilot instrumentation: 003 và132, đủ cả modes/hardware/repeats (=16/48 executions), còn32 chỉ sau structural logging/resource acceptance. Không dùng parity pass hoặc efficacy làm tiêu chí scale. Missing/log-invalid execution được giữ lại; tối đa1 technical retry nếu chưa có outputs hoàn chỉnh và phải ghi riêng, không mặc định cho phép budget.

## Logging bắt buộc trước admission
Worker mới riêng phiên bản, không sửa historical worker. Snapshot private probabilities cho query/scoring frames tại budget0 và sau từng query1–5, SHA/dtype/shape, threshold-0.5 masks, candidate entropy/rank/margin từ state trước quyết định, paid subsets/prompt order/reset state. Save all5query steps, không chỉ budgets0/1/2/5. Logging là observer, không nhập lại vào selector. Ghi actualGPU/torch/CUDA/cuDNN/TF32/autocast/determinism/seed/checkpoint/input/config/source hashes và startup/worker time. Chưa có worker instrumented này trong đợt CPU; không gọi protocol là run-ready.

## Kiểm định kỹ thuật
Pair cùngclip/mode/repeat qua hardware và pairrepeat trong cùnghardware. Báo bitwise hash, max/mean abs probability drift, fraction threshold flips, candidate ranks/margins, first divergent step, query order/set, Dice/IoU và AUC từ cùng snapshot. Fixed-query contrast giúp cô lập query routing nhưng không tự chứng minh kernel causality. So online adaptive query với entropy ghi ngay bước đó để kiểm selector. Không chỉ báo subset đạt. Không nới gate gốc maxmetric0.001/AUCbound0.0001/exactqueries; kết quả mới không ghi đè historicalfailedgate. Nếu cần criterion mới cho nghiên cứu khác, đăng ký riêng và trình bày cả hai.

## Compute admission — chưa có quyền tự vượt cap
Cumulative conservativeplanning đã dùng21.766637GPUh, cap24: headroom2.233363GPUh. Admission thực tế phải audit tài nguyên/billing và thời gian mới; không suy remaining credits từ headroom. Ước lượng sơ bộ raw maximum59s ×1.25 +30s startup cho48calls =1.383333GPUh, **chưa bao gồm overhead snapshot/pairedmode instrumentation/failedcalls**; giữ thêm0.5GPUh => tổng23.649970. Chỉ chạy nếu pilot thật + startup/failure/instrumentation reserve vẫn trong cap24 và thời gian còn hợp lệ. Runtime proxy không phải billing. Max2workers/global4, không tănglimit/unpauseaccount. Operationaldeadline trước22Oct16:05:46UTC giữ nguyên, không launch sau đó. Nếu instrumentation/availablehardware/time không đủ, bàn giao planned-only.

## Kết thúc
Giữ mọi failures và artifacts riêng, không tune efficacy, không mở validation/test. Kết luận có thể là reproducibility unresolved. Chỉ quyết định nghiên cứu xác nhận tiếp sau xem diagnostics, không tự sửa bài thành phương pháp superiority.
