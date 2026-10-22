# Điều tra tái lập bằng outputs đã lưu — 22/10/2026

Mốc code được chốt: `adb41c1f207744bdb4901505c35a0e3f7454f7bd`, tag `freeze-completion-20261022`. Nhánh mới `research/reproducibility-20261022`; không sửa thuật toán hay gate gốc. Đây là diagnostic posthoc trên development đã biết kết quả, không phải validation mới.

## Phát hiện từ 170 records

- Đối chiếu lại đủ 170 clip/9 nguồn, 680 hàng ngân sách và 850 bước query lịch sử. Tái dựng query từ entropy lịch sử với tie-break frame rồi head khớp toàn bộ.
- Input SHA của mỗi cặp lịch sử/hiện tại khớp. Phiên bản PyTorch ghi nhận đều `2.7.1+cu126`; điều này chưa chứng minh toàn bộ môi trường/kernel/settings giống nhau.
- Budget 0 khớp đến sai số tối đa 2.22e-16 trên Dice/IoU được ghi. Đây là metric equality, không chứng minh mọi tensor giống từng bit.
- Giữ nguyên 20 failures. **17 failures có cùng toàn bộ chuỗi query**, nên khác biệt query không phải lời giải thích đầy đủ. Cả 20 chuyển từ historical A10 sang current A10G, nhưng còn các clip A10G đạt; chưa xác lập quan hệ nhân quả.
- 11/20 vượt metric threshold 0.001; 9 trường hợp còn lại trượt AUC bound 0.0001 dù metric từng mốc không quá 0.001. Không nới threshold.

| Clip | Query đầu khác | Budget đầu vượt metric threshold | Final query set giống | Historical entropy margin tại bước khác |
|---|---:|---:|---|---:|
|044|5|1|False|0.001177266|
|132|4|1|True|0.000321233|
|148|2|2|True|0.011526484|

044 và 132 đã lệch metric từ budget 1, trước query divergence. 148 lệch query tại bước 2; chỉ chấm các budget 0/1/2/5 nên không xác định thời điểm tensor bắt đầu lệch chính xác. 132/148 có cùng tập cuối nhưng thứ tự khác, còn 044 khác tập cuối. Thứ tự correction có thể quan trọng; records hiện tại chưa cô lập được tác động đó.

Các margin trong bảng thuộc **historical paid state**, không phải margin đo ở current state. Current trace không lưu entropy từng candidate hoặc tensor từng bước. Vì thế không thể kết luận lỗi chỉ do near-tie, kernel, mixed precision, reset state hay thứ tự lời gọi. Chưa có cơ sở gán một root cause cụ thể hoặc sửa selector để cứu kết quả.

## Hướng khoanh vùng tiếp

Ưu tiên logging xác suất trước/sau correction, actual GPU, toàn bộ precision/determinism flags, candidate entropy và thứ tự prompt. So fixed historical query replay với adaptive replay, giữ cùng baseline/checkpoint/input, trên hardware được xác nhận. Fixed replay vẫn lệch sẽ cho thấy query selection không phải yếu tố duy nhất; adaptive lệch thêm mới định lượng được nhánh chọn query. Cần cả lặp trong cùng hardware để phân biệt repeatability với khác biệt giữa hardware. Observer mới chỉ nhận prediction tensors; không nhận gold và không tham gia quyết định query.

Không GPU job nào được tạo trong đợt điều tra này. Validation/test và original failed gate giữ nguyên. Protocol bước tiếp được chuẩn bị riêng; chỉ dispatch sau khi worker instrumentation và resource/time admission đủ bằng chứng, không lặp theo hiệu quả hay hardware thuận lợi.
