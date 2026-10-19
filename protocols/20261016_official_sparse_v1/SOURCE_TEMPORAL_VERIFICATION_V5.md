# Xác minh source identity khi timestamp clip khác vài frame

Ghi trước lượt verification v5 đầy đủ; không có segmentation quality/policy results được xem khi quyết định. V3 đạt 292/300 bằng fixed offset sequence. Diagnostic trên toàn 300 best candidates cho thấy bảy failures còn lại khớp sáu ảnh khi cho lệch tối đa năm native frames; clip017 first image lệch 16 frames nhưng cả sáu ảnh khớp cùng video trong khoảng dưới sáu giây. Không kết luận nguyên nhân encoder hoặc provider editing khi chưa có bằng chứng.

V5 xác minh **video source identity**, không khẳng định exact clip cut time hoặc affine timestamp mapping. Giữ nguyên pixel thresholds: mean MAE≤.02, max-frame MAE≤.03, mọi correlation≥.99, best-other-source margin≥.01 và ratio≥2.

Thay mô hình thời gian cứng bằng local sequence correspondence: sample đầu được tìm trong ±25 frames quanh nominal candidate start; năm sample sau trong ±5 quanh các offsets [24,49,74,99,124]. Các matched frames phải strictly increasing, khoảng cách liên tiếp 10–50 frames và tổng span100–160frames. Ghi từng matched frame/jitter/error/correlation. Quy tắc này áp dụng cho **mọi shortlisted candidate ở mọi clip và cả alternate sources**, không chỉ tám failures. Không giảm pixel thresholds hoặc dùng expected counts để chọn source.

Chỉ verify source sau all-candidate comparison, official pool/count/duration consistency và visual QA toàn bộ contributing sources cùng mọi original failure. Exact source start ghi là nominal/recovered approximate với six-frame timestamps, không dùng cho alignment nhãn Task 3–Task 2 (alignment đó đã audit riêng). Nếu v5 vẫn không đạt thì giữ unresolved/exclude trước clinical-quality analysis và báo limitation; không tự gán ca.
