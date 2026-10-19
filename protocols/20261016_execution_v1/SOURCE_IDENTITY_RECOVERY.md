# Khôi phục nhóm ca mổ từ video — không phải clinical evaluation

Người dùng yêu cầu tiếp tục thực hiện ngày 16/10/2026. Sau khi NAS timeout, tìm được mirror công khai `DBD123/MedHorizon`, revision `3b4e39d5eba8f23c2123ce9557950063da6e0ac6`, thư mục `autolaparo/`. Mirror có 21 MP4, tổng 104.049.666.773 bytes. Metadata của mirror ghi 20 records dù có 21 video; không coi metadata này là chứng minh nguồn gốc.

Chỉ dùng quyền nghiên cứu phi thương mại AutoLaparo đã cấp; giữ license nguồn CC-BY-NC-SA 4.0 và citation. License chung của mirror không thay thế điều khoản nguồn. Không tải video xuống ổ D đang gần đầy. Dữ liệu nằm trong private Modal volume `isbi-gynsurg-task1-recovery-v1` tại profile `nhi544609`, theo revision/video ID; không sửa volume dữ liệu cũ.

## Kiểm tra và bằng chứng

1. Tải nguyên video theo pinned revision, so số byte và SHA256 với LFS manifest; receipt chỉ là bằng chứng integrity của mirror, chưa chứng minh đây là bản nguyên gốc của nhà cung cấp.
2. Đối chiếu 1920×1080, 25 fps và thời lượng với nhãn Task 1 chính thức đã có. Ghi mọi sai khác, không âm thầm đổi thời gian.
3. Lập chỉ mục RGB 32×18 ở 2 fps của cả 21 video. Tìm cửa sổ 5 giây bằng frame đầu/cuối của clip. Không dùng tên clip, thứ tự counts hay prediction để gán ca.
4. Mỗi clip giữ sáu coarse candidates tốt nhất và ít nhất một ứng viên thuộc source video khác. Tinh chỉnh ±30 raw frames quanh mỗi vị trí ở đúng 25 fps.
5. Kiểm tra độc lập toàn bộ sáu frame tại offsets [0,24,49,74,99,124], RGB 128×72. Lưu MAE từng frame, mean/max, source SHA256 và panel query/source để xem lại.

Engineering acceptance được đặt trước kết quả refinement: mean normalized MAE ≤0.012, max frame MAE ≤0.025, khoảng cách tới ứng viên tốt nhất ở source khác ≥0.01 và tỷ số lỗi ≥2. Ties và thiếu ứng viên source khác đều fail. Đây là bằng chứng pixel identity, không phải xác nhận danh tính bệnh nhân hoặc chất lượng segmentation.

Chỉ xem xét đưa mapping vào manifest sau khi tìm trên đủ 21 video, các kiểm tra trên đạt, số clip mỗi video và pools khớp supplementary, kiểm tra visual các ca và outliers. Nếu không đạt, mở rộng retrieval/refinement hoặc giữ unresolved. Counts dùng làm cross-check sau matching, tuyệt đối không dùng để lấp assignment thiếu.

Không mở final test, tune policy hoặc tính primary Dice/AUC trong bước recovery. Việc xem pixel để xác minh nguồn được ghi riêng với quyền truy cập dữ liệu; không coi test hoàn toàn untouched. Nhãn tail, scope/completeness review và điều kiện thống kê trong plan vẫn giữ nguyên.

## Development amendment v2 — 10:15 UTC, trước full-source refinement

Smoke trên tám candidates thuộc official train ở source 02/03 cho thấy raw MAE khoảng 0.010–0.014, dù panel sáu frame tương ứng gần như trùng nhau. Đã sửa bất nhất resize: hai phía đều dùng OpenCV INTER_AREA; lỗi nhỏ vẫn còn. Hai panel đã xem (058→02,079→03) có correlation toàn ảnh >0.998 và từng frame >0.9979, gợi ý sai khác mã hóa/photometric chứ chưa chứng minh byte-identical original video. Không ghi kết luận transcoding là fact khi chưa có video chính thức để so hash.

Thay engineering rule v1 bằng v2, giữ lịch sử v1: mean raw normalized MAE ≤0.02, max frame MAE ≤0.03, và **cả sáu frame RGB correlation ≥0.99**; vẫn yêu cầu margin tới source khác ≥0.01, ratio ≥2 và đầy đủ 21-source search, duration/pool/count cross-check, visual source/outlier audit. NaN/constant images, thiếu alternative source hoặc ties đều fail. Thay đổi này được phát triển trên retrieval smoke thuộc train, trước full-source/held-out refinement; không thay primary research endpoint, policy parameters hoặc scientific split. Không tự duyệt các failure còn lại.

V2 verification queries: `data/manifests/task3_identity_verification_cv2_v2.npz`; provenance ghi interpolation và input hash. V1 PIL-bilinear arrays/reports giữ nguyên để truy vết. Các panel v2 đang lưu chỉ là source-identity evidence, không phải segmentation results.
