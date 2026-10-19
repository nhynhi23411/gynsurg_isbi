# Vận hành protocol — draft, chưa mở final test

Giữ nguyên câu hỏi, primary contrast full-policy vs uncertainty-only, endpoint procedure-macro Dice normalized budget AUC và paired procedure bootstrap trong plan. Các chi tiết dưới đây làm rõ đầu vào sau khi khôi phục Task 3; chưa có kết quả correction evaluation.

## Target và backbone

Checkpoint joint cố định có SHA256 `c8c763312cba9cff5bb02d492e548cce0cd0975d13c54c1a40f93e8f578bf192`. Manifest 5.955 training rows đã kiểm hash: 4.992 instrument, 725 official anatomy, 238 verified anatomy; không có AutoLaparo row hoặc ảnh trùng SHA256 với Task 3. Đây là bằng chứng cho artifact cụ thể, không xóa lịch sử benchmark AutoLaparo trong paper trước.

Target draft: uterus + hợp các loại dụng cụ được AutoLaparo annotation. Đề xuất mapping grasper/dissecting-grasping→GynSurg grasper (1), LigaSure→sealer-divider (5), electric hook→hook (6), gộp shaft và manipulator. Bản export matched dùng `--instrument-classes 1,5,6 --merge-selected`; không cộng miễn phí các lớp ngoài scope. Clinical lead cần chốt mapping/completeness trước evaluation. Không dùng tên các pixel 20–160 để suy từng loại nếu chưa có lookup chính thức.

SAM2.1 Tiny cố định, offline hai chiều, state riêng mỗi structure. Tắt optional SAM2 postprocessing một cách tường minh vì môi trường không build CUDA hole-filling extension; mọi policy dùng cùng setting. Không train SAM2 trong vòng đầu.

## Query và scoring pools đề xuất

Mỗi clip có 250 raw frames, 25 fps. Task 3 có sáu frame 0-based [0,24,49,74,99,124], đã đối chiếu pixel với video. Draft query pool [0,24,49,74,99] × hai structure = 10 correction candidates. Scoring pool [124,149,174,199,224] × hai structure, tách query hoàn toàn. Bốn tail-frame mới cần bác sĩ gán nhãn; không tạo pseudo-gold hoặc tự điền absent.

Budget [0,1,2,4]/100 raw frames cho counts [0,2,5,10], actual densities [0,0.8,2,4]. Code có cap khi pool nhỏ hơn budget và báo actual density. Kết quả chỉ đại diện các scoring snapshots, không phải dense accuracy trên toàn 250 frame. Query window và scoring window hạn chế phải được nêu rõ khi diễn giải.

Sensitivity ±1s dùng hợp query windows của mọi policy/seed, rồi rerun với cùng scoring pool. Với draft này, common scoring tail [149,174,199,224] còn lại khi tính hợp tại ngân sách tối đa. `scripts/build_common_scoring.py` tạo input sensitivity riêng; config `exclusion_seconds=0` khi rerun để tránh loại thêm theo từng policy. Analyzer kiểm hash scoring frame và từ chối paired comparison nếu khác tập frame.

## Grouping và thống kê

Supplementary chính thức cho biết 16 video có clip Task 2: 9 trong pool 01–10, 3 trong 11–16, 4 trong 17–21. Chưa có bằng chứng từng clip thuộc video nào. Counts và clip order không đủ để đánh dấu mapping verified. Dùng `procedure_mapping_tasks_v1.csv` để điền evidence; không suy procedure_id=clip_id.

Official train/val/test có 170/57/73 clip. Theo thông tin hiện có, official test chỉ có 4 procedure đóng góp clip; không khai 73 clip là 73 ca hay 21 ca. Mục tiêu 8–10 held-out procedures trong plan chưa đạt với riêng official test này. Cần báo hạn chế/power hoặc có cohort bổ sung hợp lệ; không tạo CI hẹp bằng frame-level resampling.

## Thứ tự tiếp tục

Hoàn tất mapping và nhãn tail, clinical scope review và quyền dùng theo institution. Validate metadata/references. Chạy một clip validation end-to-end trên GPU, đo VRAM/runtime thật rồi chọn quy mô development. So sánh/ablation và khóa siêu tham số trên development/validation. Freeze split, sources, configs, endpoint, failure threshold và test-access record trước test cuối. Chưa có final test hay policy comparison được thực hiện.
