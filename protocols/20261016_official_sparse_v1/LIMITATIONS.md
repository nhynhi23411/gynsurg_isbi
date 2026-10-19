# Giới hạn cần giữ trong báo cáo

Nhãn chính thức nằm ở sáu snapshots 0–5 giây. Task 2 đặt motion anchor tại giây thứ năm theo mô tả nhà cung cấp. Vì vậy amended benchmark không đánh giá correction sau motion anchor hoặc cuối clip 10 giây. Propagation có thể dễ hơn và policy gain có thể nhỏ; không suy hiệu quả trên motion/occlusion khó từ tập này.

Chỉ có ba candidate frames × hai structures và tối đa năm corrections ở grid hiện tại. Những điểm budget cao gần bão hòa pool, nên phải xem cả curve và query logs, không chỉ điểm cuối. Mọi phương pháp dùng cùng giới hạn này. Không mở rộng candidate pool bằng pseudo-gold để tăng gain.

Source grouping là procedure/video identity; patient identity không được xác minh. Official final-test pool có bốn contributing procedures. Development feasibility có một clip mỗi nguồn để kiểm khả thi, không phải cohort đầy đủ hoặc independent final result. Intervals ở một procedure là degenerate; intervals ở ít procedure có precision hạn chế.

Nếu development không cho tín hiệu khác uncertainty, ghi kết quả âm/trung tính và dùng proceed/redirect criterion của plan; không chọn lại endpoint, bỏ failures hoặc xem test để tìm gain. Đánh giá rộng hơn sau motion anchor cần annotation bổ sung hoặc dataset phù hợp khác có provenance/grouping/reference tốt.
