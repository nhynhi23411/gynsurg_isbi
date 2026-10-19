# Quyết định S8 trước correction experiments

Đọc method review và snapshot `materials/code/extracted/SASVi/SASVi-main/src/sam2/eval_sasvi.py`, đặc biệt overseer masks và scene/class-buffer-triggered prompting. SASVi dùng predicted prompts được sinh tự động; thiết kế hiện tại sử dụng oracle full-mask correction có chi phí một action. Thay predicted prompt bằng GT sẽ thay method và supervision.

Do chưa có validated matched-supervision adaptation, S8 tạm không thuộc primary matched-budget suite. Ghi rõ unavailable/incompatible supervision trong paper; không tạo số comparison, không gọi entropy/disagreement heuristic là SASVi. Điều này thực hiện điều kiện “chỉ so khi cùng supervision và engine” của plan, không chứng minh method mới hơn SASVi.

Các comparisons chạy đầu tiên: framewise zero, uniform, random 100–109, uncertainty, error, error+diversity, error+proximity, full. EVA-VOS/HVSA/MA-SAM2/ReSurgSAM2 giữ ở related work với khác biệt supervision/engine đã ghi. Chưa có exhaustive novelty hoặc PI sign-off.
