# Draft split theo procedure để đạt mục tiêu feasibility — chưa áp dụng

Ngày 16/10/2026, trước primary correction evaluation. Official AutoLaparo test pool chỉ có bốn source videos đóng góp clip; không coi 73 test clips là 73 independent cases. Plan muốn 8–10 held-out procedures. Không có kết quả policy/test nào để chọn split dựa performance.

Một phương án cần review/freeze sau source recovery là chia lại **16 source procedures có clip** cho nghiên cứu correction mới, vẫn giữ nguyên official split làm provenance. Frozen GynSurg checkpoint đã được audit không có AutoLaparo training rows; vì vậy official training pool của AutoLaparo không tự động là training exposure của checkpoint này. Lịch sử các benchmark cũ vẫn phải disclose và không gọi dataset pristine.

Quy tắc dự thảo, không dùng độ khó/accuracy:

- Development: source 01/02/04, vì các clip 001/058/086 đã được xem model trong technical preview. Assignment từng clip vẫn cần pixel verification trước khi xác nhận các source này; không coi counts là mapping proof.
- Validation: ba source có clip với ID nhỏ nhất còn lại: 03/05/06. Dùng riêng để chọn alpha/beta, các tham số nuisance và failure threshold.
- Proposed held-out test: 07/08/10/11/13/16/17/18/19/21, tổng mười procedures theo supplementary. Không đưa original official-test sources sang development/validation.

Nếu source correspondence hoặc exposure audit không phù hợp thì phương án chưa có hiệu lực. Đây là repartition của các case đã có, không phải cohort mới hay external independent validation. Recovery đã xem pixels cho identity và metadata; quyền truy cập hành chính này phải disclose riêng với performance evaluation.

Trước áp dụng cần chốt với scientific/clinical lead: mục tiêu correction study trên split mới, reference completeness/adjudication, nguồn ID đã verify, overlap với nghiên cứu cũ, và test access ledger. Phải lưu split manifest hash cùng protocol amendment trước tuning/final test. Không thay `approved_for_test=false`, không cập nhật sequence split hoặc chạy inference trên proposed test trong draft này. Nếu giữ official split thay vì repartition thì báo trung thực n=4 test procedures và giới hạn precision/power.

Các preview/benchmark đã chạy chỉ là engineering checks trên clip thuộc development dự kiến; không được biến chúng thành effect estimate hay tuning outcome. Nhãn tail chưa có vẫn là gate độc lập trong cả hai phương án.
