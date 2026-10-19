# GPU, precision và cache — engineering review trước scientific freeze

Ngày 16/10/2026. Các benchmark dùng clip 001 thuộc official train pool, 250 frames, working 720×405. Prompt tại frame 0/structure 0 và frame 124/structure 1 đều lấy từ frozen framewise predictions; không dùng gold. Không có clinical Dice/AUC hoặc policy comparison.

| GPU / precision | Replay đầu (s) | Replay thứ hai (s) | Chú thích |
|---|---:|---:|---|
| A10 float32, không cache | 42.76 | 85.37 | Reset/replay cả hai structure ở lần thứ hai |
| A10 float32, structure cache | 42.89 | 43.37 | Tái dùng output của structure đã chọn ở lần trước |
| A10 bfloat16, structure cache | 18.68 | 19.07 | Autocast theo ví dụ video predictor của official SAM2 README |
| L4 bfloat16, structure cache | 23.53 | 23.47 | Cùng clip, engine và prompt source |

Tại mỗi precision riêng, warm-prefix output khớp bitwise với cold-prefix output đã lưu trong cache. Đây không phải bằng chứng numerical equivalence giữa float32 và bfloat16, giữa hai GPU hoặc về accuracy lâm sàng. Framewise segmenter vẫn float32; chỉ SAM propagation có tùy chọn autocast. Config mặc định nghiên cứu vẫn chưa được freeze theo precision/hardware cho test.

## Cache và annotation budget

Cache chỉ tạo/fetch trong `correct()` sau khi policy đã chọn query và mở mask có tính budget. Key gồm clip/frame-image hashes, baseline probabilities, checkpoint/config/precision, engine source, structure và SHA256 của đúng tập prompt đã chọn của structure đó. Không chuyển output conditioned-on-mask của structure khác vào structure chưa được sửa. Mỗi run vẫn ghi đủ query log và actual budget.

Engine đang replay prompts theo frame order, reset state độc lập theo structure; do đó output của một structure chỉ phụ thuộc đúng tập selected masks và immutable context. Cache là memoization của phép tính này, không thay selection score hay supervision. Với năm candidate frames/structure và hai structures, tối đa 2×(2^5−1)=62 nonempty prompt sets. Một cached structure output float16 ở 250×405×720 chiếm 145.800.000 bytes; cả 62 state khoảng 9,04 GB. Suite mặc định bounded cache 10 GiB và cần RAM container phù hợp (benchmark/khuyến nghị hiện dùng 24 GiB). LRU vẫn bảo vệ khi input dimensions/candidate pool lớn hơn; không hứa bound 62 cho thiết kế khác.

`python -m src.run_suite` dùng chung predictor/cache cho một reviewed clip qua các recipes; giữ grouping/scope/reference/test gates như CLI từng policy. CSV đánh dấu runtime là amortized suite execution. Order-dependent warm timing không được dùng làm speed advantage riêng của một policy; secondary runtime comparison cần uncached benchmark trên fixed cases riêng.

## Giá và kế hoạch compute

Giá niêm yết Modal kiểm ngày 16/10/2026: A10 $0.000306/s, L4 $0.000222/s, CPU $0.0000131/core/s, RAM $0.00000222/GiB/s. Với request 4 CPU/24 GiB, base resource rate tương ứng khoảng $0.00041168/s và $0.00032768/s; chưa tính idle/load/region/storage/egress. Nguồn: https://modal.com/pricing . Đây là rate model, không phải invoice hoặc remaining credit balance.

Trên benchmark nhỏ này A10 bfloat16 nhanh hơn L4 bfloat16; resource cost cho một cold structure replay gần nhau. A10+bfloat16 là candidate cho validation/prefreeze, cần chốt cùng setting cho mọi policy. Không bật đại trà chỉ từ throughput và không suy clinical quality tương đương. So sánh chất lượng vẫn đợi references/scope và đúng split. Billing snapshots thực tế phải được kiểm trước đợt cohort; không tự chế remaining credit từ credits-used.

Run artifacts: `results/full_clip_technical_20261016T100939Z`, `...T104706Z`, `...T105455Z`, `...T105726Z`. Mỗi JSON có source/checkpoint/export/clip hashes, GPU, precision khi được bổ sung, timings và cache checks. Các kết quả chỉ là engineering; không điền vào bảng primary của manuscript.
