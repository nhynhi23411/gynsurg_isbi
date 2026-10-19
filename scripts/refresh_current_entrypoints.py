"""Publish current entry points while retaining explicitly historical audit snapshots."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPORT='REPORT_OFFICIAL_SPARSE_EXECUTION_20261016.md'

def put(path,text):
    path.write_text(text.strip()+'\n',encoding='utf8')

def main():
    if (ROOT/'REPORT_GYNSURG_FULL_EXECUTION_20261019.md').exists():
        print('Legacy16October entrypoint writer skipped. Use finalize_official_sparse.py and current full-training reports; preserve current README/RUNBOOK.')
        return
    banner=f'''**Snapshot lịch sử trước remediation.** Các số 282/300 mapping, 22 tests và gate nhãn tail trong phần dưới thuộc thời điểm audit ban đầu. Hiện hành: 300/300 mapping verified, 33 tests pass, official-only protocol đủ references; xem [{REPORT}]({REPORT}) và `EXECUTION_STATUS.md`. Original tail protocol vẫn thiếu 2.400 masks; không phải scope đang chạy.\n\n'''
    for path in [ROOT/'AUDIT_REPORT_20261016.md',ROOT.parent/'AUDIT_ISBI_20261016.md']:
        old=path.read_text(encoding='utf8')
        if not old.startswith('**Snapshot lịch sử'):
            if path.parent!=ROOT:
                notice=banner.replace(f']({REPORT})',f'](ISBI_GynSurg/{REPORT})')
            else:notice=banner
            put(path,notice+old)
    put(ROOT/'NEEDS_INPUT.md',f'''# Trạng thái đầu vào hiện hành

Ưu tiên code theo yêu cầu người dùng. [{REPORT}]({REPORT}) là báo cáo mới nhất.

Không còn thiếu source mapping cho 300 clip. Task 1 đủ 21 video trên private Modal; Task 2/3 và references của protocol official-only đã đủ. Không cần liên hệ tác giả hoặc tạo nhãn mới để tái chạy benchmark này.

Original 250-frame tail protocol vẫn cần 2.400 masks và clinical review nếu quay lại scope đó; các mục này không được tự khai đã hoàn thành. Official-only 125-frame scope dùng nhãn được dataset phát hành và engineering QA; clinical review vẫn false.

Vấn đề hiện tại là hiệu quả phương pháp: default full và các cấu hình interaction được chọn chưa hơn uncertainty trong development/validation. Giữ final test khóa vì chưa đạt proceed criterion. Không thiếu quyền chạy Modal; không xin lại cùng quyền. Tên tác giả, funding, ethics và nộp bài để pending theo yêu cầu ưu tiên code.
''')
    put(ROOT/'RUNBOOK.md',f'''# Chạy và kiểm code GynSurg

Đọc [{REPORT}]({REPORT}), `configs/active_protocol.json` và amendment `protocols/20261016_official_sparse_v1/PROTOCOL.md`. Plan gốc/snapshot giữ nguyên. Metadata của scope đang chạy là `data/manifests/sequence_official_sparse_v1`; `sequence_review_v5_grouped` thuộc protocol tail cũ.

Từ project root, dùng Python `C:/Users/pc/.venvs/isbi_gynsurg/Scripts/python.exe`:

```powershell
python -m unittest discover -s tests -v
python scripts/check_official_sparse_gates.py
python scripts/finalize_official_sparse.py
python scripts/audit_official_sparse_evidence.py
```

Finalizer làm mới source checksum ledger; audit sau đó kiểm ledger và consistency của evidence. Không chạy finalizer để che một thay đổi chưa review: provenance riêng của mỗi GPU run vẫn giữ snapshot thực tế của run đó. Audit này không rerun GPU hoặc full raw checksum.

Các GPU outputs feasibility/grid/validation đã có và được kiểm accepted cohort. Commands tái chạy tạo run mới/tốn compute:

```powershell
$env:MODAL_PROFILE='nhi544609'
python -m modal run scripts/modal_official_sparse.py --stage pilot
python -m modal run scripts/modal_official_sparse.py --stage feasibility
python -m modal run scripts/modal_sparse_tuning.py
python -m modal run scripts/modal_official_sparse.py --stage validation
```

Không chạy lại để đọc report; xem CSV/JSON trong results. Small-artifact fetch có thể resume mà không rerun inference. Các analysis scripts không overwrite output directory đã tồn tại. Xem `--help` của từng script để truyền run ID thật.

Raw/NoiSoi bất biến, outputs versioned. Prediction không thay gold. Mọi query GT đều tính budget; scoring disjoint; có per-frame metrics/input/config/checkpoint/runtime provenance. Không chỉnh approved_for_test để bỏ gate. Hiện chưa đạt proceed criterion về lợi ích interaction-aware, nên không chạy final test. Trước confirmatory study phải freeze protocol, software, hardware/precision, splits và test-access ledger.
''')
    note=f'''**Trạng thái code hiện hành (16/10/2026):** 300/300 mapping verified; 33 tests pass; 144 feasibility + 108 grid + 9 validation case-runs đã chạy. Official-only scope đủ references, không tạo nhãn mới. Chưa chứng minh full policy hơn uncertainty, final test khóa. Đọc [{REPORT}]({REPORT}) và [RUNBOOK.md](RUNBOOK.md). Phần chuẩn bị ban đầu bên dưới được giữ để tham khảo.\n\n'''
    path=ROOT/'README.md'
    if path.exists() and not path.read_text(encoding='utf8').startswith('**Trạng thái code hiện hành'):
        put(path,note+path.read_text(encoding='utf8'))
    root_note=f'''**Cập nhật GynSurg 16/10/2026:** đã hoàn tất code và các lượt feasibility/tuning/validation được liệt kê trong [báo cáo hiện hành](ISBI_GynSurg/{REPORT}). 300/300 source mapping verified; 33 tests pass; không cần nhãn mới cho amended official-only scope. Kết quả chưa chứng minh lợi ích của policy đề xuất; final test chưa mở. OvarianUS giữ ở giai đoạn chuẩn bị theo ưu tiên GynSurg. Các trạng thái chuẩn bị trong phần dưới là lịch sử.\n\n'''
    for name in ['README_ISBI.md','DELIVERY_STATUS.md']:
        path=ROOT.parent/name
        if path.exists() and not path.read_text(encoding='utf8').startswith('**Cập nhật GynSurg'):
            put(path,root_note+path.read_text(encoding='utf8'))
    print('Updated current code entrypoints; preserved original plan and historical audits')

if __name__=='__main__':main()
