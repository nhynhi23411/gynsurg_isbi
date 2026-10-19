# Archive và dung lượng
Các archive segmentation được giữ nguyên và verifiedCRC. Dữ liệu local đã accessible trong data/raw/local; archive mới instrument/original có thể chưa extract vì dung lượng. Trước giải nén cần đủ tổng uncompressedbytes cộng reserve1.5GB, xem *.verified.json. Dùng ổ khác nếu cần. Không đổi .part thành .zip bằng tay. Có thể đọc ảnh bằng zipfile + PIL.Image.open(BytesIO(...)) để tránh extracttoàn bộ. Nếu tạo manifest cho archive, ghi rõ archive+memberpath và dùng loader có hỗ trợZIP; loader TorchScript sequence hiện đọc video/JPEG, không đọc ZIP trực tiếp.

Chạy `python scripts/fetch_archives.py` chỉ khi không còn downloader khác cùngoutput; có resume và tự skipverified. MMOTU có fetch riêng tại project2, không cầnthưmụcNoiSoi.
