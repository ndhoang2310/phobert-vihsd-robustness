# Protocol baseline đầu tiên

## Mục đích

Đưa pipeline dữ liệu/EDA và lần fine-tune PhoBERT đầu tiên lên Kaggle; tạo checkpoint cố định cho các phép thử robustness về sau.
Không dùng mốc F1 63,5–65% trong context như một điểm PhoBERT cần tái lập: paper ViHSD 2021 không có PhoBERT trong bảng kết quả.

## Lựa chọn được chốt trước đánh giá test

- Dataset: archive chính thức tại commit `70fe50c9c94145e3b9be5614a048cbd5b480fa54`.
- Giữ split chính thức; loại đúng hai ID train rỗng theo ledger. Không deduplicate train/validation/test.
- Model: `vinai/phobert-base`; revision đã resolve `01daacda68afe13d83023d16ec647239e344a1e6`.
- Preprocess: Unicode NFC, khoảng trắng/ký tự control, RDRSegmenter; giữ case, dấu, emoji, stopword, UGC.
- Mỗi comment là một mẫu; ghép đầy đủ các câu segmenter trả lại bằng khoảng trắng.
- Tokenizer slow chính thức, special tokens, right truncation, max length 256, dynamic padding tới bội 8.
- Seed 42; model kiểm tra overfit bị hủy, full model được khởi tạo lại sau khi đặt seed.
- Train 5 epoch, effective batch 32, LR 2e-5, AdamW, linear schedule, warmup 10%, clip 1.0.
- Precision chỉ chọn BF16 khi GPU hỗ trợ native (`including_emulation=False`); T4 dùng FP16.
- Chỉ dùng validation Macro-F1 để chọn best; tie giữ checkpoint đầu. Không tune nhiều config hoặc seed trong đợt này.
- Test chỉ chạy ở bước cuối, sau khi chọn best.

## Bằng chứng chất lượng cần giữ

1. Manifest/checksum và audit trên dữ liệu gốc.
2. Ledger loại mẫu và source hashes.
3. EDA raw + tokenization, bảng số liệu và hình đã xem.
4. Kiểm thử pipeline; smoke loss, gradient, model save/reload; optimizer state roundtrip; overfit kiểm tra model học được tập nhỏ.
5. Log từng 50 optimizer step và validation cuối mỗi epoch.
6. Checkpoint best và last, môi trường/package, config thực tế.
7. Metrics test, dự đoán có ID, confusion matrix và learning curves.

## Giới hạn diễn giải

- Một seed không đủ kết luận mức cải thiện có ý nghĩa thống kê.
- Trùng giữa train/test ảnh hưởng độ độc lập của đánh giá. Không coi điểm trên split chính thức là đánh giá độc lập hoàn toàn.
- Chưa so sánh preprocessing khác, FGM, dictionary defense hoặc XLM-R.
- EDA không chứng minh segmentation là nguyên nhân model sai.
- `<unk>` được đo riêng, không đồng nhất với phân mảnh BPE.
- Hardware/runtime khác có thể làm kết quả khác nhẹ dù giữ seed.

## Bước tiếp theo sau baseline

Đóng băng checkpoint/config và xây noise testbed với seed, provenance và kiểm tra bảo toàn nhãn. Khi cần tuning robustness, dùng validation nhiễu được tạo độc lập; không dùng test nhiễu để chọn epsilon hoặc cấu hình. Chỉ sau đó mới triển khai FGM và các đối chứng.

## Sửa lỗi trước baseline hợp lệ

Version 1 bị đánh dấu không hợp lệ vì pyjnius `String(str)` làm đổi mã Unicode ngoài BMP. Version 2 dùng `String(byte[], UTF-8)`, kiểm thử roundtrip trên emoji/family ZWJ/chữ Hán supplementary và kiểm tra bảo toàn toàn bộ non-BMP trên từng mẫu khi segmentation. Đổi preprocessing fingerprint để không tái dùng cache cũ. Không dùng điểm của version 1 làm kết quả baseline.
