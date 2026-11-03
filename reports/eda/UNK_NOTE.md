# Unicode và cách diễn giải `<unk>`

## Bảo toàn Unicode

Đã sửa cầu nối Java sang `String(byte[], UTF-8)`. Kiểm tra toàn bộ train/validation/test: **0 mẫu thay đổi chuỗi ký tự non-BMP qua segmentation**. Điều này bao gồm emoji supplementary. Các kết quả của preprocessing v1 bị đánh dấu không hợp lệ và được tách riêng.

## Mẫu có `<unk>`

`<unk>` không đồng nghĩa segmentation sai; emoji cũng có thể nằm ngoài vocabulary PhoBERT.

| Split | Số mẫu | Mẫu có `<unk>` | Tỷ lệ |
|---|---:|---:|---:|
| test | 6680 | 995 | 14.90% |
| train | 24046 | 3480 | 14.47% |
| validation | 2672 | 377 | 14.11% |

## Phân tầng train theo tín hiệu emoji

| Có tín hiệu emoji | Số mẫu | Mẫu có `<unk>` | Tỷ lệ |
|---|---:|---:|---:|
| False | 20618 | 53 | 0.26% |
| True | 3428 | 3427 | 99.97% |

Đây là tương quan bề mặt, không chứng minh emoji là nguyên nhân duy nhất sinh `<unk>`. Tỷ lệ mẫu có `<unk>` khác tỷ lệ `<unk>` trên tổng token nội dung; phân mảnh BPE được đo riêng.
