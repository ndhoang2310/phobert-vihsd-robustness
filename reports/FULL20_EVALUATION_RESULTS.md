# Đánh giá Full20 — đủ 9 checkpoint

Đã xác minh độc lập đủ **99 model/profile evaluations**. Mỗi checkpoint đã tái lập validation predictions trước khi đánh giá test. Không training thêm, không chọn checkpoint bằng test.

## Kết quả chính

Macro-F1 (%), trung bình ± độ lệch chuẩn mẫu qua **3 training seeds**. Với seen/unseen, lấy trung bình 3 noise seeds trong từng training run trước.

| Model | Test gốc | Empirical seen | Empirical unseen |
|---|---:|---:|---:|
| clean | 65.62 ± 0.18 | 62.74 ± 0.66 | 63.54 ± 0.06 |
| fgm | 66.58 ± 0.40 | 63.77 ± 0.54 | 64.69 ± 0.32 |
| augmentation | 65.83 ± 0.30 | 65.04 ± 0.75 | 63.53 ± 0.20 |

## Chênh lệch so với clean

Chênh lệch tính trên cùng training seed. SD qua 3 seed là mô tả độ biến thiên, không phải CI hay kiểm định ý nghĩa thống kê.

| Model | Category | Mean delta (điểm %) | Số seed delta > 0 |
|---|---|---:|---:|
| augmentation | empirical_seen | +2.30 | 3/3 |
| augmentation | empirical_unseen | -0.01 | 1/3 |
| augmentation | original | +0.21 | 3/3 |
| fgm | empirical_seen | +1.02 | 3/3 |
| fgm | empirical_unseen | +1.15 | 3/3 |
| fgm | original | +0.96 | 3/3 |

Nhóm so sánh cùng IDs đủ điều kiện trong cả 6 empirical views: **4,069 câu**, xem shared_eligible.csv. Changed-only scores, coverage và tỷ lệ sửa thực tế nằm trong all_profiles.csv; không coi các câu không bị sửa là thành công robustness.

Test gốc có 809/6680 câu trùng train theo NFC/khoảng trắng/casefold; báo riêng nhóm trùng và không trùng trong original_cohorts.csv. Giữ nguyên official split.

## Giới hạn

Test và benchmark nhiễu đã được xem ở vòng trước; đây là đánh giá chẩn đoán bổ sung. Synthetic toxic labels kế thừa có lọc, chưa được người gán lại. Nguồn human normalization thiên về câu có non-standard words; không tuyên bố mô phỏng toàn bộ phân phối UGC thực. Các giới hạn phụ thuộc nguồn/duplicate đã ghi trong nghiên cứu noise v2 vẫn áp dụng. Chưa có CI về bất định training với 3 seeds; không tuyên bố ý nghĩa thống kê từ bảng mean/SD.

Bảng điểm từng run/profile: full20_evaluation/all_profiles.csv. Trung bình từng run: per_training_seed.csv. Delta theo seed: matched_seed_deltas.csv. F1 từng lớp và nhóm overlap: original_cohorts.csv. Đường stress: comparison.png.
