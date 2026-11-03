# ViHSD — convergence protocol v1 (2026-10-28)

Mục tiêu: thay phép so sánh ngân sách cố định 5 epoch bằng chính sách dừng thống nhất, đánh giá độ biến thiên qua training seeds. Protocol được khóa trước khi chạy các model mới; không đổi theo điểm test đã xem.

## Thiết kế đã chốt

- Ba nhánh: clean PhoBERT; FGM embedding-weight ε=0,1 đã chọn bằng validation vòng trước; augmentation v2.
- Ba training seeds cố định: **42, 123, 2026**, tổng **9 runs**. Hoàn tất theo block seed để các nhánh có ngân sách đối xứng. Không chọn seed tốt nhất để báo cáo; báo mean/std qua đủ seeds.
- Khởi tạo mới từ pretrained revision `01daacda68afe13d83023d16ec647239e344a1e6`. Không kéo dài checkpoint 5 epoch với scheduler đã gần về 0.
- Tối đa **20 epochs**, early stopping theo **clean validation Macro-F1**, **patience 3**, **min_delta 0**. Ba epoch liên tiếp không vượt mốc tốt nhất thì dừng. Best checkpoint luôn là maximum validation Macro-F1; hòa giữ checkpoint xuất hiện đầu tiên. Không khẳng định early stopping bảo đảm tìm optimum toàn cục.
- Cùng AdamW LR 2e-5, weight decay 0.01, max length 256, microbatch 8 × accumulation 4, FP16 T4. Cùng linear scheduler có horizon 20 epochs, warmup 10% horizon; early stopping không kéo lại hoặc thay đổi scheduler.
- Vì horizon/warmup khác vòng 5 epoch, đây là protocol training mới; không diễn giải mọi chênh lệch với run cũ là tác động riêng của việc thêm epochs.
- Augmentation: cùng bank v2, chọn thử 50% câu/epoch và tối đa một observed lexical event/câu đủ điều kiện; seed của variant/selection phụ thuộc training seed. Không tăng N hoặc optimizer-update budget mỗi epoch.
- Giữ official splits, raw dữ liệu và 2 train exclusions như vòng trước. Giữ Unicode/emoji pipeline v2.

## Noise và test

Bank **đóng băng nguyên bản**, SHA-256 `d0b9de546827c3b8d35f98a8a72568eb55005846fe68af28d1b6b07c87d18930`, 541 mappings train/422 unseen. Không fit lại weights, thêm mapping hay đổi source sau khi xem test.

Các job training **không tính predictions/test metrics**. Mỗi epoch lưu validation probabilities + IDs để tính lại độc lập. Chỉ đánh giá selected checkpoints khi tất cả condition/seed pairs đã chốt. Dùng lại v2 views và seeds noise 201/202/203, báo theo training seeds, không nhầm noise-seed variation với training-seed variation.

Test ViHSD đã được xem ở vòng trước, nên vòng này vẫn là đánh giá bổ sung trên benchmark có sẵn, không gọi là blind confirmatory test. Giới hạn synthetic labels, source-selection bias và cross-split duplicates vẫn giữ như báo cáo v2.

## Resume và bằng chứng

Checkpoint last.pt lưu model/optimizer/scheduler/scaler/RNG, config horizon, next epoch/batch, history, early-stopping counters và partial epoch loss. Chỉ resume cùng config và source; luôn mang best checkpoint đã chọn đi kèm. Run đã dừng do patience không được chạy thêm epoch khi resume. Checkpoint 5 epoch không được dùng để resume protocol 20 epoch.

Kiểm thử có full tiny-model training/backward trên CPU: dừng đúng patience, giữ earliest best khi hòa, save validation probabilities, resume stopped run không train thêm, interrupted resume khớp weights/scheduler và counters của uninterrupted run. Full production training vẫn yêu cầu CUDA.

Mỗi run phải có TRAINING_COMPLETE.json, đủ validation predictions, history/selection/config/environment, best weights và last.pt. Tính lại validation Macro-F1 từ đúng IDs/labels; mô phỏng lại stopping state để xác minh epoch dừng.

## Thực thi Kaggle

Mỗi notebook private, self-contained source + bank snapshot, không phụ thuộc output checkpoint cũ hoặc gắn credential vào notebook. Ba nhánh dùng cùng model/data revisions và cùng runtime policy. Tối đa 2 GPU sessions trên account được chọn qua bridge. Bridge không cho gọi nhóm quota/auth/config: không tự mở SDK session để lách giới hạn này; Kaggle server xác nhận khả năng submit và nếu báo thiếu quota thì queue giữ WAITING_QUOTA. Không tự thay đổi billing hoặc cấu hình mở khóa vault.

Nếu thiếu credential hoặc quota, ghi rõ PREPARED/WAITING_AUTH/WAITING_QUOTA và không gọi là đã training. Mọi lệnh Kaggle có credential chạy qua `/home/hoang/.local/bin/kaggle-account`; mỗi invocation mở hộp mật khẩu riêng, người dùng nhập trực tiếp. Không đọc/export key, không dùng credential directories cũ hoặc tạo account manager mới.
