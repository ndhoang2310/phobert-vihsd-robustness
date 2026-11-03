# Kế hoạch thực thi: dữ liệu thật → noise → EDA → training Kaggle

Trạng thái cuối: research, data audit, generator, 28 kiểm thử, hai training runs chính, đánh giá và xác minh artifacts đã hoàn tất. FGM ε=0,1 version 1 COMPLETE; augmentation GPU version 1 được phục hồi/xác minh bởi CPU version 3 COMPLETE. Bảng kết quả cuối và trạng thái platform nằm trong `../ROBUSTNESS_RUNS.json`, không suy ra thành công chỉ từ việc push notebook.

## 1. Khóa bài toán và đối chứng

- Bài toán ViHSD 3 lớp CLEAN/OFFENSIVE/HATE. Metric chính Macro-F1 vì mất cân bằng lớp; accuracy, từng lớp và confusion matrix là bổ sung.
- Giữ official train/validation/test. Chỉ loại 2 train ID rỗng, còn 24.046 mẫu. Không thêm/xóa test để làm đẹp điểm.
- Baseline PhoBERT hợp lệ là Kaggle version 2, checkpoint epoch 3 chọn trên validation. Giữ revision pretrained, Unicode preprocessing, tokenizer và max length.
- Test gốc là UGC nguyên bản có nhãn thật, vẫn có noise tự nhiên. Test synthetic kế thừa nhãn có lọc; không gọi là bộ nhãn toxic được người xác nhận lại.

## 2. Nguồn thực nghiệm cho noise

- Tải ViLexNorm và Small-LexNormViHSD từ repo tác giả, pin commit và SHA từng file. Lưu nguyên bản, split và source IDs.
- Dùng normalization pairs do người tạo; không tự bịa typo hoặc slang. ViSoLex/dictionary phụ thuộc ViLexNorm, không tính là nguồn xác nhận độc lập.
- ViLexNorm chọn câu chứa NSW, lọc câu ngắn/emoji; không dùng để suy ra tỷ lệ lỗi của toàn bộ mạng xã hội. SmallLex là normalization của ViHSD train, không phải toxic test độc lập.
- TRAIN dùng học mapping và trọng số. Published linguistic test chỉ là diagnostic reference đã được xem khi phát triển, không giả vờ là blind final test.
- Kiểm tra nguồn trùng official evaluation; công khai phần residual dependence của prior 4 loại edit và cross-split duplicates.

Đầu ra: source manifests, `data/noise_v2/bank.json`, `qualification.json`, `source_dependence_audit.json` và `NOISE_RESEARCH_V2.md`.

## 3. Build data code và noise có truy vết

1. Align original/normalized theo anchor, chỉ giữ replace spans ngắn và lexical candidates. Loại broad rewrites, insertion/deletion và protected pronouns/register.
2. Gộp evidence từ hai TRAIN sources; lọc ambiguity có tính cả identity uses. Ngưỡng purity >=0,95 là curation, không phải xác suất giữ nhãn toxic.
3. Bank train yêu cầu >=2 source comment IDs; ghi rõ 2 mappings có context trùng, không gọi source IDs là ngữ cảnh độc lập.
4. Giữ variant quen trong raw train ở bank train. Bank unseen chỉ chứa surface không có trong raw ViHSD train và không có ở bank train.
5. Base weights = số quan sát / canonical opportunities; hiệu chỉnh 4 nhóm heuristic theo SmallLex TRAIN bằng regularized JSD trên 2.000 ViHSD TRAIN comments chọn qua hash. Đây là conditional sampling process, không phải mô hình phổ quát cho mọi lỗi UGC.
6. Sinh noise trước RDRSegmenter; giữ raw, NFC, case và chuỗi non-BMP/emoji. Không sửa các span đã chứa NSW theo detector của bank tương ứng.
7. Mỗi edit ghi offset, source/replacement, mapping ID, source IDs, kind, seed. Mỗi câu có requested/actual rate, changed flag và `label_preservation_verified=False`.

Đầu ra hiện tại: 541 mappings train, 422 unseen. Generator deterministic theo version/ID/seed/partition.

## 4. EDA và cổng chấp nhận

- EDA gốc: rỗng, label distribution, duplicates/conflicts, train–test overlap, độ dài, segmentation, BPE length, UNK và truncation.
- EDA noise: evidence count, mapping ambiguity, coverage, actual rate, loại edit heuristic, held-out lexical-event coverage và ví dụ validation có truy vết.
- Không đánh đồng UNK với segmentation error; đo riêng fragmentation, segmented-unit drift và truncation.
- Kiểm thử trace replay, non-overlap, determinism, partition separation, Unicode, epsilon0 tương đương baseline, FGM gradient isolation và exact restore khi lỗi. 26 tests đã pass.
- Không đặt điều kiện “noise phải làm model giảm điểm”. Không sửa generator theo test F1.
- Chưa có reviewer mới: cổng chất lượng chứng minh tính đúng của code và nguồn quan sát; chưa chứng minh toàn bộ câu sinh tự nhiên hoặc bất biến nhãn.

## 5. Training đầu và ablation

Cấu hình chung: pretrained `vinai/phobert-base` revision khóa; seed 42; 5 epochs; AdamW LR 2e-5, weight decay 0.01; warmup 10%; max length 256; microbatch 8 × accumulation 4; FP16 T4. Cùng 3.760 optimizer steps. Khởi tạo lại từ pretrained cho mỗi model.

- Baseline: clean training đã hoàn tất, không augmentation/FGM.
- FGM epsilon1: clean training, đã hoàn tất pilot; dùng lại checkpoint và đánh giá trên v2 views.
- FGM epsilon0.1: candidate bổ sung đã được quyết định từ validation trước khi thấy test; training clean. Epsilon chọn theo clean validation Macro-F1, hòa chọn nhỏ hơn.
- Augmentation v2: chọn thử 50% comment/epoch; tối đa một observed lexical event/câu đủ điều kiện; chỉ train bank; giữ N và số steps. Không gọi 50% là tỷ lệ noise thực tế.
- FGM triển khai perturbation trên embedding weights, không mô tả như tái lập chính xác activation perturbation của Miyato. Objective `(Lclean + Ladv)/2`; direction riêng microbatch, norm FP32, bỏ padding, restore trước optimizer step.
- Best checkpoint ở từng run chọn bằng clean validation Macro-F1. Test không chọn epoch hoặc tham số.

## 6. Đánh giá và kiểm tra độc lập

Mỗi model đánh giá trên 11 profiles giống nhau: test gốc; 3 empirical seen seeds; 3 empirical unseen seeds; stress 10/20/30/50 với seed cố định. Các mức stress phải đi cùng actual rate và coverage.

- Báo toàn bộ mẫu và changed-only; prediction flips, per-class metrics, tokenizer diagnostics.
- Tính CI paired bootstrap theo nhóm original text, luôn giữ các noise seeds của cùng câu cùng nhóm. CI một training seed không phản ánh training-seed variance.
- Test gốc: thêm cohort NSW marker heuristic và nhóm không exact-overlap raw train; báo class supports, không coi chênh lệch cohort là quan hệ nhân quả.
- Tính lại metrics từ prediction probabilities; đối chiếu đúng 6.680 source IDs/labels; replay từng edit và kiểm tra nguồn/partition/Unicode.
- Kiểm tra best epoch thực sự max validation; checksum best weights; chứng minh best/last tải được từ output GPU.

## 7. Kaggle và bàn giao

- Notebook private, source snapshot + config + provenance có checksum; không đưa token vào notebook/source.
- Chỉ ghi hoàn tất khi đủ training, predictions, verification và artifacts. Nếu lỗi platform xảy ra sau khi lưu toàn bộ training/evaluation do optional reference, giữ nguyên ERROR gốc và dựng notebook CPU recovery có kiểm tra độc lập; không chạy lại GPU hoặc che lỗi.
- Lưu report Markdown + HTML offline, bảng CSV, biểu đồ đã xem trực tiếp, source notebook, checkpoint và lệnh tái lập.
- Run bị timeout/evaluation dở là pilot, không trộn với bảng chính v2. Notebook recovery phải ghi rõ training diễn ra ở version GPU nào.

## 8. Phần xác nhận tiếp theo (ngoài kết luận first-seed)

Cần nhiều training seeds, kiểm chứng ngôn ngữ/nhãn toxic độc lập và nguồn UGC ngoài ViHSD trước khi kết luận tổng quát. Bổ sung model ViSoBERT/XLM-R và augmentation+FGM sau khi first-seed ablation được kiểm tra, không tự gọi chúng là đã chạy.
