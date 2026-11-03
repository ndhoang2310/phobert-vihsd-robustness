# Protocol chặng 2: nhiễu có căn cứ dữ liệu thật → training

## Quyết định trước thực nghiệm

Baseline khóa: notebook Kaggle version 2, seed 42, best epoch 3. Không thay preprocessing UTF-8 hoặc split ViHSD. Chưa có người gán nhãn mới; dùng cặp normalization có human annotation và nhãn toxic ViHSD sẵn có. Không tuyên bố synthetic pairs đã được human xác nhận giữ nhãn toxic.

## Nguồn nghiên cứu

- ViLexNorm (EACL 2024): https://aclanthology.org/2024.eacl-long.85/; nguồn https://github.com/ngxtnhi/ViLexNorm, CC BY-NC-SA 4.0. Cặp original/normalized là nguồn chính. Dữ liệu đã lọc câu ngắn và loại emoji; mọi phân phối chỉ có điều kiện trên corpus này, không coi là tỷ lệ UGC phổ quát.
- ViSoLex (COLING 2025): https://aclanthology.org/2025.coling-demos.18/; từ điển https://github.com/AnhHoang0529/vn-nsw-dictionary được trích từ ViLexNorm, không phải bằng chứng độc lập. Không dùng toàn bộ từ điển để vô tình đưa variant test vào train.
- ViSoBERT (EMNLP 2023): https://aclanthology.org/2023.emnlp-main.315/; tham khảo đặc trưng UGC/emoji, không coi UNK là bằng chứng segmentation sai.
- Miyato et al. (ICLR 2017): https://research.google/pubs/adversarial-training-methods-for-semi-supervised-text-classification/; nghiên cứu embedding adversarial perturbations. Code đợt này dùng FGM trên embedding weights, phải mô tả đúng biến thể triển khai.

## Data và nguồn biến thể

1. Pin source revision/checksum; kiểm tra rỗng, duplicates, overlap ViLexNorm–ViHSD.
2. Chỉ fit mapping/frequency từ ViLexNorm train, không fit bằng ViHSD validation/test hoặc ViLexNorm test.
3. Align original→normalized bằng các anchor không đổi; chỉ giữ thay thế ngắn có thể truy ngược về cặp có human annotation. Không sinh biến thể mới chưa từng xuất hiện.
4. Lọc mapping nhập nhằng, liên quan URL/entity/number, thay đổi phủ định hoặc thay thế từ đồng nghĩa có thể đổi mức độc hại. Chỉ dùng biến thể hình thức/chính tả/viết tắt có độ tin cậy cao.
5. Giữ variant surfaces riêng theo hash (20%) làm unseen-variant bank; không xuất hiện trong augmentation training bank.
6. Loại các cặp nguồn trùng ViHSD validation/test khỏi nguồn học mapping. Đây là chống nhiễm nguồn, không dùng nhãn/điểm test để điều chỉnh quy tắc.

## Noise testbed

- Natural cohorts: ViHSD gốc không chỉnh chữ, phân nhóm theo detector NSW đã khóa, độ dài và nhãn. Detector là heuristic; không coi noise density là gold.
- Empirical noise: sample biến thể thực đã quan sát, trọng số theo tần suất trong nguồn train; không chia đều loại nhiễu tùy ý.
- Controlled stress: mức thêm nhiễu 10/20/30/50% trên số đơn vị whitespace nguồn; ghi requested/actual/coverage, không ép đủ mức nếu không có mapping phù hợp. Tách khỏi empirical profile.
- Sinh trước segmentation, giữ emoji/Unicode, không thay đổi raw. Mỗi mẫu có edit offsets, mapping ID, source sentence ID, seed và checksum.
- Seed train augmentation 4242; seed validation noise 101/102/103; seed test noise 201/202/203. Không tuning bằng test.
- Synthetic label preservation là giả định có lọc, không phải chứng nhận human. Dùng natural cohorts có nhãn thật làm đối chứng bắt buộc.

## Cổng chất lượng

- Seed determinism, offsets replay, source evidence, non-overlap edits và Unicode phải pass 100%.
- Variant bank train/unseen không trùng surface sau NFC/casefold.
- Đo coverage, actual rate, lệch loại edit, edit distance, density và tokenization; báo rõ domain/selection bias và vùng thiếu coverage.
- Kiểm tra trên held-out ViLexNorm dev/test cho khả năng bao phủ biến thể thật. Không tự bổ sung quy tắc để làm F1 model giảm.
- Không yêu cầu model phải giảm điểm mới coi testbed hợp lệ.

## Training trên Kaggle

M0: baseline đã khóa; M1: clean + FGM; M2: empirical augmentation. M3 (augmentation + FGM) là bước xác nhận sau nếu quota cho phép.

- Cùng pretrained revision và seed 42, LR 2e-5, 5 epoch, max length 256, effective batch 32, FP16 trên T4.
- Khởi tạo lại từ pretrained cho mỗi run, không lấy model baseline đã fine-tune làm khởi đầu.
- API Kaggle tại thời điểm triển khai báo quota tổng 21.600 giây, đã dùng khoảng 11.831 giây. Vì vậy epsilon **1.0 được chốt trước** cho first-seed FGM; sweep {0.1,0.5,1.0,2.0} dời sang đợt xác nhận, không gọi epsilon 1.0 là đã tối ưu. Direction lấy bằng autograd.grad riêng cho microbatch, không dùng gradient đã cộng dồn. Norm FP32; không tấn công padding/position/classifier; restore chính xác trong finally.
- Objective FGM: (L_clean + L_adv)/2, rồi chia gradient accumulation; unscale/clip/step một lần mỗi optimizer update. Lambda=0 đi đúng nhánh baseline.
- Mỗi run chọn checkpoint bằng clean validation Macro-F1 như M0. Không chọn tham số theo test. Khi có quota để sweep sau này, dùng trung bình clean/empirical validation Macro-F1 với ngân sách giảm clean <=1 điểm phần trăm; đây là thiết kế chọn model tương lai, không phải bước đã chạy.
- M2: 50% comment training được chọn để thử augmentation (đây là thiết kế training, không phải tỷ lệ thực tế của UGC); tối đa một bản/comment/epoch, chỉ dùng bank train; không tăng số optimizer steps. Không sửa lại mẫu đã là NSW tại span đó. Nếu chạy M3, dùng cùng epsilon chốt trước của M1.
- Đợt này hoàn tất first-seed ablation và epsilon selection. Multi-training-seed và XLM-R/ViSoBERT là chặng xác nhận tiếp theo, không tuyên bố thống kê từ một seed.

## Test và đầu ra

Chỉ evaluate test sau khi khóa generator, chốt epsilon và train xong model tương ứng. Cùng test variants cho mọi model. Báo clean, empirical seen/unseen, controlled stress, natural cohorts và nhóm test không trùng train. Metrics: Macro-F1, từng lớp, delta so baseline, prediction flip, UNK/fragmentation/truncation. CI bootstrap phải resample theo original sample ID (không coi 3 biến thể của cùng câu là 3 mẫu độc lập).

Kaggle phải COMPLETE, có checkpoint best/last, source/config/data provenance và prediction IDs. Tính lại metrics độc lập, kiểm tra checkpoint tải được. Thành công kỹ thuật không đồng nghĩa FGM chắc chắn cải thiện.

## Giới hạn nguồn và audit

Empirical-event profile thêm một sự kiện lexical đã quan sát trên mỗi câu đủ điều kiện. Đây là profile biến thể dựa trên corpus, không mô phỏng đầy đủ phân phối số lỗi mỗi comment của mạng xã hội. Variant sampling dùng số lần quan sát làm trọng số; tỷ lệ chọn câu training 50% là thiết kế tối ưu hóa. Corpus ViLexNorm được chọn theo câu có NSW nên không dùng nó ước lượng tỷ lệ UGC có lỗi.

## Candidate thứ hai, chốt bằng validation trước khi xem test

Sau hai epoch, FGM epsilon 1.0 có validation Macro-F1 0.5952, thấp hơn baseline. Quota API còn khoảng 8.108 giây tại lần kiểm tra tiếp theo, đủ cho một run bổ sung. Vì vậy thêm candidate epsilon **0.1**; không chạy grid 0.5/2.0 hay multi-seed. Chọn giữa 0.1 và 1.0 bằng **clean validation Macro-F1 cao nhất**, tie ưu tiên epsilon nhỏ. Mỗi run vẫn chọn checkpoint theo clean validation Macro-F1. Quyết định này được ghi trước mọi kết quả test nhiễu; không dùng test để chọn epsilon. M3 combo tiếp tục để chặng sau.


## Frozen primary v2 protocol (2026-10-26)

The pilot v1 corpus bank had limited coverage in the subsequently located human-normalized Small-LexNormViHSD resource. Version 2 is a language/data revision, not tuning against classifier test F1. Both pilot checkpoints are retained as references and re-evaluated on the SAME frozen v2 views. No v1/v2 score mixing.

V2 combines ViLexNorm TRAIN and Small-LexNormViHSD released TRAIN (1962 comments). The latter comes from ViHSD train, not an independent toxic test set. Source comments overlapping official evaluation text under NFC/casefold/whitespace-free matching are excluded. Identity uses enter ambiguity filtering; train mappings require two distinct source comments and >=0.95 observed target purity. This curation threshold is not semantic-label confidence. Base weights are conditional variant counts / canonical opportunities. Four edit-family multipliers fit Small-LexNorm TRAIN event proportions using 2000 hash-selected ViHSD TRAIN comments and regularized JSD optimization, log bounds [-3,3]. No held-out frequencies or toxic labels fit the weights. The released 219-comment linguistic test was inspected during development: it is a diagnostic reference, NOT a blind final test.

Frozen bank: 541 train and 422 unseen mappings; unseen surfaces absent from raw ViHSD train. No synthetic toxic-label reannotation is available. Coverage and distribution agreement are limited linguistic evidence, not proof of complete naturalness or label invariance.

Primary runs: augmentation-v2 (50% comments selected per epoch; one eligible observed lexical edit), clean-trained FGM epsilon0.1. Attach clean-trained FGM epsilon1 and pilot augmentation checkpoints to augmentation-v2 for same-view evaluation. Preserve baseline version2. Run original, 3 empirical seen, 3 unseen, and 4 controlled-stress profiles per model (11 profiles). Skip family-specific extra profiles to conserve GPU; record UNK, fragmentation, segmentation and truncation diagnostics for every v2 view. All runs use 5 epochs/seed42 and clean-validation checkpoint selection. FGM epsilon chosen by clean validation only, ties smaller epsilon.

Timeouts extended after pilot augmentation completed all training epochs but its evaluation exceeded the 1800-second runtime. Pilot checkpoint is available; lack of completion marker is retained in the audit.

Kaggle rejects the timeout-cancelled pilot augmentation as a notebook source. It is excluded from primary v2 references; archived pilot outputs remain diagnostic. FGM epsilon1 reference is included. This execution dependency change does not change the v2 bank or primary model configuration.

### Post-freeze data dependence audit

No exact original-text duplicates across the two released source TRAIN corpora. Two accepted mappings have multiple source IDs but one unique original/normalized text pair; distinct source IDs do not imply independent contexts. The 71 excluded evaluation-overlap SmallLex rows are excluded from mapping evidence. The four-family calibration prior uses the ENTIRE released SmallLex TRAIN, including these natural cross-split duplicates; this is a residual source-dependence limitation. No evaluation toxic labels or predictions fit the prior. Preserve this disclosure rather than silently changing an already-running frozen experiment.
