# Phân công thuyết trình: Hoàng, Bảo, Nguyên

Dựa trên `BAO_CAO_TONG_HOP.md`, kết quả chính là Full20 và đánh giá đủ 9 checkpoint/99 model-profile evaluations.

## 1. Nguyên tắc chia và thứ tự trình bày

- Thứ tự: **Hoàng -> Bảo -> Nguyên**.
- Mỗi người đảm nhận **6 phần nội dung**, có thể triển khai thành 6 slide chính.
- Thời lượng đề xuất: khoảng **7-8 phút/người**, tổng 21-24 phút, chưa tính hỏi đáp. Nếu thời lượng khác, giữ tỷ lệ một phần ba cho mỗi người.
- Cân bằng theo độ sâu kỹ thuật: Hoàng giải thích mô hình và tối ưu; Bảo giải thích thuật toán sinh nhiễu và augmentation; Nguyên giải thích thiết kế đánh giá, xác minh và phân tích kết quả.
- Không cần đọc tuần tự mọi mục trong báo cáo. Dùng số liệu Full20 làm kết quả chính; lịch sử pilot chỉ nhắc khi giải thích quyết định kỹ thuật.

| Người | Mạch nội dung đảm nhận | Phần kỹ thuật chính | Slide dự kiến |
|---|---|---|---|
| Hoàng | Bài toán, dữ liệu, pipeline PhoBERT, FGM và training | Segmentation/BPE, gradient embedding, AMP/accumulation, scheduler, checkpoint selection | 1-6 |
| Bảo | Nguồn nhiễu, xây bank, thuật toán generator và augmentation | Alignment/curation, conditional sampling, hiệu chỉnh loại edit bằng JSD, seed và edit trace | 7-12 |
| Nguyên | Thiết kế đánh giá, kiểm chứng, kết quả và kết luận | Validation gate, semantic hash/trace replay, tổng hợp training/noise seeds, paired delta và degradation | 13-18 |

## 2. Hoàng: mô hình và quá trình huấn luyện

### Phần 1 / slide 1. Bài toán và mục tiêu

Nội dung cần nói:

- Phân loại bình luận tiếng Việt thành CLEAN, OFFENSIVE, HATE.
- Nhiễu UGC có thể làm thay đổi biểu diễn đầu vào và dự đoán của PhoBERT.
- Ba nhánh so sánh: PhoBERT clean, PhoBERT + FGM, PhoBERT + augmentation.
- Mục tiêu: đo hiệu năng gốc và khả năng chịu nhiễu; không mặc định một phương pháp thắng mọi loại nhiễu.

### Phần 2 / slide 2. ViHSD và các phát hiện EDA dẫn tới quyết định kỹ thuật

Nội dung cần nói:

- Split gốc: 24.048 train, 2.672 validation, 6.680 test; loại đúng 2 mẫu train rỗng, còn 24.046 mẫu để học.
- CLEAN chiếm khoảng 82,69% train; always-CLEAN đạt 83,05% accuracy test nhưng chỉ 30,25% Macro-F1.
- Vì sao chọn Macro-F1 làm chỉ số chính và vẫn báo thêm accuracy/F1 từng lớp.
- Giữ official split; có duplicate và nhãn xung đột, không tự sửa nhãn hoặc bỏ mẫu test.

Hình/bảng nên dùng: phân bố nhãn và bảng số mẫu theo split.

### Phần 3 / slide 3. Pipeline tiền xử lý và biểu diễn của PhoBERT

Nội dung cần nói:

- Luồng raw text -> NFC/khoảng trắng -> RDRSegmenter -> PhoBERT BPE -> classifier.
- Giữ emoji, case, teencode và ký tự kéo dài; không dùng dictionary normalization trong ba nhánh hiện tại.
- Lỗi cầu nối Java từng làm thay đổi non-BMP; sửa bằng đường truyền UTF-8 và kiểm tra bảo toàn ký tự.
- Max length 256; test chỉ có 1/6.680 câu bị cắt.
- UNK không đồng nghĩa segmentation lỗi; cơ chế segmentation gây sai nhãn chưa được kiểm chứng nhân quả.

### Phần 4 / slide 4. Cơ chế FGM và implementation

Nội dung cần nói:

- FGM perturb ma trận trọng số embedding, khác với việc sửa mặt chữ của augmentation.
- Tính gradient clean loss theo embedding; chuẩn hóa toàn ma trận bằng norm Frobenius; nhân epsilon.
- Chạy adversarial forward với embedding đã perturb; loss tối ưu là trung bình clean và adversarial loss.
- Phục hồi embedding trước optimizer update; padding gradient bằng 0.
- Gradient dùng chọn hướng không lấy từ buffer tích lũy; bỏ scale AMP trước khi chuẩn hóa.
- Epsilon 0,1 được chọn bằng validation sau pilot hai mức 1,0 và 0,1; chưa phải sweep đầy đủ.

Công thức cần giải thích:

$$
g=\nabla_E\mathcal L_{\mathrm{clean}},\qquad
r_{\mathrm{adv}}=\varepsilon\frac{g}{\lVert g\rVert_F},\qquad
\mathcal L_{\mathrm{total}}=\frac{\mathcal L_{\mathrm{clean}}+\mathcal L_{\mathrm{adv}}}{2}.
$$

### Phần 5 / slide 5. Thiết kế training Full20

Nội dung cần nói:

- 3 phương pháp × 3 training seed 42/123/2026 = 9 run.
- Mọi run khởi tạo mới từ cùng pretrained revision, chạy đủ 20 epoch, tắt early stopping.
- AdamW, LR 2e-5, weight decay 0,01; microbatch 8 × accumulation 4, effective batch 32.
- Linear scheduler với horizon 20 epoch và warmup 10%; 15.040 optimizer update/run.
- Validation mỗi epoch; giữ checkpoint có validation Macro-F1 cao nhất, hòa chọn epoch sớm nhất.
- Test không tham gia lựa chọn checkpoint.

### Phần 6 / slide 6. Kết quả training, checkpoint và chi phí

Nội dung cần nói:

- Phân biệt train đủ 20 epoch với việc chọn checkpoint epoch 20.
- FGM seed 123 có best validation tại epoch 15; các run khác đạt best ở epoch 4/6/7/9.
- Validation mean best-checkpoint: clean 69,00; FGM 70,08; augmentation 68,67 (%).
- Thời gian trung bình/run: clean 55,7 phút; FGM 112,4 phút; augmentation 69,3 phút.
- FGM tốn khoảng 2,02 lần thời gian clean ở thiết lập hiện tại; đây là thời gian train, chưa phải inference latency.
- Kết quả validation chỉ dùng chọn checkpoint; phần sau mới kiểm tra robustness trên test.

Hình nên dùng: đường validation qua 20 epoch và bảng thời gian training.

**Chuyển nội dung cho Bảo:** các checkpoint đã được huấn luyện; cần giải thích bộ nhiễu được xây như thế nào và augmentation đã đưa loại biến thể nào vào train.

Tham chiếu báo cáo: mục 1-4, phần FGM trong mục 7, mục 8-9 và phần thời gian ở mục 16.

## 3. Bảo: dữ liệu nhiễu và augmentation

### Phần 7 / slide 7. Vì sao cần nguồn noise có bằng chứng

Nội dung cần nói:

- Hạn chế của tự nghĩ rule typo/teencode: khó chứng minh biến thể đã xuất hiện trong dữ liệu thật.
- Khai thác chiều ngược của cặp normalization: canonical -> biến thể không chuẩn quan sát được.
- Hai nguồn có human normalization annotation: ViLexNorm và Small-LexNormViHSD.
- Train nguồn: 8.372 và 1.962 cặp; SmallLex có quan hệ với ViHSD train.
- Annotation normalization không phải annotation giữ nguyên toxic label sau khi áp vào ngữ cảnh mới.

### Phần 8 / slide 8. Alignment, curation và chia bank

Nội dung cần nói:

- Trích cặp canonical/variant từ alignment; mỗi mapping có source IDs và thông tin loại edit.
- Lọc mapping mơ hồ; mức nhất quán tối thiểu 0,95; bằng chứng ít nhất 2 source comment IDs cho train mappings.
- Bank v2 có 541 mapping train và 422 mapping unseen.
- Unseen yêu cầu surface vắng trong train bank và raw ViHSD train; không khẳng định chưa có trong pretraining.
- Loại 71 source pairs trùng evaluation khỏi mapping evidence.
- Giới hạn còn lại: source IDs không luôn là ngữ cảnh độc lập; prior loại edit vẫn dùng full SmallLex TRAIN.

### Phần 9 / slide 9. Lấy mẫu có trọng số và hiệu chỉnh loại edit

Nội dung cần nói:

- Trọng số ban đầu dựa trên số lần quan sát cặp chia số cơ hội xuất hiện canonical.
- Nhóm bốn loại edit heuristic: abbreviation, diacritics, elongation, spelling_variant.
- Fit hệ số loại edit bằng prior SmallLex TRAIN và 2.000 ViHSD TRAIN comments chọn bằng hash.
- Mục tiêu JSD có regularization; không fit bằng toxic test labels hoặc test F1.
- Vì corpus nguồn chọn NSW, đây là phân phối có điều kiện trên mapping đủ điều kiện, không phải phân phối lỗi toàn bộ UGC.

### Phần 10 / slide 10. Thuật toán NoiseGenerator

Nội dung cần nói:

- Tìm canonical spans hợp lệ; tránh chồng lấn và tránh một số span URL/mention.
- Chọn span và variant theo trọng số, với seed gắn sample ID để tái lập được.
- Empirical mode áp dụng tối đa một lexical event; event có thể gồm nhiều đơn vị.
- Stress mode dùng mức yêu cầu 10/20/30/50% đơn vị khoảng trắng, làm tròn lên và bị giới hạn bởi số span hợp lệ.
- Lưu offset, original/replacement, mapping ID và source IDs để replay.
- Giữ chuỗi non-BMP; câu không có mapping phù hợp được giữ nguyên.

Nên dùng một ví dụ có edit trace từ artifact, không tự tạo thêm ví dụ rồi gọi là dữ liệu đã quan sát.

### Phần 11 / slide 11. Augmentation trong training

Nội dung cần nói:

- Thử chọn 50% câu mỗi epoch bằng hash; chỉ sửa nếu câu có span đủ điều kiện.
- Tối đa một event/câu được áp dụng; không tăng số mẫu hoặc số optimizer update mỗi epoch.
- Seed chọn câu và biến thể phụ thuộc training seed.
- Chỉ bank train đi vào augmentation; bank unseen được giữ cho đánh giá.
- Phân biệt tỷ lệ chọn 50% trong augmentation với mức stress yêu cầu 50% khi đánh giá.
- Đối chiếu với FGM: augmentation thay mặt chữ, FGM perturb trọng số embedding; không có nhánh kết hợp hai phương pháp trong đợt này.

### Phần 12 / slide 12. Chất lượng benchmark và cách đọc mức noise

Nội dung cần nói:

- Coverage lexical events trên in-domain held-out tăng từ 22,76% lên 32,79%; JSD loại edit giảm từ 0,0794 xuống 0,0078.
- Coverage còn hạn chế; các số này không chứng minh mọi câu sinh ra tự nhiên hoặc giữ toxic label.
- Empirical seen sửa 78,98% câu; unseen sửa 63,47% câu.
- Stress yêu cầu 50% chỉ đạt mean actual rate 24,59%; yêu cầu 10% cho 13,15% do làm tròn và event nhiều đơn vị.
- Phân biệt coverage, actual edit rate và requested rate.
- Toxic labels của câu tổng hợp vẫn là nhãn kế thừa, chưa được người gán lại.

Hình/bảng nên dùng: so sánh nguồn v1/v2 và bảng requested/actual rate.

**Chuyển nội dung cho Nguyên:** đã có benchmark được khóa; cần đánh giá mọi checkpoint trên đúng cùng các câu nhiễu và tổng hợp kết quả với đúng đơn vị seed.

Tham chiếu báo cáo: mục 5-6, phần augmentation trong mục 7 và định nghĩa coverage/rate ở mục 12.

## 4. Nguyên: đánh giá, phân tích kết quả và kết luận

### Phần 13 / slide 13. Thiết kế đánh giá và xác minh

Nội dung cần nói:

- Mỗi checkpoint được đánh giá trên original, 3 seen noise seeds, 3 unseen noise seeds và 4 stress levels: 11 profile/model, tổng 99.
- Chỉ dùng checkpoint đã khóa bằng validation; không chọn model hoặc chỉnh tham số theo test.
- Trước test inference, tái lập validation predictions và kiểm tra tokenized inputs; cả 9 checkpoint có max probability difference bằng 0.
- Semantic hashes bảo đảm mọi model nhận cùng IDs, labels, text nhiễu và edit traces.
- Sau download, tính lại metrics từ probabilities, kiểm tra labels/IDs và replay edits; đủ 99 profile vượt xác minh.

### Phần 14 / slide 14. Cách tính chỉ số và tổng hợp seed

Nội dung cần nói:

- Macro-F1 là trung bình F1 của CLEAN/OFFENSIVE/HATE, không phải accuracy hoặc weighted F1.
- Training seed và noise seed là hai nguồn biến thiên khác nhau.
- Với seen/unseen: tính F1 từng view -> trung bình 3 noise seed trong mỗi training run -> mean/sample SD qua 3 training seed.
- Paired delta ghép đúng training seed với clean; không chọn seed tốt nhất để báo cáo.
- SD không phải confidence interval; 3 training seeds và 9 tổ hợp noise/training không tương đương 9 model độc lập.

### Phần 15 / slide 15. Kết quả chính

Nội dung cần nói:

| Phương pháp | Test gốc | Seen | Unseen |
|---|---:|---:|---:|
| Clean | 65,62 | 62,74 | 63,54 |
| FGM | 66,58 | 63,77 | 64,69 |
| Augmentation | 65,83 | 65,04 | 63,53 |

- Bảng là mean Macro-F1 (%); slide chính nên lấy bảng mean ± SD đầy đủ trong báo cáo.
- FGM hơn clean trên cả 3 seed ở original/seen/unseen; mean delta +0,96/+1,02/+1,15 điểm.
- Augmentation mạnh nhất trên seen: +2,30 điểm; unseen gần như không tăng: khoảng -0,01 điểm.
- Không đánh đồng kết quả trên biến thể quen thuộc với tổng quát hóa sang biến thể mới.

### Phần 16 / slide 16. Stress và phân biệt absolute score với degradation

Nội dung cần nói:

- Ở stress yêu cầu 50%: clean 58,97; FGM 59,99; augmentation 62,68 Macro-F1 (%).
- Định nghĩa degradation: điểm original của chính model trừ điểm noisy.
- FGM mất 1,89 điểm trên unseen so với 2,08 của clean; phần chênh lệch degradation khoảng 0,20 điểm.
- Absolute unseen gain +1,15 điểm của FGM có phần lớn lợi thế đã xuất hiện ở original (+0,96 điểm).
- Augmentation giảm mất điểm seen còn 0,79 so với 2,88 của clean; ở unseen lại mất khoảng 2,30 điểm.
- Đây là phân rã mô tả, không phải kiểm chứng nhân quả cơ chế.

Hình nên dùng: biểu đồ kết quả chính/stress và biểu đồ degradation.

### Phần 17 / slide 17. Phân tích lỗi và kiểm tra độ nhạy

Nội dung cần nói trên slide chính:

- OFFENSIVE vẫn là lớp khó: FGM có F1 khoảng 43,85%, thấp hơn nhiều CLEAN 94,28% và HATE 61,60%.
- Dùng confusion matrix để giải thích nhóm nhầm lẫn, không chỉ đọc Macro-F1 tổng.
- Minh họa một case có sample ID: `test:000213`, ngon -> ngol; clean dự đoán OFFENSIVE, FGM và augmentation dự đoán CLEAN theo gold kế thừa.
- Case được chọn để minh họa; không dùng một ví dụ để ước lượng tỷ lệ sửa lỗi hoặc chứng minh RDR gây lỗi.

Đưa vào slide dự phòng nếu không đủ thời gian:

- Nhóm 5.871 test IDs không trùng train: FGM vẫn hơn clean khoảng 0,91 điểm.
- Cùng 4.069 IDs đủ điều kiện ở cả 6 empirical views: FGM cao hơn clean ở unseen khoảng 1,20 điểm; augmentation thấp hơn khoảng 0,11 điểm.
- Nhóm này chọn theo eligibility; không đại diện toàn UGC và không thay thế test độc lập mới.

### Phần 18 / slide 18. Trả lời RQ, giới hạn và hướng tiếp theo

Nội dung cần nói:

- RQ1: có baseline và đo được suy giảm dưới nhiễu; chưa exact-reproduce paper gốc hoặc so với XLM-R.
- RQ2: FGM tăng absolute score trong benchmark hiện tại; chưa đủ kết luận ý nghĩa thống kê hoặc cơ chế nhân quả.
- RQ3: FGM cải thiện điểm gốc; augmentation mạnh seen/stress; chưa có dictionary-defense baseline và latency comparison.
- Giới hạn chính: test đã được xem ở vòng trước, synthetic labels chưa được người gán lại, duplicate/source dependence, chỉ 3 training seeds.
- Bước tiếp: đối chứng còn thiếu, test độc lập có gold toxic labels, kiểm chứng giữ nhãn, ablation và latency.
- Kết luận giữ đúng phạm vi: FGM tốt hơn về điểm gốc/unseen trong ba nhánh đã chạy; augmentation mạnh biến thể seen/stress, không có phương pháp thắng mọi dạng noise.

Tham chiếu báo cáo: mục 10-15, phần kiểm thử ở mục 16, mục 17-18 và phụ lục A.

## 5. Phân công phần hỏi đáp

| Người trả lời chính | Nhóm câu hỏi cần chuẩn bị |
|---|---|
| Hoàng | Vì sao PhoBERT cần segmentation; lỗi Unicode; vì sao train 20 epoch nhưng chọn checkpoint trước đó; FGM perturb ở đâu; norm, AMP, accumulation, phục hồi embedding; chi phí train |
| Bảo | Noise dựa trên dữ liệu thật đến mức nào; vì sao dùng hai nguồn; alignment/curation; ý nghĩa 0,95; JSD; seen/unseen; 71 source overlaps; coverage và actual rate; giữ nhãn |
| Nguyên | Vì sao Macro-F1; ghép training/noise seeds; mean/SD; validation gate/hash/replay; độ tin cậy kết quả; absolute score/degradation; overlap/cohorts; giới hạn thống kê và đối chứng còn thiếu |

## 6. Những điểm cần thống nhất trước khi ghép slide

- Cùng dùng kết quả Full20, không ghép điểm pilot 5 epoch hoặc convergence cũ vào bảng chính.
- Cùng dùng đơn vị Macro-F1 (%) và gọi chênh lệch là điểm phần trăm.
- Phân biệt tắt early stopping với chọn epoch cuối: mọi run train đủ 20, checkpoint chọn bằng validation.
- Phân biệt “test gốc” với văn bản hoàn toàn chuẩn; dữ liệu gốc vẫn có UGC tự nhiên.
- Phân biệt annotation normalization nguồn với gold toxic label của câu được sinh mới.
- Không khẳng định đã làm XLM-R, normalization defense, FGM + augmentation, latency hoặc annotation giữ nhãn.
- Không dành slide chính cho chi tiết tài khoản Kaggle, lỗi optional mount hoặc toàn bộ hash; để phụ lục khi cần truy vết.
- Mỗi người chuẩn bị hình/bảng cho phần mình và nắm một giới hạn quan trọng của chính phần đó.
