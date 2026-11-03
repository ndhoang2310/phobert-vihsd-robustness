# Lộ trình đọc code

## Luồng tổng thể

1. `data.py`: tải archive pin SHA256, tạo ID/split, audit và preprocessing.
2. `pipeline.py`: ráp data/audit/EDA, segmentation, tokenizer và provenance.
3. `training.py`: tokenization, DataLoader, model, forward/backward, optimizer/scheduler, validation và checkpoint.
4. `adversarial.py`: FGM perturb trọng số embedding; gradient microbatch, AMP, padding và phục hồi trọng số.
5. `noise.py`: alignment, span selection, sampling, event traces và `NoiseGenerator`.
6. `noise_v2.py`: nguồn in-domain, curation và calibration trọng số v2.
7. `robustness.py`: augmentation callback, mã hóa câu nhiễu và đánh giá shared profiles.
8. `convergence.py`: wrapper training dùng chung cho convergence cũ và Full20; config quyết định early stopping.
9. `full20_eval.py`: tìm checkpoint mount, validation gate, semantic view hash và 11 profile test/model.
10. `scripts/verify_full20_evaluation.py`: tính lại metric, đối chiếu gold labels và replay edit traces.
11. `scripts/summarize_full20_evaluation.py`: tổng hợp đúng training/noise seeds, paired delta, cohorts và biểu đồ.

## Hoàng: pipeline, FGM và tối ưu

Đọc theo thứ tự:

- `configs/full20.json`: ngân sách 20 epoch, ba seed và scheduler horizon.
- `src/vihsd/data.py`: NFC và UTF-8 Java bridge.
- `src/vihsd/training.py`: `predict`, `train`, checkpoint/resume và accumulation.
- `src/vihsd/adversarial.py`: `FGM.direction`, `FGM.attack`, `backward_with_fgm`.
- `src/vihsd/convergence.py`: fresh initialization, callback và completion marker.
- `tests/test_convergence.py`, phần FGM trong `tests/test_robustness.py`.

Cần phân biệt: embedding-weight perturbation với thay mặt chữ; train đủ 20 epoch với chọn epoch cuối; gradient dùng chọn hướng với buffer gradient đã tích lũy.

## Bảo: nguồn nhiễu và augmentation

Đọc theo thứ tự:

- `reports/noise_v2/NOISE_RESEARCH_V2.md` và `source_dependence_audit.json`.
- `src/vihsd/noise.py`: `align_pairs`, `NoiseGenerator.candidate_spans`, `generate`, `generate_view`.
- `src/vihsd/noise_v2.py`: lọc mapping, unseen surface và hiệu chỉnh loại edit.
- `src/vihsd/robustness.py`: `encode_noisy_view` và `augmentation_factory`.
- `scripts/qualify_noise_v2.py`: đối chiếu loại edit/coverage nguồn held-out.
- Các test alignment, purity, generator, emoji và augmentation trong `tests/test_robustness.py`.

Cần phân biệt: annotation normalization với nhãn toxic; requested rate với actual rate/coverage; source IDs với ngữ cảnh độc lập; unseen train surface với unseen pretraining.

## Nguyên: evaluation và đọc kết quả

Đọc theo thứ tự:

- `reports/FULL20_EVALUATION_PROTOCOL.md`.
- `src/vihsd/training.py`: `metrics`, `predict`.
- `src/vihsd/full20_eval.py`: `locate`, validation gate, `view_sha`, `prepare_views`, `run_evaluation`.
- `scripts/verify_full20_evaluation.py`.
- `scripts/summarize_full20_evaluation.py`.
- `tests/test_full20_evaluation.py`.
- `reports/full20_evaluation/`: `all_profiles.csv`, `per_training_seed.csv`, `matched_seed_deltas.csv`, `shared_eligible.csv`, `original_cohorts.csv`.

Cần phân biệt: SD với CI; training seed với noise seed; absolute Macro-F1 với degradation; full test với changed-only/shared-eligible cohorts.

## Notebook nào là kết quả chính?

- Training: `kaggle/full20-{clean,fgm,augmentation}-s{42,123,2026}/train.ipynb`.
- Evaluation: `kaggle/full20-evaluation-*/evaluate.ipynb`.
- Các notebook `convergence-*`, `robustness-*` và baseline là lịch sử riêng. Không ghép điểm của chúng vào bảng Full20.

Notebook frozen nhúng source/bank để lần theo đúng code đã đưa lên Kaggle. Để đọc implementation, ưu tiên `src/vihsd/` vì dễ theo dõi hơn một cell snapshot lớn.

## Phần điều phối và báo cáo

- `scripts/launch_full20.py`, `launch_full20_evaluation.py`: công cụ vận hành dành cho máy/tài khoản gốc; không tự chạy trên máy khác.
- `scripts/build_*`: builder theo campaign, có guard tránh ghi đè lịch sử; cần đúng artifact đầu vào.
- `scripts/render_research_report.py`: Markdown -> TeX -> PDF.
- `scripts/render_presentation_assignment.py`: bản phân công -> TeX -> PDF.
- `scripts/build_research_report.py`: assembler lịch sử có guard; chỉnh Markdown hiện tại trực tiếp.
