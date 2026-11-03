# PhoBERT robustness trên ViHSD

Code và hồ sơ thực nghiệm của dự án **đánh giá và tăng cường độ bền vững của PhoBERT trước nhiễu văn bản mạng xã hội tiếng Việt**. Ba phương pháp: PhoBERT clean, embedding-weight FGM (ε = 0,1), và augmentation bằng biến thể lexical có nguồn human normalization.

## Bắt đầu đọc ở đâu?

1. [Báo cáo tổng hợp (Markdown)](reports/final_report/BAO_CAO_TONG_HOP.md) hoặc [PDF biên dịch từ LaTeX](output/pdf/BAO_CAO_TONG_HOP_PHOBERT_VIHSD.pdf).
2. [Phân công thuyết trình Hoàng / Bảo / Nguyên](reports/final_report/PHAN_CONG_THUYET_TRINH.md) và [PDF](output/pdf/PHAN_CONG_THUYET_TRINH.pdf).
3. [Lộ trình đọc code](docs/CODE_GUIDE.md): module, thuật toán và thứ tự đọc cho từng người.
4. [Cài đặt, kiểm thử và tái chạy](docs/REPRODUCIBILITY.md).

## Kết quả chính đã hoàn tất

**9 training run**, mỗi run đủ **20 epoch**, không early stopping; checkpoint chọn bằng validation Macro-F1. Training seeds: 42, 123, 2026. Sau khi khóa checkpoint, đánh giá mỗi model trên 11 profile, tổng **99 model/profile evaluations** đã được xác minh độc lập.

Macro-F1 (%), mean ± sample SD qua ba training seed:

| Phương pháp | Test gốc | Empirical seen | Empirical unseen |
|---|---:|---:|---:|
| Clean | 65,62 ± 0,18 | 62,74 ± 0,66 | 63,54 ± 0,06 |
| FGM ε=0,1 | 66,58 ± 0,40 | 63,77 ± 0,54 | 64,69 ± 0,32 |
| Augmentation v2 | 65,83 ± 0,30 | 65,04 ± 0,75 | 63,53 ± 0,20 |

Với empirical seen/unseen, trung bình ba noise seed trong từng training run trước, rồi mới tổng hợp qua training seed. Xem [báo cáo đánh giá](reports/FULL20_EVALUATION_RESULTS.md) và [bảng 99 profile](reports/full20_evaluation/all_profiles.csv).

FGM cao hơn clean về điểm gốc và unseen; augmentation mạnh trên seen/stress. Phần lợi thế riêng về mức suy giảm do noise của FGM nhỏ hơn absolute score gain. Test đã được xem trong các vòng trước; synthetic toxic labels được kế thừa, chưa được người gán lại. Ba seed và sample SD chưa chứng minh ý nghĩa thống kê. Chưa chạy XLM-R, dictionary normalization defense, FGM + augmentation, latency comparison hoặc causal segmentation ablation.

## Cấu trúc code

| Thư mục | Nội dung |
|---|---|
| `src/vihsd/` | Data/audit, RDRSegmenter, tokenization, training, FGM, noise, evaluation |
| `configs/` | Cấu hình baseline, convergence cũ và Full20 |
| `tests/` | Kiểm thử pipeline, noise, FGM, resume, full-budget training, validation/hash gates |
| `scripts/` | Build notebook, điều phối các campaign lịch sử, xác minh output và tổng hợp báo cáo |
| `kaggle/full20-*/train.ipynb` | Chín notebook training chứa source và noise-bank snapshot đã khóa |
| `kaggle/full20-evaluation-*/evaluate.ipynb` | Năm notebook đánh giá theo owner, dùng chín checkpoint |
| `notebooks/` | Notebook baseline/pilot lịch sử |
| `reports/final_report/` | Báo cáo MD/TeX, phân công và bảng số liệu tổng hợp |
| `reports/eda/`, `reports/noise_v2/` | EDA và nghiên cứu nguồn nhiễu |
| `reports/full20_evaluation/` | Điểm từng profile/seed, paired delta, cohorts và bằng chứng xác minh |

## Cài đặt và kiểm thử

Python 3.10 trở lên; môi trường phát triển đã dùng Python 3.12. Cài PyTorch phù hợp với CPU/CUDA của máy trước nếu cần.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Các kiểm thử cơ bản và tiny-model training chạy trên CPU. Một số kiểm thử tích hợp bỏ qua khi chưa tải corpus hoặc tài nguyên Java/VnCoreNLP; `skip` không phải bằng chứng rằng integration đã chạy. Training production và evaluation suites hiện yêu cầu CUDA. Data preparation cần Java runtime.

Đọc/audit dữ liệu chính thức, chưa chạy training:

```bash
python -m vihsd.cli doctor
python -m vihsd.cli download
python -m vihsd.cli audit
python -m vihsd.cli eda
```

`python -m vihsd.cli train` mặc định dùng **baseline pilot 5 epoch**, không phải campaign Full20. Xem hướng dẫn tái chạy trước khi launch notebook hoặc chạy các dispatcher.

## Dữ liệu và phiên bản

- ViHSD: [repository tác giả](https://github.com/sonlam1102/vihsd), commit `70fe50c9c94145e3b9be5614a048cbd5b480fa54`; checksum archive kiểm tra trong `src/vihsd/data.py`.
- Official train/validation/test: 24.048 / 2.672 / 6.680. Loại đúng hai train IDs rỗng, còn 24.046 mẫu học; giữ split đánh giá.
- Model: `vinai/phobert-base`, revision `01daacda68afe13d83023d16ec647239e344a1e6`.
- Noise v2: ViLexNorm TRAIN và Small-LexNormViHSD released TRAIN; 541 seen mappings, 422 unseen mappings. Xem [nghiên cứu noise](reports/noise_v2/NOISE_RESEARCH_V2.md).
- Pipeline giữ Unicode/emoji qua UTF-8 Java bridge; NFC/khoảng trắng -> RDRSegmenter -> PhoBERT BPE, max length 256.

Repo chứa code, frozen notebooks, hình và bảng tổng hợp. Corpus gốc, predictions cấp mẫu, model weights, môi trường local và credential không nằm trong Git. Các notebook frozen có bản sao noise bank nghiên cứu; không coi đây là bản phân phối toàn bộ corpus nguồn. Dữ liệu và mô hình bên thứ ba tuân theo điều kiện của nguồn tương ứng.

## Đọc và biên dịch báo cáo

Markdown là nguồn nội dung chính. PDF được biên dịch từ TeX, bố cục đơn giản.

```bash
# Cần Tectonic hoặc XeLaTeX; font Noto Serif và DejaVu Sans Mono
python scripts/render_research_report.py
python scripts/render_presentation_assignment.py
```

Nguồn TeX có sẵn trong `reports/final_report/`; có thể biên dịch trực tiếp từ thư mục đó. Dùng `--tex-only` với renderer báo cáo nếu chỉ muốn sinh TeX.

## Nguồn tham khảo

- [ViHSD, Luu et al., IEA/AIE 2021](https://arxiv.org/abs/2103.11528).
- [PhoBERT, Findings of EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.92/).
- [ViLexNorm, EACL 2024](https://aclanthology.org/2024.eacl-long.85/).
- [Automatic Textual Normalization for Hate Speech Detection / Small-LexNormViHSD](https://arxiv.org/abs/2311.06851).
- [Adversarial Training Methods for Semi-Supervised Text Classification, ICLR 2017](https://arxiv.org/abs/1605.07725).
