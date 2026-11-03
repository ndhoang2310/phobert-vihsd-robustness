# Báo cáo PhoBERT / ViHSD

Bản nguồn nội dung là `BAO_CAO_TONG_HOP.md`. Chỉnh trực tiếp file này.

PDF được biên dịch từ `BAO_CAO_TONG_HOP_PHOBERT_VIHSD.tex` bằng Tectonic (engine XeTeX/LaTeX), hoặc XeLaTeX. Bản hiện tại dùng bố cục article A4, font serif, bảng đen trắng và số trang thông thường. Không tạo PDF bằng HTML hoặc ReportLab.

## Cập nhật nội dung và biên dịch

Tại thư mục gốc dự án:

```bash
.venv/bin/python scripts/render_research_report.py
```

Lệnh này đọc Markdown, sinh lại `.tex`, rồi gọi Tectonic/XeLaTeX. PDF nằm tại `output/pdf/BAO_CAO_TONG_HOP_PHOBERT_VIHSD.pdf`. Log biên dịch và metadata kiểm chứng engine được lưu cùng artifact.

Chỉ sinh LaTeX:

```bash
.venv/bin/python scripts/render_research_report.py --tex-only
```

Muốn chỉnh LaTeX thủ công rồi biên dịch, chạy từ thư mục chứa `.tex` và `assets/`:

```bash
tectonic --keep-logs BAO_CAO_TONG_HOP_PHOBERT_VIHSD.tex
```

Thao tác sinh lại từ Markdown sẽ ghi đè các sửa đổi thủ công trong `.tex`. Giữ nội dung chính trong Markdown để hai bản không lệch nhau.

Máy biên dịch cần font Noto Serif và DejaVu Sans Mono. Tectonic tự tải các gói TeX ở lần chạy đầu; các lần sau dùng cache. Có thể dùng XeLaTeX thay thế. Không chạy lại `scripts/build_research_report.py`: script đó là bộ lắp ráp bản đầu tiên và đã có guard bảo vệ Markdown hiện tại.

## Dữ liệu kèm theo

- `all_profiles.csv`: 99 dòng kết quả đã được xác minh.
- `per_training_seed.csv`: điểm tổng hợp trong từng training seed.
- `matched_seed_deltas.csv`: chênh lệch ghép cùng seed so với clean.
- `original_cohorts.csv`: F1 từng lớp và nhóm trùng/không trùng train.
- `shared_eligible.csv`: so sánh trên cùng 4.069 original IDs đủ điều kiện.
- `training_inventory.csv`: epoch được chọn, thời gian, notebook và SHA256 model.
- `illustrative_cases.json`: hai ví dụ chẩn đoán có source edit traces.
- `VERIFICATION.json`: bằng chứng xác minh 9 model, 99 profile.
- `LATEX_BUILD.json`: engine, lệnh biên dịch và hash MD/TeX/PDF.
- `REPORT_QA.json`: kiểm tra nội dung và bố cục PDF cuối.

Không kèm credential hoặc model weights. Các kết quả và giới hạn nghiên cứu giữ nguyên; thay đổi lần này là nguồn soạn thảo và cách trình bày.
