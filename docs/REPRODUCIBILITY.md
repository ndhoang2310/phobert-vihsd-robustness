# Cài đặt, kiểm thử và tái chạy

## 1. Đọc và kiểm thử mà không dùng GPU

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Các test unit/tiny-model không launch Kaggle. Một số integration test cần archive ViHSD hoặc cache VnCoreNLP và sẽ `skip` nếu chưa có. Không coi các test bị skip là integration đã được xác nhận trên máy mới.

Trên môi trường nghiên cứu đầy đủ, vòng evaluation đã ghi 36 test pass. Phiên bản package của môi trường CPU gốc nằm trong `requirements-local.lock.txt`; đây là hồ sơ môi trường, không phải cam kết phù hợp mọi CUDA/hardware.

## 2. Chuẩn bị dữ liệu và EDA

Cài Java runtime trên máy, sau đó:

```bash
python -m vihsd.cli doctor
python -m vihsd.cli download
python -m vihsd.cli audit
python -m vihsd.cli eda
python -m vihsd.cli prepare --config configs/full20.json
```

ViHSD tải từ repository tác giả và bị chặn nếu SHA256 khác. Nguồn normalization tải theo revision ghi trong `noise.py`/`noise_v2.py`. Cần tuân theo điều kiện sử dụng của nguồn.

`prepare` tạo dữ liệu/tokens và EDA; nó không train. Không tự đổi split, nhãn hoặc deduplicate test để làm đẹp kết quả.

## 3. Các cấp độ tái lập

| Mục tiêu | Yêu cầu |
|---|---|
| Đọc code/báo cáo | Clone repo; xem Markdown/PDF/CSV |
| Unit tests | Python và dependencies; CPU được |
| Data/segmentation integration | Corpus nguồn, Internet ở lần tải đầu và Java/VnCoreNLP |
| Train Full20 | CUDA, dữ liệu/tokenizer và frozen noise bank đúng SHA |
| Tính lại 99 profile metrics | Saved predictions và noisy views trong output Kaggle gốc |
| Chạy lại inference | Selected weights, validation evidence, test views và CUDA |

Repo chỉ chứa bảng tổng hợp, không chứa raw predictions, model weights hoặc full corpora. Script verification/summarization cần các artifact ngoài Git này; không thể chạy đầy đủ chỉ từ CSV tổng hợp.

## 4. Tái chạy notebook training

Các notebook frozen trong `kaggle/full20-*` chứa source và noise bank đã khóa, không cần credential trong notebook. Đây là cách thuận tiện để xem đúng training program đã submit.

Để chạy một thực nghiệm mới:

1. Sao chép notebook vào notebook mới của tài khoản Kaggle có GPU.
2. Đặt tên/owner/slug mới và bật Internet/GPU; không ghi đè phiên nghiên cứu gốc.
3. Giữ hoặc sửa protocol có chủ đích, lưu version mới, ghi rõ thay đổi.
4. Dùng best validation checkpoint; không dùng test để chọn epoch/threshold.
5. Kiểm tra completion marker, đủ 20 epoch và saved validation evidence.

Metadata upload theo owner của máy gốc không được commit. Nếu dùng CLI, tự tạo `kernel-metadata.json` cho owner/slug của bạn. Các dispatcher dùng bridge `/home/hoang/.local/bin/kaggle-account` chỉ hoạt động khi có công cụ/credential bridge tương ứng; repo không chứa bridge, kho tài khoản hoặc key.

## 5. Tái chạy evaluation

Notebook `kaggle/full20-evaluation-*` mount training inputs private thuộc owner gốc, pin version. Máy/tài khoản mới cần có quyền đọc đúng inputs hoặc thay bằng các run mới tương ứng. Không giả rằng private weights truy cập được từ mọi tài khoản.

Evaluation phải qua validation gate trước test, giữ frozen noise bank và đối chiếu semantic view hashes. Khi đưa output về đúng layout workspace, dùng `scripts/verify_full20_evaluation.py` và `scripts/summarize_full20_evaluation.py` theo schema mô tả trong source. Các manifest vận hành theo máy/tài khoản không có trong Git; người tái chạy cần tạo manifest cho run mới, không giả COMPLETE cho artifact chưa xác minh.

## 6. Báo cáo MD/TeX/PDF

```bash
python scripts/render_research_report.py --tex-only
python scripts/render_research_report.py
python scripts/render_presentation_assignment.py
```

Cần Tectonic hoặc XeLaTeX cùng font Noto Serif và DejaVu Sans Mono. Markdown là bản nội dung chính; `.tex` là nguồn biên dịch PDF. Xem thêm `reports/final_report/README.md`.

## 7. Những giới hạn cần giữ khi tái sử dụng kết quả

- Test đã được xem ở các vòng phát triển trước: supplementary diagnostic evaluation, không phải blind confirmatory test.
- Toxic label của câu synthetic chưa được người gán lại.
- Có duplicate official split và source dependence trong calibration prior.
- Chỉ ba training seed; SD không phải CI hay kiểm định ý nghĩa thống kê.
- Chưa có XLM-R, dictionary defense, nhánh kết hợp FGM + augmentation hoặc latency benchmark.
