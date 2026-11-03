"""Build a self-contained private Kaggle notebook from the current source snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def build(username='ndhoang2310'):
    destination = ROOT / 'kaggle/baseline'
    destination.mkdir(parents=True, exist_ok=True)
    sources = {str(p.relative_to(ROOT / 'src')): p.read_text(encoding='utf-8')
               for p in sorted((ROOT / 'src/vihsd').glob('*.py'))}
    config = json.loads((ROOT / 'configs/baseline.json').read_text())
    source_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    bootstrap = '''import os, sys, subprocess, json, shutil
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["MPLBACKEND"] = "Agg"
WORK = Path("/kaggle/working")
ROOT = WORK / "vihsd_project"
ROOT.mkdir(parents=True, exist_ok=True)
# Retain the CUDA-compatible PyTorch already installed by Kaggle.
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
    "transformers==4.57.1", "huggingface-hub>=0.34,<1", "accelerate>=1.2,<2", "py-vncorenlp==0.1.4",
    "seaborn>=0.13,<0.14", "pyarrow>=17", "scikit-learn>=1.5", "pandas>=2.2,<4"])
if not shutil.which("java"):
    subprocess.check_call(["apt-get", "update", "-qq"])
    subprocess.check_call(["apt-get", "install", "-y", "-qq", "default-jre-headless"])
java_home = str(Path(shutil.which("java")).resolve().parent.parent)
os.environ["JAVA_HOME"] = java_home
os.environ["JDK_HOME"] = java_home
print(subprocess.check_output(["nvidia-smi"], text=True))
import torch
assert torch.cuda.is_available(), "Full baseline requires a GPU. Enable GPU in Kaggle settings."
assert torch.cuda.device_count() == 1, "This protocol uses exactly one visible GPU"
print("GPU:", torch.cuda.get_device_name(0), "Python:", sys.version)
'''
    install_source = 'SOURCE_SNAPSHOT = ' + repr(sources) + '\n'
    install_source += '''for relative_path, content in SOURCE_SNAPSHOT.items():
    target = ROOT / "src" / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
sys.path.insert(0, str(ROOT / "src"))
'''
    install_source += 'CONFIG = ' + repr(config) + '\n'
    install_source += f'print("SOURCE_SHA256", {source_hash!r})\n'
    install_source += '(ROOT / "config.json").write_text(json.dumps(CONFIG, indent=2))\n'
    cells = [
        nbformat.v4.new_markdown_cell('''# ViHSD · PhoBERT baseline đầu tiên

**Mục tiêu:** kiểm tra dữ liệu → EDA → RDRSegmenter → PhoBERT → kiểm tra học → fine-tune → đánh giá test gốc.

- Baseline mới phục vụ robustness; không gọi là tái lập một điểm PhoBERT trong paper ViHSD 2021.
- Nguồn dữ liệu: archive tại repo chính thức của tác giả, pin commit và SHA-256 trong code.
- Giữ nguyên split; 2 dòng train rỗng được loại với ledger cụ thể. Validation/test không thay đổi.
- Giữ emoji, case, dấu, stopword, teencode và ký tự kéo dài.
- Chọn model bằng **validation Macro-F1**. Test gốc chỉ đánh giá sau khi chọn checkpoint.
- Một GPU, seed 42, 5 epoch. Không thêm nhiễu hoặc FGM trong lần này.

Artifact ở `/kaggle/working/vihsd_project/`: `reports/`, `runs/`, dữ liệu và config để tái hiện.
Source được nhúng trực tiếp từ repo TMG để notebook không phụ thuộc một GitHub cá nhân chưa publish.
'''),
        nbformat.v4.new_markdown_cell('## 1. Môi trường và source snapshot'),
        nbformat.v4.new_code_cell(bootstrap),
        nbformat.v4.new_code_cell(install_source),
        nbformat.v4.new_markdown_cell('''## 2. Audit, preprocessing và EDA

Các dòng trùng và xung đột nhãn được báo cáo, không tự deduplicate split chính thức. Các ví dụ kiểm tra preprocessing chỉ lấy từ train. Thống kê BPE được đo trước truncation, gồm special tokens.'''),
        nbformat.v4.new_code_cell('''from vihsd.pipeline import prepare_all
STAGES = prepare_all(ROOT, CONFIG)
FRAMES, ENCODED, TOKENIZER, CONFIG = STAGES
'''),
        nbformat.v4.new_code_cell('''from IPython.display import display, Markdown, Image
import pandas as pd
EDA = ROOT / "reports/eda"
display(Markdown((EDA / "EDA_REPORT.md").read_text().split("![", 1)[0]))
display(pd.read_csv(EDA / "label_counts.csv"))
display(pd.read_csv(EDA / "ugc_rates.csv"))
for picture in sorted(EDA.glob("0*.png")):
    display(Image(filename=str(picture)))
print("Audit:")
print((ROOT / "reports/audit/audit.json").read_text())
print("Tokenization:")
print((EDA / "tokenization_summary.json").read_text())
'''),
        nbformat.v4.new_markdown_cell('''## 3. Kiểm tra học và training đầy đủ

Smoke test dùng pretrained weights thật, kiểm tra loss/gradient, save/reload và optimizer state. Sau đó overfit 48 mẫu train. Model kiểm tra được hủy trước khi khởi tạo lại baseline với seed 42.

Full training chỉ dùng train và validation. Dự đoán test chạy một lần sau khi hoàn tất lựa chọn checkpoint. Checkpoint gần nhất (`last.pt`) chứa optimizer, scheduler và RNG để resume. Chỉ load file checkpoint do chính run này tạo.
'''),
        nbformat.v4.new_code_cell('''from vihsd.pipeline import run_training
RESULT = run_training(ROOT, CONFIG, stages=STAGES)
'''),
        nbformat.v4.new_markdown_cell('## 4. Kết quả và artifact'),
        nbformat.v4.new_code_cell('''RUN = ROOT / "runs" / CONFIG["run_name"]
display(Markdown((RUN / "BASELINE_REPORT.md").read_text()))
for name in ["training_curves.png", "confusion_matrix.png"]:
    display(Image(filename=str(RUN / "evaluation" / name)))
print("RUN_COMPLETE:", (ROOT / "RUN_COMPLETE.json").exists())
print("Best checkpoint:", RUN / "best")
print("Resume checkpoint:", RUN / "last.pt")
print("Predictions:", RUN / "evaluation/predictions.parquet")
''')]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    notebook.metadata.language_info = {'name': 'python'}
    notebook.metadata['source_sha256'] = source_hash
    nbformat.validate(notebook)
    nbformat.write(notebook, destination / 'vihsd_phobert_baseline.ipynb')
    nbformat.write(notebook, ROOT / 'notebooks/vihsd_phobert_kaggle.ipynb')
    metadata = {'id': f'{username}/vihsd-phobert-first-training', 'title': 'ViHSD PhoBERT First Training',
                'code_file': 'vihsd_phobert_baseline.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                'is_private': True, 'enable_gpu': True, 'enable_internet': True,
                'dataset_sources': [], 'competition_sources': [], 'kernel_sources': []}
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'directory': str(destination), 'source_sha256': source_hash, 'kernel': metadata['id']}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--username', default='ndhoang2310')
    build(parser.parse_args().username)
