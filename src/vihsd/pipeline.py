"""Shared stages used by CLI and embedded Kaggle notebook."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
from pathlib import Path
from .data import audit, download, dump_json, load_raw, prepare, sha256_bytes
from .eda import create_eda


def ensure_java():
    """Use the installed JVM; pyjnius binary wheels don't need javac at runtime."""
    if not shutil.which('java'):
        raise RuntimeError('Java runtime is required for RDRSegmenter')
    home = Path(shutil.which('java')).resolve().parent.parent
    os.environ.setdefault('JAVA_HOME', str(home))
    os.environ.setdefault('JDK_HOME', str(home))
    return subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT, text=True)


def prepare_all(root, config):
    from transformers import AutoTokenizer
    from .training import resolve_config, tokenize_frames
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    print('STAGE: DATA_AUDIT', flush=True)
    raw = load_raw(root)
    audit_summary = audit(raw, root / 'reports/audit')
    # Write raw EDA before potentially expensive Java/model downloads.
    create_eda(raw, root / 'reports/eda', audit_summary)
    print('STAGE: SEGMENTATION', flush=True)
    java_version = ensure_java()
    processed = prepare(root)
    config = resolve_config(config)
    print('MODEL_REVISION:', config['model_revision'], flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], revision=config['model_revision'], use_fast=False)
    assert config['max_length'] <= tokenizer.model_max_length
    print('STAGE: TOKENIZATION_AND_EDA', flush=True)
    encoded, diagnostics = tokenize_frames(processed, tokenizer, config['max_length'], root / 'data/tokenized')
    create_eda(raw, root / 'reports/eda', audit_summary, token_lengths=diagnostics, token_frames=processed)
    dump_json(root / 'reports/java.json', {'java_version': java_version})
    dump_json(root / 'reports/resolved_config.json', config)
    source_dir = Path(__file__).parent
    source_hashes = {p.name: sha256_bytes(p.read_bytes()) for p in source_dir.glob('*.py')}
    dump_json(root / 'reports/provenance.json', {'source_sha256': source_hashes,
              'data': json.loads((root / 'data/raw/manifest.json').read_text()),
              'processed': json.loads((root / 'data/processed/latest.json').read_text()),
              'config': config})
    print('STAGE: DATA_READY', {s: len(f) for s, f in processed.items()}, flush=True)
    return processed, encoded, tokenizer, config


def run_training(root, config, stages=None):
    from .training import smoke_and_overfit, train, evaluate_checkpoint
    root = Path(root).resolve()
    frames, encoded, tokenizer, config = stages or prepare_all(root, config)
    print('STAGE: SMOKE_AND_OVERFIT', flush=True)
    checks = smoke_and_overfit(frames, encoded, tokenizer, config, root / 'reports/checks')
    run_dir = root / 'runs' / config['run_name']
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / 'reports/provenance.json', run_dir / 'provenance.json')
    print('STAGE: FULL_TRAINING', flush=True)
    best = train(frames, encoded, tokenizer, config, run_dir)
    print('STAGE: FINAL_TEST', flush=True)
    result = evaluate_checkpoint(best, frames, encoded, tokenizer, run_dir / 'evaluation', config['eval_batch_size'])
    write_baseline_report(root, run_dir, result)
    dump_json(root / 'RUN_COMPLETE.json', {'run_dir': str(run_dir), 'test_metrics': result, 'checks': checks})
    print('RUN_COMPLETE', flush=True)
    return result


def write_baseline_report(root, run_dir, result):
    history = json.loads((run_dir / 'history.json').read_text())
    best = json.loads((run_dir / 'best/selection.json').read_text())
    majority = json.loads((root / 'reports/eda/majority_baseline.json').read_text())['test']
    text = f'''# Baseline PhoBERT — lần training đầu tiên

Đây là baseline mới phục vụ nghiên cứu robustness, không phải tái lập điểm PhoBERT trong bảng kết quả bài ViHSD 2021.

- Checkpoint chọn bằng **validation Macro-F1**, epoch **{best['epoch']}**, score **{best['score']:.4f}**.
- Test gốc: **Macro-F1 {result['macro_f1']:.4f}**, **accuracy {result['accuracy']:.4f}**.
- Majority baseline test: Macro-F1 {majority['macro_f1']:.4f}, accuracy {majority['accuracy']:.4f}.
- Số epoch hoàn thành: {len(history)}. Một seed 42, chưa đánh giá độ biến thiên qua nhiều seed.
- Train gốc 24.048 mẫu; loại 2 mẫu rỗng có ID được ghi trong audit, học trên 24.046 mẫu. Validation/test giữ nguyên.
- Giữ nguyên duplicate trong split chính thức. Trùng giữa các split và xung đột nhãn là giới hạn của kết quả này.
- Không sửa teencode, bỏ dấu, xóa emoji hay stopword. Chưa thêm nhiễu nhân tạo và chưa áp dụng FGM.

## Chỉ số từng lớp

| Nhãn | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
'''
    for label in ['CLEAN', 'OFFENSIVE', 'HATE']:
        c = result['per_class'][label]
        text += f"| {label} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1-score']:.4f} | {c['support']:.0f} |\n"
    text += '\n## Hướng đọc tiếp\n\nXem confusion matrix và error_review.csv để mô tả lỗi. Không suy luận segmentation gây lỗi nếu chưa làm ablation. Đợt tiếp theo: cố định checkpoint và xây bộ nhiễu có kiểm tra bảo toàn nhãn.\n'
    (run_dir / 'BASELINE_REPORT.md').write_text(text, encoding='utf-8')
