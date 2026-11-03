"""EDA artifacts with counts, train-only inspection and an HTML review report."""
from __future__ import annotations
import html
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from .data import LABELS, dump_json, normalize_text

PALETTE = ['#3979a8', '#d79c34', '#bd5066']


def features(frames):
    result = []
    for split, frame in frames.items():
        f = frame[['sample_id', 'split', 'label']].copy()
        text = frame.text_raw.fillna('').astype(str)
        f['characters'] = text.str.len()
        f['whitespace_units'] = text.str.split().str.len()
        f['has_emoji'] = text.str.contains(r'[\U0001F000-\U0001FAFF\u2600-\u27BF]', regex=True)
        f['has_url'] = text.str.contains(r'https?://|www\.', case=False, regex=True)
        f['has_mention_marker'] = text.str.contains(r'@\w+|<person\s*name>', case=False, regex=True)
        f['has_hashtag'] = text.str.contains(r'#\w+', regex=True)
        f['has_digit'] = text.str.contains(r'\d', regex=True)
        f['has_elongation'] = text.map(lambda t: bool(re.search(r'([^\W\d_])\1{2,}', t, re.IGNORECASE)))
        f['has_uppercase'] = text.map(lambda t: any(c.isupper() for c in t))
        f['empty_after_normalization'] = text.map(lambda t: not normalize_text(t))
        result.append(f)
    return pd.concat(result, ignore_index=True)


def create_eda(frames, output, audit_summary, token_lengths=None, token_frames=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', context='notebook', font='DejaVu Sans')
    plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 160, 'axes.titleweight': 'bold'})
    f = features(frames)
    f.to_csv(output / 'sample_features.csv', index=False)
    counts = pd.crosstab(f.split, f.label).reindex(index=['train', 'validation', 'test'], columns=[0, 1, 2], fill_value=0)
    counts.columns = LABELS
    counts.to_csv(output / 'label_counts.csv')
    proportions = counts.div(counts.sum(axis=1), axis=0)
    proportions.to_csv(output / 'label_proportions.csv')
    quantiles = f.groupby(['split', 'label'])[['characters', 'whitespace_units']].quantile([0, .25, .5, .75, .9, .95, .99, 1])
    quantiles.to_csv(output / 'length_quantiles.csv')
    ugc_columns = [c for c in f if c.startswith('has_')]
    ugc = f.groupby('split')[ugc_columns].mean().reindex(['train', 'validation', 'test'])
    ugc.to_csv(output / 'ugc_rates.csv')
    figures = []

    def save(name, fig):
        fig.savefig(output / name, bbox_inches='tight')
        plt.close(fig)
        figures.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    counts.plot.bar(ax=axes[0], color=PALETTE, rot=0)
    proportions.plot.bar(ax=axes[1], stacked=True, color=PALETTE, rot=0)
    axes[0].set(title='Số mẫu theo nhãn và split', ylabel='Số mẫu', xlabel='')
    axes[1].set(title='Tỷ lệ nhãn theo split', ylabel='Tỷ lệ', xlabel='', ylim=(0, 1))
    axes[1].legend(loc='lower right')
    save('01_labels.png', fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for split, color in zip(['train', 'validation', 'test'], PALETTE):
        subset = f[f.split == split]
        sns.histplot(data=subset, x='characters', bins=np.linspace(0, 1000, 51), stat='probability',
                     element='step', fill=False, label=split, color=color, ax=axes[0])
        sns.ecdfplot(data=subset, x='whitespace_units', label=split, color=color, ax=axes[1])
    axes[0].set(title='Độ dài ký tự (khung nhìn 0–1.000)', xlabel='Số ký tự', ylabel='Tỷ lệ mẫu')
    axes[0].legend()
    axes[1].set(title='Phân phối tích lũy độ dài', xlabel='Đơn vị cách nhau bằng khoảng trắng', ylabel='Tỷ lệ tích lũy', xlim=(0, 200))
    axes[1].legend()
    save('02_lengths.png', fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.boxplot(data=f[f.split == 'train'], x='label', y='whitespace_units', hue='label',
                palette=PALETTE, showfliers=False, legend=False, ax=ax)
    ax.set(xticks=[0, 1, 2], xticklabels=LABELS, xlabel='Nhãn', ylabel='Đơn vị cách nhau bằng khoảng trắng',
           title='Độ dài theo nhãn — train (ẩn outlier để dễ đọc)')
    save('03_length_by_class.png', fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(ugc * 100, annot=True, fmt='.1f', cmap='Blues', ax=ax, cbar_kws={'label': '% mẫu'})
    ax.set(title='Tín hiệu bề mặt UGC — tỷ lệ mẫu (%)', xlabel='', ylabel='')
    ax.set_xticklabels(['Emoji', 'URL', 'Mention marker', 'Hashtag', 'Chữ số', 'Lặp ký tự', 'Chữ hoa'], rotation=20, ha='right')
    save('04_ugc.png', fig)

    majority = int(frames['train'].label.value_counts().idxmax())
    baseline = {}
    for split, frame in frames.items():
        predicted = np.full(len(frame), majority)
        baseline[split] = {'majority_label': majority, 'accuracy': accuracy_score(frame.label, predicted),
                           'macro_f1': f1_score(frame.label, predicted, labels=[0, 1, 2], average='macro', zero_division=0)}
    dump_json(output / 'majority_baseline.json', baseline)
    examples = pd.concat([frames['train'].query('label == @label').sample(30, random_state=42) for label in [0, 1, 2]])
    examples.to_csv(output / 'train_review_sample.csv', index=False)
    if token_lengths is not None:
        token_df, token_summary = token_diagnostics(token_frames if token_frames is not None else frames, token_lengths, output)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for split, color in zip(['train', 'validation', 'test'], PALETTE):
            sns.ecdfplot(data=token_df[token_df.split == split], x='bpe_tokens', label=split, color=color, ax=ax)
        for threshold in [100, 128, 256]:
            ax.axvline(threshold, color='#8b9299', linestyle='--', linewidth=1)
        ax.set(title='Độ dài BPE trước truncation (gồm special tokens)', xlabel='Số token', ylabel='Tỷ lệ tích lũy', xlim=(0, 400))
        ax.legend()
        save('05_bpe_lengths.png', fig)
    else:
        token_summary = None
    raw_dup = audit_summary['duplicates']['normalized']
    empty_n = int(f.empty_after_normalization.sum())
    train_clean_rate = proportions.loc['train', 'CLEAN'] * 100
    text = f'''# EDA ViHSD — baseline PhoBERT phục vụ robustness

## Dữ liệu và phạm vi

Nguồn: archive trong GitHub chính thức `sonlam1102/vihsd`, phiên bản/checksum trong manifest.
Giữ nguyên split chính thức. Test gốc nghĩa là chưa thêm nhiễu nhân tạo; vẫn có lỗi gõ và UGC tự nhiên.
Các thống kê test chỉ mô tả dữ liệu, không dùng để lựa chọn siêu tham số hoặc sửa preprocessing.

## Phát hiện chính

- Tổng **{len(f):,}** mẫu gốc; train **{len(frames['train']):,}**, validation **{len(frames['validation']):,}**, test **{len(frames['test']):,}**.
- CLEAN chiếm **{train_clean_rate:.2f}%** train. Accuracy cao có thể do dự đoán lớp đa số; chọn **Macro-F1** làm chỉ số chính.
- Baseline luôn đoán CLEAN trên validation: accuracy **{baseline['validation']['accuracy']:.4f}**, Macro-F1 **{baseline['validation']['macro_f1']:.4f}**.
- Có **{empty_n}** dòng rỗng sau chuẩn hóa kỹ thuật. Hai dòng rỗng thuộc train được ghi vào exclusion ledger; raw và split đánh giá giữ nguyên.
- Sau NFC/khoảng trắng: **{raw_dup['cross_split_groups']:,} nhóm trùng giữa các split**, **{raw_dup['conflicting_label_groups']:,} nhóm cùng văn bản khác nhãn** trên toàn bộ dữ liệu. Đây là giới hạn quan trọng của đánh giá split chính thức; không tự deduplicate.
- Emoji/chữ hoa/ký tự lặp là tín hiệu bề mặt, không phải nhãn độc hại. Giữ nguyên để baseline phản ánh UGC.

## Quyết định pipeline

1. Giữ nguyên train/validation/test chính thức; chỉ loại hai mẫu train không có nội dung theo ledger có ID cụ thể.
2. Unicode NFC và khoảng trắng; giữ dấu, case, emoji, URL, hashtag, teencode và ký tự kéo dài.
3. Truyền UTF-8 qua cầu nối Java để bảo toàn emoji/non-BMP, rồi RDRSegmenter và PhoBERT BPE; không xóa stopword, không sửa chính tả bằng từ điển.
4. `max_length=256` gồm special tokens. Thống kê truncation dưới đây dùng để ghi nhận mất thông tin; cấu hình được định trước và không điều chỉnh theo test.
5. Cross-entropy thường, không resampling/class weight ở lần chạy đầu. Báo cáo precision/recall/F1 theo từng lớp.

## Cách đọc số liệu

Độ dài theo khoảng trắng là số đơn vị phân cách, không phải số từ tiếng Việt sau segmentation.
Mention marker chỉ nhận `@...` hoặc placeholder, không suy ra được toàn bộ tên người được mention.
Emoji và lặp ký tự được nhận dạng heuristic nên không phải annotation chuẩn.
Không suy luận quan hệ nhân quả giữa segmentation và sai nhãn từ EDA.
'''
    if token_summary:
        text += '\n## Tokenization thực tế\n\n'
        for split, stats in token_summary.items():
            text += f"- **{split}**: median {stats['p50']:.0f}, p95 {stats['p95']:.0f}, p99 {stats['p99']:.0f} token; cắt ở 256: {stats['truncated_at_256_rate']*100:.2f}%; `<unk>` trên token nội dung: {stats['unk_token_rate']*100:.4f}%.\n"
        text += '\nKhông đồng nhất `<unk>` với phân mảnh: hai chỉ số được đo riêng trong bảng tokenization.\n'
    for name in figures:
        text += f'\n![{name}]({name})\n'
    (output / 'EDA_REPORT.md').write_text(text, encoding='utf-8')
    html_parts = ["<!doctype html><meta charset='utf-8'><title>EDA ViHSD</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;color:#203040;line-height:1.6}img{width:100%;margin:20px 0}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}</style>",
                  '<h1>EDA ViHSD — baseline PhoBERT</h1>', '<pre>' + html.escape(text.split('![', 1)[0]) + '</pre>', counts.to_html()]
    html_parts += [f'<img src="{name}" alt="{name}">' for name in figures]
    (output / 'index.html').write_text('\n'.join(html_parts), encoding='utf-8')
    return {'n': len(f), 'majority_baseline': baseline, 'tokenization': token_summary, 'figures': figures}


def token_diagnostics(frames, token_data, output):
    """token_data: split -> lengths, unk_counts, content_counts (all before truncation)."""
    rows = []
    summary = {}
    for split, frame in frames.items():
        details = token_data[split]
        lengths = np.asarray(details['lengths'])
        assert len(lengths) == len(frame)
        units = frame.text_segmented.str.split().str.len().clip(lower=1).to_numpy()
        df = frame[['sample_id', 'split', 'label']].copy()
        df['bpe_tokens'] = lengths
        df['bpe_per_segmented_unit'] = np.asarray(details['content_counts']) / units
        df['unk_tokens'] = details['unk_counts']
        rows.append(df)
        stats = {'p50': float(np.quantile(lengths, .5)), 'p95': float(np.quantile(lengths, .95)),
                 'p99': float(np.quantile(lengths, .99)), 'max': int(lengths.max()),
                 'mean_bpe_per_segmented_unit': float(df.bpe_per_segmented_unit.mean()),
                 'unk_token_rate': float(sum(details['unk_counts']) / max(1, sum(details['content_counts'])))}
        for threshold in [100, 128, 256]:
            stats[f'truncated_at_{threshold}_rate'] = float((lengths > threshold).mean())
        summary[split] = stats
    data = pd.concat(rows, ignore_index=True)
    data.to_csv(Path(output) / 'token_lengths.csv', index=False)
    dump_json(Path(output) / 'tokenization_summary.json', summary)
    return data, summary
