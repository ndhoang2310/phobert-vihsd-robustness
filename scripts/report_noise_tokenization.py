"""Summarize segmentation/BPE effects on shared views, without equating UNK to errors."""
import json
from pathlib import Path
import pandas as pd
from vihsd.data import load_raw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/robustness'
root=ROOT/'artifacts/robustness/augmentation-v2/robustness_project'
d=json.loads((root/'data/tokenized/token_diagnostics.json').read_text())['test']
frame=load_raw(ROOT)['test'];n=len(frame)
# Read authoritative original lengths from the actual Kaggle preprocessing run.
assert len(d['lengths'])==n and len(d['unk_counts'])==n
rows=[{'profile':'original','mean_bpe_length':sum(d['lengths'])/n,'unk_token_rate':sum(d['unk_counts'])/max(1,sum(d['lengths'])-2*n),'truncation_rate':sum(x>256 for x in d['lengths'])/n,'mean_fragmentation':None,'delta_segmented_units_mean':0,'coverage':0,'actual_rate_mean':0}]
summary=json.loads((root/'robustness/augmentation-v2/summary.json').read_text())
for profile,r in summary.items():
 if profile=='original':continue
 rows.append({'profile':profile,**r['tokenization'],'coverage':r['coverage'],'actual_rate_mean':r['actual_rate_mean']})
for row in rows:row['unk_tokens_total']=round(row['unk_token_rate']*(row['mean_bpe_length']-2)*n)
table=pd.DataFrame(rows);table.to_csv(OUT/'tokenization_effects.csv',index=False)
text='''# Ảnh hưởng của noise lên segmentation và BPE

Số token tính trước truncation. UNK token rate dùng số token nội dung, bỏ BOS/EOS; truncation rate là tỷ lệ câu dài hơn max length 256. Fragmentation là số BPE content tokens / số đơn vị RDRSegmenter. Các chỉ số mô tả hành vi tokenizer, không phải gold segmentation accuracy. Không gọi mọi UNK hoặc mọi thay đổi segmentation là lỗi.

UNK rate có thể giảm vì mẫu số BPE tăng, ngay cả khi tổng UNK gần như không đổi; xem `unk_tokens_total` trong CSV trước khi diễn giải. Không suy ra tokenizer tốt hơn chỉ từ tỷ lệ UNK thấp hơn.\n\nOriginal fragmentation không được ghi trực tiếp trong summary nên để trống, không suy đoán. Xem thêm segmented-unit drift và changed-only F1 trong bảng kết quả model. Các mức stress là yêu cầu; luôn đọc cùng coverage và actual rate.

| Profile | BPE length | UNK token rate | Truncation | Fragmentation | Δ RDR units | Coverage | Actual rate |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
for r in rows:
 frag='—' if r['mean_fragmentation'] is None else f"{r['mean_fragmentation']:.3f}"
 text+=f"| {r['profile']} | {r['mean_bpe_length']:.2f} | {r['unk_token_rate']:.4%} | {r['truncation_rate']:.2%} | {frag} | {r['delta_segmented_units_mean']:+.3f} | {r['coverage']:.2%} | {r['actual_rate_mean']:.2%} |\n"
(OUT/'TOKENIZATION_EFFECTS.md').write_text(text);print(table.to_string(index=False))
