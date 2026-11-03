"""Supplement EDA with Unicode integrity and unknown-token interpretation."""
import json,re
from pathlib import Path
import pandas as pd
from vihsd.data import dump_json,supplementary_sequence
ROOT=Path(__file__).resolve().parents[1]
eda=ROOT/'reports/eda'
processed=Path(json.loads((ROOT/'data/processed/latest.json').read_text())['path'])
integrity={}
for split in ['train','validation','test']:
 frame=pd.read_parquet(processed/f'{split}.parquet')
 changes=sum(supplementary_sequence(a)!=supplementary_sequence(b) for a,b in zip(frame.text_normalized,frame.text_segmented))
 integrity[split]={'rows':len(frame),'non_bmp_sequence_changed_rows':changes}
 assert changes==0
assert 'utf8-v2' in frame.preprocess_version.iloc[0]
dump_json(eda/'segmentation_integrity.json',integrity)
f=pd.read_csv(eda/'token_lengths.csv').merge(pd.read_csv(eda/'sample_features.csv')[['sample_id','has_emoji']],on='sample_id',validate='one_to_one')
f['has_unk']=f.unk_tokens.gt(0)
summary=f.groupby('split').agg(samples=('sample_id','count'),samples_with_unk=('has_unk','sum'),unk_sample_rate=('has_unk','mean'))
by_emoji=f.query('split == "train"').groupby('has_emoji').agg(samples=('sample_id','count'),samples_with_unk=('has_unk','sum'),unk_sample_rate=('has_unk','mean'))
summary.to_csv(eda/'unk_sample_rates.csv');by_emoji.to_csv(eda/'train_unk_by_emoji.csv')
text='''# Unicode và cách diễn giải `<unk>`\n\n## Bảo toàn Unicode\n\nĐã sửa cầu nối Java sang `String(byte[], UTF-8)`. Kiểm tra toàn bộ train/validation/test: **0 mẫu thay đổi chuỗi ký tự non-BMP qua segmentation**. Điều này bao gồm emoji supplementary. Các kết quả của preprocessing v1 bị đánh dấu không hợp lệ và được tách riêng.\n\n## Mẫu có `<unk>`\n\n`<unk>` không đồng nghĩa segmentation sai; emoji cũng có thể nằm ngoài vocabulary PhoBERT.\n\n| Split | Số mẫu | Mẫu có `<unk>` | Tỷ lệ |\n|---|---:|---:|---:|\n'''
for split,r in summary.iterrows():text+=f"| {split} | {int(r.samples)} | {int(r.samples_with_unk)} | {r.unk_sample_rate:.2%} |\n"
text+='\n## Phân tầng train theo tín hiệu emoji\n\n| Có tín hiệu emoji | Số mẫu | Mẫu có `<unk>` | Tỷ lệ |\n|---|---:|---:|---:|\n'
for emoji,r in by_emoji.iterrows():text+=f"| {bool(emoji)} | {int(r.samples)} | {int(r.samples_with_unk)} | {r.unk_sample_rate:.2%} |\n"
text+='\nĐây là tương quan bề mặt, không chứng minh emoji là nguyên nhân duy nhất sinh `<unk>`. Tỷ lệ mẫu có `<unk>` khác tỷ lệ `<unk>` trên tổng token nội dung; phân mảnh BPE được đo riêng.\n'
(eda/'UNK_NOTE.md').write_text(text)
p=ROOT/'reports/DATA_REVIEW.md';text=p.read_text().replace('12 kiểm thử','14 kiểm thử');text+='\n## Unicode integrity\n\nPreprocessing hợp lệ là `nfc-space-rdr-utf8-v2`. Cầu nối UTF-8 được kiểm tra trên Java thật; toàn bộ train/validation/test có 0 mẫu thay đổi chuỗi non-BMP. Version 1 không được dùng để báo cáo baseline.\n' if '## Unicode integrity' not in text else '';p.write_text(text)
print(json.dumps(integrity,indent=2));print(summary)
