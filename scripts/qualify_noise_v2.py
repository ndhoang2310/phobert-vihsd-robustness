"""Language-only v2 qualification, using held-out in-domain human normalization."""
import collections,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from vihsd.noise import fetch_corpus,align_pairs,NoiseGenerator,generate_view
from vihsd.noise_v2 import fetch_in_domain
from vihsd.data import load_raw,dump_json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/noise_v2';OUT.mkdir(parents=True,exist_ok=True)
c,_=fetch_corpus(ROOT);d,_=fetch_in_domain(ROOT);f=load_raw(ROOT)
banks={v:json.load(open(ROOT/f'data/{folder}/bank.json')) for v,folder in [('pilot_v1','noise'),('dual_source_v2','noise_v2')]}
reference=[p for r in d['test'].itertuples() for p in align_pairs(r.original,r.normalized)];ref=collections.Counter(p['kind'] for p in reference)
kinds=['abbreviation','diacritics','elongation','spelling_variant'];p=np.array([ref[k] for k in kinds],float);p/=p.sum()
rows=[];summary={}
for version,bank in banks.items():
 view=generate_view(f['validation'],NoiseGenerator(bank),101);events=[e for edits in view.edits for e in edits];counts=collections.Counter(e['kind'] for e in events);q=np.array([counts[k] for k in kinds],float);q/=q.sum()
 supported={(r['canonical'],r['variant']) for r in bank['rows'] if r['partition']=='train'}
 summary[version]={'train_mappings':sum(r['partition']=='train' for r in bank['rows']),'unseen_mappings':sum(r['partition']=='unseen' for r in bank['rows']),'in_domain_heldout_event_coverage':sum((r['canonical'],r['variant']) in supported for r in reference)/len(reference),'validation_changed_rate':float(view.changed.mean()),'event_type_js_bits_vs_in_domain_heldout':float(jensenshannon(p,q,base=2)**2)}
 for i,kind in enumerate(kinds):rows.append({'kind':kind,'source':version,'proportion':q[i]})
 if version=='dual_source_v2':
  sample=view[view.changed].sample(min(30,int(view.changed.sum())),random_state=42).copy();sample['edits']=sample.edits.map(json.dumps);sample.to_csv(OUT/'validation_examples.csv',index=False)
for i,kind in enumerate(kinds):rows.append({'kind':kind,'source':'human_in_domain_heldout','proportion':p[i]})
table=pd.DataFrame(rows);table.to_csv(OUT/'type_mix_qualification.csv',index=False);dump_json(OUT/'qualification_comparison.json',summary)
sns.set_theme(style='whitegrid',font='DejaVu Sans');fig,ax=plt.subplots(figsize=(11,4.8));sns.barplot(data=table,x='kind',y='proportion',hue='source',ax=ax,palette='colorblind');ax.set(title='Kiểm tra kiểu edit bằng nguồn human-annotated in-domain',xlabel='',ylabel='Tỷ lệ sự kiện trong tập đã lọc');ax.tick_params(axis='x',rotation=10);fig.tight_layout();fig.savefig(OUT/'01_human_reference_comparison.png',dpi=160);plt.close(fig)
a=json.load(open(OUT/'qualification.json'));v2=summary['dual_source_v2'];v1=summary['pilot_v1']
text=f'''# Noise v2 — nguồn human-annotated và kiểm tra in-domain

## Vì sao có phiên bản 2

Pilot lấy biến thể từ ViLexNorm và cho thấy độ bao phủ hạn chế khi đối chiếu Small-LexNormViHSD. Quyết định v2 dựa trên dữ liệu/ngôn ngữ, không tối ưu theo test F1. Kết quả pilot được giữ riêng, không trộn điểm giữa hai testbed.

## Nguồn có sẵn

- ViLexNorm: https://aclanthology.org/2024.eacl-long.85/ ; revision `{banks['dual_source_v2']['source_revision']}`. Dùng {len(c['train']):,} cặp train.
- Small-LexNormViHSD: https://arxiv.org/abs/2311.06851 ; repo https://github.com/AnhHoang0529/Small-LexNormViHSD , revision `{banks['dual_source_v2']['in_domain_revision']}`. Dùng {len(d['train']):,} cặp released train để học nguồn biến thể; giữ {len(d['test'])} cặp released test làm kiểm tra linguistic. Corpus này được xây từ ViHSD train, không phải toxic test độc lập.

## Chất lượng annotation gốc

[ViLexNorm, mục 3.2](https://aclanthology.org/2024.eacl-long.85.pdf) báo agreement kiểm tra chuẩn hóa theo câu 88,46% giữa tác giả và annotators, 93,54% giữa hai tác giả. Đây không phải Cohen kappa hoặc xác suất giữ nhãn toxic của generator.

[Small-LexNormViHSD, mục 3.3](https://arxiv.org/html/2311.06851) dùng hai người/câu; kappa cho quyết định token cần chuẩn hóa là 0,9014, còn agreement về cùng candidate là 0,7449. Hai chỉ số khác nhau và không thể dùng để chứng nhận nhãn toxic của câu tổng hợp.

## Thay đổi kỹ thuật

1. Mọi mapping đều có cặp gốc/chuẩn hóa và source ID. Không tự sáng tác typo hoặc tiếng lóng.
2. Trọng số chọn edit là `count(canonical, variant) / canonical_opportunities`, rồi chuẩn hóa trên các span đủ điều kiện. Sau đó nhân hệ số loại edit học từ tỷ lệ sự kiện Small-LexNorm TRAIN, tối ưu JSD có regularization và giới hạn log-weight [-3,3] trên 2000 ViHSD TRAIN comments chọn bằng hash, không dùng nhãn toxic. Đây là tỷ lệ trong corpus có chọn lọc NSW, không phải xác suất lỗi phổ quát của UGC.
3. Lọc thêm variant xuất hiện như từ không đổi trong annotation; tỷ lệ target thống nhất >=0,95 và ít nhất 2 source comments cho mapping train. Ngưỡng này là quy tắc curation, **không phải 95% xác suất giữ nhãn toxic**.
4. Variant quen thuộc trong raw ViHSD train được giữ ở bank train; chỉ đưa vào unseen khi surface vắng trong train thật. Giữ tách biệt giữa hai bank.
5. Loại {len(banks['dual_source_v2']['excluded_evaluation_overlap_source_ids'])} source pairs trùng evaluation sau bỏ whitespace/NFC/casefold; không lấy nhãn hoặc điểm test để fit.
6. Giữ ALL CAPS khi áp dụng variant và bảo toàn toàn bộ non-BMP/emoji qua preprocessing.

219 cặp linguistic test đã được xem trong quá trình phát triển, nên đây là diagnostic reference, không phải blind final test. Tham số chỉ fit trên TRAIN; không dùng test F1 của classifier.

## Kết quả kiểm tra nguồn held-out

| Chỉ số | Pilot v1 | Dual-source v2 |
|---|---:|---:|
| Mapping train | {v1['train_mappings']} | {v2['train_mappings']} |
| Mapping unseen | {v1['unseen_mappings']} | {v2['unseen_mappings']} |
| Coverage lexical events đủ điều kiện trên human in-domain held-out | {v1['in_domain_heldout_event_coverage']:.2%} | {v2['in_domain_heldout_event_coverage']:.2%} |
| Validation comment có sửa | {v1['validation_changed_rate']:.2%} | {v2['validation_changed_rate']:.2%} |
| JSD loại edit so human in-domain held-out (bits) | {v1['event_type_js_bits_vs_in_domain_heldout']:.4f} | {v2['event_type_js_bits_vs_in_domain_heldout']:.4f} |

Đây là coverage trên {len(reference)} events qua bộ lọc alignment, không phải toàn bộ NSW do con người gán nhãn. Coverage vẫn chưa đầy đủ: các từ nhiều nghĩa/cách mở rộng như một số viết tắt cảm thán và từ chỉ người bị loại để giảm rủi ro thay đổi nhãn. Không ép coverage bằng cách thêm rule tự suy đoán.

Audit bổ sung: không có exact original-text overlap giữa hai source TRAIN. Có 2 mapping mà nhiều source IDs chỉ tương ứng 1 cặp văn bản duy nhất, nên số source IDs không phải số ngữ cảnh độc lập. Bộ lọc 71 source overlaps áp dụng cho mapping evidence; prior 4 loại edit vẫn dùng toàn bộ SmallLex released TRAIN và có thể chứa các comment train trùng evaluation do split gốc. Không dùng nhãn hoặc dự đoán evaluation. Đây là giới hạn contamination còn lại, không gọi protocol là hoàn toàn không leakage.

Các loại edit được phân loại heuristic từ cặp normalization do người tạo; nguồn không cung cấp annotation thủ công cho từng loại noise này. Thứ tự: cùng dạng bỏ dấu → diacritics; dạng lặp ký tự → elongation; surface ngắn hơn → abbreviation; còn lại → spelling_variant. Vì vậy nhãn abbreviation bao gồm cả một số lỗi rút ngắn như `tiền → tìn`, không đồng nghĩa mọi edit thuộc loại này đều là viết tắt theo phân tích ngôn ngữ.

## Giới hạn kết luận

Chưa có người gán lại toxic labels cho các câu tổng hợp, nên việc giữ nhãn là giả định có lọc. Kiểm tra linguistic và JSD không chứng nhận mọi câu mới tự nhiên. Vì vậy đánh giá chính luôn gồm test ViHSD gốc có nhãn thật, nhóm UGC tự nhiên và paired corruption tests được ghi rõ là synthetic. Một sự kiện lexical/comment ở empirical profile; mức 10/20/30/50% chỉ là controlled stress và phải đọc tỷ lệ sửa thực tế.

![So sánh nguồn](01_human_reference_comparison.png)
'''
(OUT/'NOISE_RESEARCH_V2.md').write_text(text);print(json.dumps(summary,indent=2))
