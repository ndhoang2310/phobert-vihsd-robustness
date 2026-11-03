"""Document source evidence, selection bias and diagnostic fidelity (not human validation)."""
import collections,difflib,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from vihsd.noise import fetch_corpus,align_pairs,NoiseGenerator,generate_view
from vihsd.data import load_raw,dump_json,normalize_text

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/noise';OUT.mkdir(parents=True,exist_ok=True)
corpus,manifest=fetch_corpus(ROOT);frames=load_raw(ROOT);bank=json.load(open(ROOT/'data/noise/bank.json'));audit=json.load(open(OUT/'noise_audit.json'))
gen=NoiseGenerator(bank);generated=generate_view(frames['validation'],gen,101)
reference=[r for row in corpus['validation'].itertuples() for r in align_pairs(row.original,row.normalized)]
synthetic=[e for edits in generated.edits for e in edits]
kind_order=['abbreviation','diacritics','elongation','spelling_variant']
a=collections.Counter(r['kind'] for r in reference);b=collections.Counter(r['kind'] for r in synthetic)
p=np.array([a[k] for k in kind_order],dtype=float);p/=p.sum();q=np.array([b[k] for k in kind_order],dtype=float);q/=q.sum()
js=float(jensenshannon(p,q,base=2)**2)
mix=pd.DataFrame({'kind':kind_order,'human_aligned_validation':p,'generated_vihsd_validation':q});mix.to_csv(OUT/'type_mix_comparison.csv',index=False)
# The annotation's unchanged tokens expose ambiguity not captured by changed-span purity.
unchanged=collections.Counter()
for row in corpus['train'].itertuples():
 x=normalize_text(row.original).casefold().split();y=normalize_text(row.normalized).casefold().split()
 for tag,i,j,k,l in difflib.SequenceMatcher(a=x,b=y,autojunk=False).get_opcodes():
  if tag=='equal':unchanged.update(x[i:j])
flags=[]
for r in bank['rows']:
 if r['partition']=='train' and unchanged[r['variant']]>0:
  flags.append({'mapping_id':r['mapping_id'],'canonical':r['canonical'],'variant':r['variant'],'changed_count':r['count'],'unchanged_token_count':unchanged[r['variant']]})
pd.DataFrame(flags).to_csv(OUT/'context_ambiguity_flags.csv',index=False)
summary={'human_reference_aligned_events':len(reference),'generated_events':len(synthetic),'event_type_js_divergence_bits':js,
         'context_ambiguity_flagged_train_mappings':len(flags),'interpretation':'Diagnostic only; no numeric score certifies semantic/toxic-label preservation. Source is selected normalization corpus, not representative UGC.'}
dump_json(OUT/'fidelity_diagnostics.json',summary)
sns.set_theme(style='whitegrid',font='DejaVu Sans')
fig,ax=plt.subplots(figsize=(11,4.8));mix.set_index('kind').plot.bar(ax=ax,rot=12,color=['#3979a8','#d79c34']);ax.set(title='Phân phối loại edit: nguồn human vs mẫu sinh (khác domain)',xlabel='',ylabel='Tỷ lệ sự kiện');fig.tight_layout();fig.savefig(OUT/'01_type_mix.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.6));lengths=[len(t.split()) for t in corpus['train'].original];target=[len(t.split()) for t in frames['train'].text_raw];sns.ecdfplot(x=lengths,ax=axes[0],label='ViLexNorm train');sns.ecdfplot(x=target,ax=axes[0],label='ViHSD train');axes[0].set(xlim=(0,80),xlabel='Đơn vị whitespace',ylabel='Tỷ lệ tích lũy',title='Lệch độ dài giữa nguồn và domain đích');axes[0].legend();sns.histplot(generated.loc[generated.changed,'actual_rate'],bins=20,ax=axes[1],color='#3979a8');axes[1].set(title='Tỷ lệ thêm edit thực tế — validation có đổi',xlabel='Đơn vị nguồn bị tác động / tổng đơn vị',ylabel='Số mẫu');fig.tight_layout();fig.savefig(OUT/'02_domain_and_rates.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.6));parts=pd.Series(audit['bank_counts']);parts.plot.bar(ax=ax,color=['#bd5066','#3979a8'],rot=0);ax.set(title='Mapping có bằng chứng human-normalization',xlabel='Partition',ylabel='Số mapping');fig.tight_layout();fig.savefig(OUT/'03_bank_partitions.png',dpi=150);plt.close(fig)
examples=generated[generated.changed].sample(min(30,int(generated.changed.sum())),random_state=42).copy();examples['edits']=examples.edits.map(json.dumps);examples.to_csv(OUT/'validation_examples.csv',index=False)
text=f'''# Nghiên cứu và EDA noise — dựa trên biến thể có thật

## Nguồn và cách sử dụng

1. **ViLexNorm, EACL 2024**: cặp original/normalized do người gán nhãn, https://aclanthology.org/2024.eacl-long.85/ ; repo tác giả https://github.com/ngxtnhi/ViLexNorm . Dùng train để trích mapping và tần suất; dev/test để báo độ bao phủ và độ lệch, không fit lại bằng test.
2. **ViSoLex, COLING 2025**: https://aclanthology.org/2025.coling-demos.18/ . Từ điển đi kèm được trích từ ViLexNorm nên không xem là nguồn xác nhận độc lập, không nhập toàn bộ dictionary để làm lộ held-out variants.
3. **ViSoBERT, EMNLP 2023**: https://aclanthology.org/2023.emnlp-main.315/ . Tham khảo UGC và emoji; không đồng nhất UNK với segmentation sai.
4. **Miyato et al., ICLR 2017**: https://research.google/pubs/adversarial-training-methods-for-semi-supervised-text-classification/ . Tham khảo perturbation trên embeddings; code dùng biến thể FGM embedding-weight được mô tả riêng trong protocol.

ViLexNorm có chọn lọc câu chứa NSW, đã lọc câu ngắn và loại emoji khi xây corpus. Do đó số liệu của corpus không phải ước lượng tỷ lệ toàn bộ comment mạng xã hội có noise. Bảo toàn emoji của ViHSD bằng preprocessing UTF-8, không suy ra cách thêm emoji từ corpus đã bỏ emoji.

## Kết quả trích xuất

- Revision nguồn: `{manifest['revision']}`; split đọc trực tiếp: train {len(corpus['train']):,}, dev {len(corpus['validation']):,}, test {len(corpus['test']):,}.
- Training bank: **{audit['bank_counts'].get('train',0)}** mapping; held-out bank: **{audit['bank_counts'].get('unseen',0)}** mapping.
- Mỗi edit có mapping ID và source sentence IDs; không tạo biến thể chưa từng thấy trong nguồn.
- Held-out surfaces không có trong training bank và không xuất hiện nguyên dạng trong raw ViHSD train sau NFC/casefold với boundary. Đây là kiểm tra bề mặt, không chứng minh model chưa gặp trong pretraining.
- Mẫu validation được sửa: **{generated.changed.sum()}/{len(generated)} ({generated.changed.mean():.2%})**. Mẫu không đủ điều kiện giữ nguyên và được tính riêng.
- JSD loại edit so với human-aligned dev: **{js:.4f} bits**. Đây là mức lệch đo được, không phải chứng nhận naturalness.
- Có **{len(flags)}** mapping train có variant cũng xuất hiện như token không đổi trong annotation. Purity chỉ đo trong các span được sửa; cần ngữ cảnh để hiểu các từ đồng hình. Danh sách được công khai, không coi dictionary là bộ phân giải nghĩa.

## Những kết luận được phép và chưa được phép

- Có thể nói: biến thể được lấy từ human-normalized social-media pairs; có provenance, weight theo số lần quan sát, seed cố định và held-out variants.
- Chưa thể nói: mọi câu mới đều giống người thật hoặc đã được human xác nhận giữ nhãn toxic. Dataset normalization không cung cấp nhãn toxic cho các câu được tổng hợp.
- Empirical-event thêm một sự kiện lexical trên mỗi câu đủ điều kiện; đây là mô hình đơn giản, không tái hiện toàn bộ tương quan phong cách hoặc mật độ lỗi tự nhiên.
- Mức 10/20/30/50% là controlled stress. Luôn báo tỷ lệ thực tế và coverage, không gọi 50% là tỷ lệ thực tế của người dùng.
- Đánh giá bắt buộc có ViHSD gốc với nhãn thật và natural cohorts từ detector đã khóa. Cohort detector là heuristic, không phải gold noise annotation.
- Không dùng test F1 để sửa generator hoặc quyết định độ khó; không dùng LLM tự gán làm gold.

## Đầu ra kỹ thuật

`noise_audit.json`, `fidelity_diagnostics.json`, `type_mix_comparison.csv`, `context_ambiguity_flags.csv`, `validation_examples.csv`, bank và aligned evidence trong data/noise. Các file có raw text là artifact nghiên cứu, không tự publish công khai.
'''
for image in ['01_type_mix.png','02_domain_and_rates.png','03_bank_partitions.png']:text+=f'\n![{image}]({image})\n'
(OUT/'NOISE_RESEARCH.md').write_text(text)
print(json.dumps(summary,indent=2))
