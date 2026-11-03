"""Post-freeze external linguistic audit, never used to tune current model/generator."""
import collections,json,re,urllib.request
from pathlib import Path
import pandas as pd
from vihsd.data import dump_json,sha256_bytes,load_raw,normalize_text
from vihsd.noise import align_pairs,NoiseGenerator
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/noise';DATA=ROOT/'data/external/small_lexnorm_vihsd';DATA.mkdir(parents=True,exist_ok=True)
REV='621b43c6e682df71214fec8c44836dd7a84ada52';rows=[];manifest={'revision':REV,'role':'post-freeze linguistic audit only; no current generator/model fitting','source_paper':'https://arxiv.org/abs/2311.06851','files':{}}
for name in ['train_data.json','test_data.json']:
 path=DATA/name;url=f'https://raw.githubusercontent.com/AnhHoang0529/Small-LexNormViHSD/{REV}/{name}'
 if not path.exists():
  with urllib.request.urlopen(url,timeout=120) as response:path.write_bytes(response.read())
 records=json.loads(path.read_text());manifest['files'][name]={'url':url,'sha256':sha256_bytes(path.read_bytes()),'rows':len(records)}
 for r in records:
  original=' '.join(r['input']);normalized=' '.join(r['output'])
  for p in align_pairs(original,normalized):rows.append({**p,'source_id':f'smalllex:{name}:{r["id"]}'})
dump_json(DATA/'manifest.json',manifest)
bank=json.load(open(ROOT/'data/noise/bank.json'));train_pairs={(r['canonical'],r['variant']) for r in bank['rows'] if r['partition']=='train'}
counts=collections.Counter((r['canonical'],r['variant']) for r in rows)
summary={'source_comments':sum(x['rows'] for x in manifest['files'].values()),'aligned_events':len(rows),'unique_aligned_pairs':len(counts),'event_coverage_by_frozen_train_bank':sum((r['canonical'],r['variant']) in train_pairs for r in rows)/len(rows),'kind_counts':dict(collections.Counter(r['kind'] for r in rows)),'used_to_fit_current_generator':False,'used_to_fit_or_select_current_models':False,'source_origin':'human normalization of selected ViHSD training comments; not an independent toxic test set'}
pd.DataFrame([{'canonical':c,'variant':v,'observed_count':n,'in_current_train_bank':(c,v) in train_pairs} for (c,v),n in counts.most_common()]).to_csv(OUT/'in_domain_pair_coverage.csv',index=False)
variants=sorted({r['variant'] for r in rows},key=lambda x:-len(x));pattern=re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(v) for v in variants)+r')(?!\w)',re.IGNORECASE)
raw=load_raw(ROOT)['test'];markers=raw.text_raw.map(lambda t:len(list(pattern.finditer(normalize_text(t)))))
pd.DataFrame({'sample_id':raw.sample_id,'label':raw.label,'in_domain_nsw_markers':markers}).to_csv(OUT/'test_in_domain_natural_cohorts.csv',index=False)
summary['test_marker_rows']=int((markers>0).sum());summary['test_marker_rate']=float((markers>0).mean());dump_json(OUT/'in_domain_audit.json',summary)
text=f'''# Bổ sung audit bằng Small-LexNormViHSD

Nguồn chính thức: https://github.com/AnhHoang0529/Small-LexNormViHSD ; paper https://arxiv.org/abs/2311.06851 . Bài báo mô tả việc lấy comment từ ViHSD train để human-normalize. Không coi đây là một bộ toxic test độc lập.

Nguồn được khảo sát sau khi đã đóng băng generator và bắt đầu training; **không dùng để sửa generator hoặc chọn model trong các run này**. Mục đích là kiểm tra độ bao phủ domain và phân tích bổ sung có nhãn toxic gốc.

- Đọc {summary['source_comments']:,} cặp input/output từ revision `{REV}`.
- Qua cùng bộ lọc alignment bảo thủ: {len(rows):,} lexical events, {len(counts):,} unique pairs.
- Training bank đã khóa bao phủ **{summary['event_coverage_by_frozen_train_bank']:.2%}** các event giữ được. Đây là giới hạn thực tế về domain coverage, không tự thêm rule sau khi xem test để nâng điểm.
- Marker từ các NSW do người gán nhãn xuất hiện trong **{summary['test_marker_rows']:,}/{len(raw):,} ({summary['test_marker_rate']:.2%})** mẫu test ViHSD. Đây vẫn là detector bề mặt, không phải nhãn noise gold của từng câu test.
- Các normalization pair này không có nhãn toxic được gán lại cho normalized text; chỉ dùng nhãn toxic gốc trên comment nguyên bản khi phân tích cohort.

Định hướng phiên bản sau: đưa nguồn in-domain này vào protocol mới trước training; đo trùng/khác nhau giữa hai nguồn và giữ riêng annotation/context ambiguities. Không xem toàn bộ các corpus từ cùng ViHSD train là dữ liệu độc lập để tính ý nghĩa thống kê.
'''
(OUT/'IN_DOMAIN_AUDIT.md').write_text(text)
print(json.dumps(summary,indent=2))
