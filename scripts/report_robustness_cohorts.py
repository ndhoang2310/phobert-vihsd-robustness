"""Descriptive diagnostics on original human-labelled test text; no model selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score
from vihsd.data import load_raw,normalize_text
from vihsd.noise import NoiseGenerator
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/robustness';OUT.mkdir(parents=True,exist_ok=True)
raw=load_raw(ROOT);test=raw['test'].copy();train_texts={normalize_text(t).casefold() for t in raw['train'].text_raw}
test['overlap_train']=test.text_raw.map(lambda t:normalize_text(t).casefold() in train_texts)
rows=[]
for condition in ['augmentation-v2','fgm-small']:
 root=ROOT/f'artifacts/robustness/{condition}/robustness_project';done=json.loads((root/'ROBUSTNESS_COMPLETE.json').read_text())
 bank=json.loads((root/'data/noise/bank.json').read_text());detector=NoiseGenerator(bank)
 test['nsw_marker']=test.text_raw.map(detector.detect)>0
 for model in [condition,'baseline']+done.get('extra_references',[]):
  if any(r['model']==model for r in rows):continue
  p=pd.read_parquet(root/f'robustness/{model}/original_predictions.parquet').set_index('sample_id').loc[test.sample_id]
  for cohort,mask in [('all',np.ones(len(test),bool)),('no_train_overlap',~test.overlap_train),('train_overlap',test.overlap_train),('nsw_marker',test.nsw_marker),('no_nsw_marker',~test.nsw_marker)]:
   y=test.label.to_numpy()[mask];pred=p.prediction.to_numpy()[mask]
   rows.append({'model':model,'cohort':cohort,'n':len(y),'macro_f1':f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0),'accuracy':accuracy_score(y,pred),'clean':int((y==0).sum()),'offensive':int((y==1).sum()),'hate':int((y==2).sum())})
table=pd.DataFrame(rows);table.to_csv(OUT/'original_test_cohorts.csv',index=False)
text='''# Chẩn đoán trên test gốc có nhãn thật

Các model được đánh giá trên cùng comment gốc, không thêm noise. `nsw_marker` là exact surface detector của bank train v2 đã khóa, không phải gold nhãn noise; detector có giới hạn case/context và chỉ nhận diện tập biến thể có trong bank. Nhóm có/không marker có thể khác chủ đề, nhãn và độ dài. Không dùng cohort metrics để chọn model hoặc suy diễn tác động nhân quả của noise.

`no_train_overlap` loại exact text match sau chuẩn hóa NFC/whitespace/casefold với toàn bộ raw train. Đây là kiểm tra mô tả bổ sung, không thay thế official split và không loại gần trùng. Class support xuất trong CSV để đọc cùng Macro-F1.

| Model | Cohort | N | Macro-F1 | Accuracy |
|---|---|---:|---:|---:|
'''
for r in rows:text+=f"| {r['model']} | {r['cohort']} | {r['n']} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} |\n"
(OUT/'ORIGINAL_TEST_COHORTS.md').write_text(text);print(table.to_string(index=False))
