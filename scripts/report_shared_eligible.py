"""Same original IDs for seen/unseen descriptive comparison; no parameter selection."""
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/robustness'
root=ROOT/'artifacts/robustness/augmentation-v2/robustness_project'
views={p.parent.name:pd.read_parquet(p) for p in (root/'data/noise_views').glob('empirical_*/view.parquet')}
assert len(views)==6
common=None
for view in views.values():
 ids=set(view.loc[view.changed,'sample_id']);common=ids if common is None else common&ids
assert common
rows=[];visited=set()
for condition in ['augmentation-v2','fgm-small']:
 job=ROOT/f'artifacts/robustness/{condition}/robustness_project';done=json.loads((job/'ROBUSTNESS_COMPLETE.json').read_text())
 for model in [condition,'baseline']+done.get('extra_references',[]):
  if model in visited:continue
  visited.add(model)
  for category,profiles in [('original',['original']),('empirical_seen',[f'empirical_train_s{s}' for s in [201,202,203]]),('empirical_unseen',[f'empirical_unseen_s{s}' for s in [201,202,203]])]:
   scores=[]
   for profile in profiles:
    p=pd.read_parquet(job/f'robustness/{model}/{profile}_predictions.parquet');p=p[p.sample_id.isin(common)]
    assert len(p)==len(common)
    scores.append(f1_score(p.label,p.prediction,labels=[0,1,2],average='macro',zero_division=0))
   rows.append({'model':model,'category':category,'n_same_original_ids':len(common),'mean_macro_f1':sum(scores)/len(scores),'noise_seeds':len(scores)})
table=pd.DataFrame(rows);table.to_csv(OUT/'shared_eligible_comparison.csv',index=False)
text='''# So sánh seen/unseen trên cùng nhóm đủ điều kiện

Full-test seen và unseen có coverage khác nhau, nên không suy ra loại nào khó hơn chỉ từ hai điểm trung bình. Bảng dưới giữ cùng các original IDs được sửa trong **cả sáu empirical views**; mỗi category dùng đúng nhóm này. Việc chọn nhóm chỉ dựa vào edit eligibility, không dựa dự đoán hay nhãn. Đây là phân tích mô tả bổ sung; các câu synthetic vẫn kế thừa nhãn chưa được người gán lại, và tập con này không đại diện toàn bộ UGC. Loại/vị trí edit giữa seen/unseen vẫn có thể khác nhau.

| Model | Profile | N cùng original IDs | Mean Macro-F1 |
|---|---|---:|---:|
'''
for r in rows:text+=f"| {r['model']} | {r['category']} | {r['n_same_original_ids']} | {r['mean_macro_f1']:.4f} |\n"
(OUT/'SHARED_ELIGIBLE_COMPARISON.md').write_text(text);print(table.to_string(index=False))
