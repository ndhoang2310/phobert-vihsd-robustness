"""Report all matched training seeds after independent verification of 99 profiles."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from vihsd.data import load_raw,normalize_text
from vihsd.training import metrics
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/full20_evaluation'
CATEGORIES={'original':['original'],'empirical_seen':[f'empirical_train_s{s}' for s in [201,202,203]],'empirical_unseen':[f'empirical_unseen_s{s}' for s in [201,202,203]],**{f'stress_{r}':[f'stress_{r}_s201'] for r in [10,20,30,50]}}

def summarize(state):
 assert all(g['status']=='COMPLETE' and g.get('verified') for g in state['groups'])
 OUT.mkdir(parents=True,exist_ok=True);raw=load_raw(ROOT);train_keys={normalize_text(t).casefold() for t in raw['train'].text_raw};test=raw['test'];overlap=test.text_raw.map(lambda t:normalize_text(t).casefold() in train_keys)
 rows=[];categories=[];cohorts=[];shared=[];predictions={};roots={}
 first=ROOT/'artifacts/full20-evaluation'/state['groups'][0]['group']/'full20_evaluation'
 views={p:pd.read_parquet(first/'data/noise_views'/p/'view.parquet') for p in sum([CATEGORIES['empirical_seen'],CATEGORIES['empirical_unseen']],[])}
 common=set.intersection(*[set(v.loc[v.changed,'sample_id']) for v in views.values()])
 assert common
 for group in state['groups']:
  root=ROOT/'artifacts/full20-evaluation'/group['group']/'full20_evaluation'
  for j in group['jobs']:
   folder=root/'robustness'/j['job'];summary=json.loads((folder/'summary.json').read_text());roots[j['job']]=folder
   for profile,r in summary.items():
    predictions[(j['job'],profile)]=pd.read_parquet(folder/f'{profile}_predictions.parquet')
    rows.append({'job':j['job'],'condition':j['condition'],'training_seed':j['seed'],'profile':profile,'macro_f1':r['macro_f1'],'accuracy':r['accuracy'],'coverage':r.get('coverage',0.),'actual_rate_mean':r.get('actual_rate_mean',0.),'changed_only_macro_f1':r.get('changed_only',{}).get('macro_f1'),'changed_only_original_macro_f1':r.get('changed_only_original_macro_f1'),'flip_rate':r.get('prediction_flip_rate_vs_original',0.)})
   for category,profiles in CATEGORIES.items():
    values=[summary[p]['macro_f1'] for p in profiles]
    categories.append({'job':j['job'],'condition':j['condition'],'training_seed':j['seed'],'category':category,'mean_macro_f1':float(np.mean(values)),'delta_vs_own_original':float(np.mean(values)-summary['original']['macro_f1']),'noise_seed_count':len(profiles)})
    if category in ['original','empirical_seen','empirical_unseen']:
     scores=[]
     for profile in profiles:
      p=predictions[(j['job'],profile)];p=p[p.sample_id.isin(common)];assert len(p)==len(common);scores.append(metrics(p.label,p.prediction)['macro_f1'])
     shared.append({'job':j['job'],'condition':j['condition'],'training_seed':j['seed'],'category':category,'n_common_ids':len(common),'mean_macro_f1':float(np.mean(scores))})
   p=predictions[(j['job'],'original')].set_index('sample_id').loc[test.sample_id]
   for name,mask in [('all',np.ones(len(test),dtype=bool)),('train_overlap',overlap.to_numpy()),('no_train_overlap',~overlap.to_numpy())]:
    r=metrics(p.label.to_numpy()[mask],p.prediction.to_numpy()[mask]);cohorts.append({'job':j['job'],'condition':j['condition'],'training_seed':j['seed'],'cohort':name,'n':int(mask.sum()),'macro_f1':r['macro_f1'],'accuracy':r['accuracy'],**{'f1_'+c:r['per_class'][c]['f1-score'] for c in ['CLEAN','OFFENSIVE','HATE']}})
 table=pd.DataFrame(rows);assert len(table)==99
 category_table=pd.DataFrame(categories);category_table.to_csv(OUT/'per_training_seed.csv',index=False);table.to_csv(OUT/'all_profiles.csv',index=False);pd.DataFrame(shared).to_csv(OUT/'shared_eligible.csv',index=False);pd.DataFrame(cohorts).to_csv(OUT/'original_cohorts.csv',index=False)
 aggregate=category_table.groupby(['condition','category']).mean_macro_f1.agg(['mean','std','count']).reset_index();assert (aggregate['count']==3).all();aggregate.to_csv(OUT/'summary.csv',index=False)
 base=category_table[category_table.condition=='clean'][['training_seed','category','mean_macro_f1']].rename(columns={'mean_macro_f1':'clean_macro_f1'})
 paired=category_table[category_table.condition!='clean'].merge(base,on=['training_seed','category'],validate='many_to_one');paired['delta_vs_clean']=paired.mean_macro_f1-paired.clean_macro_f1;paired.to_csv(OUT/'matched_seed_deltas.csv',index=False)
 # Standalone figures: training-seed uncertainty and actual stress coverage.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4.5))
 for condition in ['clean','fgm','augmentation']:
  a=aggregate[aggregate.condition==condition].set_index('category');keys=['original','empirical_seen','empirical_unseen'];x=np.arange(3)+{'clean':-.2,'fgm':0,'augmentation':.2}[condition]
  axes[0].errorbar(x,[100*a.loc[k,'mean'] for k in keys],yerr=[100*a.loc[k,'std'] for k in keys],fmt='o',capsize=3,label=condition)
  keys=['original','stress_10','stress_20','stress_30','stress_50'];axes[1].errorbar([0,10,20,30,50],[100*a.loc[k,'mean'] for k in keys],yerr=[100*a.loc[k,'std'] for k in keys],marker='o',capsize=3,label=condition)
 axes[0].set(xticks=range(3),xticklabels=['Original','Seen','Unseen'],ylabel='Macro-F1 (%)',title='Mean ± sample SD across 3 training seeds')
 axes[1].set(xlabel='Requested stress rate (%)',ylabel='Macro-F1 (%)',title='Stress response (requested rate differs from actual)')
 for ax in axes:ax.grid(alpha=.2);ax.legend()
 fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
 lines=['# Đánh giá Full20 — đủ 9 checkpoint','', 'Đã xác minh độc lập đủ **99 model/profile evaluations**. Mỗi checkpoint đã tái lập validation predictions trước khi đánh giá test. Không training thêm, không chọn checkpoint bằng test.','', '## Kết quả chính','', 'Macro-F1 (%), trung bình ± độ lệch chuẩn mẫu qua **3 training seeds**. Với seen/unseen, lấy trung bình 3 noise seeds trong từng training run trước.','', '| Model | Test gốc | Empirical seen | Empirical unseen |','|---|---:|---:|---:|']
 for condition in ['clean','fgm','augmentation']:
  a=aggregate[aggregate.condition==condition].set_index('category');lines.append('| '+condition+' | '+' | '.join(f"{100*a.loc[k,'mean']:.2f} ± {100*a.loc[k,'std']:.2f}" for k in ['original','empirical_seen','empirical_unseen'])+' |')
 lines+=['','## Chênh lệch so với clean','', 'Chênh lệch tính trên cùng training seed. SD qua 3 seed là mô tả độ biến thiên, không phải CI hay kiểm định ý nghĩa thống kê.','', '| Model | Category | Mean delta (điểm %) | Số seed delta > 0 |','|---|---|---:|---:|']
 for (condition,category),sub in paired.groupby(['condition','category']):
  if category in ['original','empirical_seen','empirical_unseen']:lines.append(f"| {condition} | {category} | {100*sub.delta_vs_clean.mean():+.2f} | {int((sub.delta_vs_clean>0).sum())}/3 |")
 lines+=['',f'Nhóm so sánh cùng IDs đủ điều kiện trong cả 6 empirical views: **{len(common):,} câu**, xem shared_eligible.csv. Changed-only scores, coverage và tỷ lệ sửa thực tế nằm trong all_profiles.csv; không coi các câu không bị sửa là thành công robustness.','',f'Test gốc có {int(overlap.sum())}/{len(test)} câu trùng train theo NFC/khoảng trắng/casefold; báo riêng nhóm trùng và không trùng trong original_cohorts.csv. Giữ nguyên official split.','', '## Giới hạn','', 'Test và benchmark nhiễu đã được xem ở vòng trước; đây là đánh giá chẩn đoán bổ sung. Synthetic toxic labels kế thừa có lọc, chưa được người gán lại. Nguồn human normalization thiên về câu có non-standard words; không tuyên bố mô phỏng toàn bộ phân phối UGC thực. Các giới hạn phụ thuộc nguồn/duplicate đã ghi trong nghiên cứu noise v2 vẫn áp dụng. Chưa có CI về bất định training với 3 seeds; không tuyên bố ý nghĩa thống kê từ bảng mean/SD.','', 'Bảng điểm từng run/profile: full20_evaluation/all_profiles.csv. Trung bình từng run: per_training_seed.csv. Delta theo seed: matched_seed_deltas.csv. F1 từng lớp và nhóm overlap: original_cohorts.csv. Đường stress: comparison.png.']
 (ROOT/'reports/FULL20_EVALUATION_RESULTS.md').write_text('\n'.join(lines)+'\n')
 (OUT/'VERIFICATION.json').write_text(json.dumps({'passed':True,'models':9,'profiles':99,'test_examples_per_profile':6680,'training_seeds':[42,123,2026],'common_eligible_ids':len(common),'groups':[{'group':g['group'],'verification':{k:v for k,v in g['verification'].items() if k!='results'}} for g in state['groups']]},indent=2)+'\n')
 print(aggregate.to_string(index=False),flush=True)
if __name__=='__main__':summarize(json.loads((ROOT/'reports/FULL20_EVALUATION_RUNS.json').read_text()))
