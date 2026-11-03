"""Recompute evaluation metrics and replay frozen source-backed edits independently."""
import json,re
from pathlib import Path
import numpy as np
import pandas as pd
from vihsd.data import load_raw,normalize_text
from vihsd.full20_eval import PROFILES,view_sha
from vihsd.training import metrics
ROOT=Path(__file__).resolve().parents[1]

def verify(root,group,state):
 root=Path(root);done=json.loads((root/'EVALUATION_COMPLETE.json').read_text())
 assert done['protocol']=='full20-evaluation-v1' and done['training_performed'] is False
 assert set(done['jobs'])=={j['job'] for j in group['jobs']} and set(done['profiles'])==set(PROFILES)
 assert done['view_hashes']==state['view_hashes']
 raw=load_raw(ROOT)['test'].set_index('sample_id')
 bank=json.loads((root/'data/noise/bank.json').read_text())
 import hashlib
 assert hashlib.sha256(json.dumps(bank,sort_keys=True).encode()).hexdigest()==state['bank_sha256']==done['bank_sha256']
 mapping={r['mapping_id']:r for r in bank['rows']};views={};events=0
 for profile in PROFILES[1:]:
  view=pd.read_parquet(root/'data/noise_views'/profile/'view.parquet');assert len(view)==6680 and view.sample_id.is_unique
  assert view_sha(view)==state['view_hashes'][profile]
  for row in view.itertuples():
   assert row.text_original==normalize_text(raw.loc[row.sample_id].text_raw) and row.label==raw.loc[row.sample_id].label
   edits=json.loads(row.edits) if isinstance(row.edits,str) else row.edits
   assert bool(edits)==bool(row.changed) and not row.label_preservation_verified
   text=row.text_original;spans=sorted(edits,key=lambda e:e['start'])
   assert all(a['end']<=b['start'] for a,b in zip(spans,spans[1:]))
   for edit in reversed(spans):
    support=mapping[edit['mapping_id']]
    assert support['partition']==('unseen' if profile.startswith('empirical_unseen') else 'train')
    assert edit['source_ids']==support['source_ids'] and edit['kind']==support['kind']
    assert normalize_text(edit['original']).casefold()==support['canonical'] and edit['replacement'].casefold()==support['variant']
    assert text[edit['start']:edit['end']]==edit['original']
    text=text[:edit['start']]+edit['replacement']+text[edit['end']:];events+=1
   assert text==row.text_noisy
   assert [c for c in text if ord(c)>0xffff]==[c for c in row.text_original if ord(c)>0xffff]
  views[profile]=view
 def check_predictions(p,labels):
  assert p.sample_id.is_unique and len(p)==len(labels)
  assert np.array_equal(p.label,labels)
  probs=p[['p_clean','p_offensive','p_hate']].to_numpy()
  assert np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all() and np.allclose(probs.sum(1),1,atol=1e-5)
  assert np.array_equal(probs.argmax(1),p.prediction)
  return probs,metrics(p.label,p.prediction)
 results=[]
 for j in group['jobs']:
  gate=root/'validation_gate'/j['job'];audit=json.loads((gate/'audit.json').read_text())
  assert audit==done['validation_gates'][j['job']] and audit['passed'] and audit['selected_epoch']==j['best_epoch']
  assert audit['kernel_input']==j['kernel']+'/'+str(j['version']) and re.fullmatch('[0-9a-f]{64}',audit['model_sha256'])
  oldroot=ROOT/'artifacts/full20'/j['job']/'full20_project'
  olddone=json.loads((oldroot/'TRAINING_COMPLETE.json').read_text())
  reference=pd.read_parquet(oldroot/'runs'/olddone['run_name']/'validation'/f"epoch_{j['best_epoch']}.parquet")
  current=pd.read_parquet(gate/'predictions.parquet');assert np.array_equal(current.sample_id,reference.sample_id)
  prob,score=check_predictions(current,reference.label)
  assert np.array_equal(current.prediction,reference.prediction)
  assert abs(score['macro_f1']-j['best_validation_macro_f1'])<1e-12
  difference=float(np.abs(prob-reference[['p_clean','p_offensive','p_hate']].to_numpy()).max())
  assert difference<2e-5 and abs(difference-audit['validation_prob_max_difference'])<1e-12
  folder=root/'robustness'/j['job'];summary=json.loads((folder/'summary.json').read_text());assert set(summary)==set(PROFILES)
  for profile in PROFILES:
   p=pd.read_parquet(folder/f'{profile}_predictions.parquet');assert len(p)==6680
   prob,score=check_predictions(p,raw.loc[p.sample_id].label)
   r=summary[profile]
   assert abs(score['macro_f1']-r['macro_f1'])<1e-12 and abs(score['accuracy']-r['accuracy'])<1e-12
   assert score['confusion_matrix']==r['confusion_matrix']
   if profile!='original':
    view=views[profile].set_index('sample_id').loc[p.sample_id];mask=view.changed.to_numpy()
    assert abs(r['coverage']-float(mask.mean()))<1e-12 and r['changed_rows']==int(mask.sum())
    assert abs(r['actual_rate_mean']-float(view.actual_rate.mean()))<1e-12
    assert abs(r['changed_only']['macro_f1']-metrics(p.label.to_numpy()[mask],p.prediction.to_numpy()[mask])['macro_f1'])<1e-12
   results.append({'job':j['job'],'condition':j['condition'],'seed':j['seed'],'profile':profile,'macro_f1':score['macro_f1'],'accuracy':score['accuracy']})
 return {'passed':True,'models':len(group['jobs']),'profiles_verified':len(results),'replayed_events':events,'results':results}
