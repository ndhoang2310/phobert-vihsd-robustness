"""Recover verified training/evaluation outputs after a missing optional reference.

This does NOT turn a failed training into success: it requires all five epochs,
selected model weights, all 11 primary/baseline/FGM1 profiles and exact terminal
FileNotFoundError referring ONLY to the unavailable pilot augmentation source.
The original platform ERROR and reconstruction provenance remain explicit.
"""
import json,hashlib,shutil,zipfile
from pathlib import Path
import numpy as np
import nbformat
from kaggle.api.kaggle_api_extended import KaggleApi
from safetensors import safe_open
from vihsd.data import load_raw,dump_json,sha256_bytes
from verify_robustness import verify_job
ROOT=Path(__file__).resolve().parents[1]
condition='augmentation-v2';ref=f'ndhoang2310/vihsd-real-noise-{condition}/1'
api=KaggleApi();api.authenticate();status=api.kernels_status(ref)
assert 'ERROR' in str(status.status),f'Expected final optional-reference error, got {status.status}'
log=api.kernels_logs(ref)
assert 'Missing reference checkpoint: augmentation' in log,'Different failure requires investigation'
folder=ROOT/f'artifacts/robustness/{condition}';folder.mkdir(parents=True,exist_ok=True)
(folder/'original_training.log').write_text(log)
pattern=r'(reports/|robustness/|data/noise/bank\.json|data/tokenized/token_diagnostics\.json|data/noise_views/.*/view\.parquet|runs/.*/(?:history|selection|config|environment|training_summary)\.json|runs/.*/best/)'
files,token=api.kernels_output(ref,str(folder),file_pattern=pattern,force=True,quiet=True,page_size=200)
assert not token
root=folder/'robustness_project'
models=[condition,'baseline','fgm-eps1-reference']
expected={'original'}|{f'empirical_{part}_s{seed}' for part in ['train','unseen'] for seed in [201,202,203]}|{f'stress_{n}_s201' for n in [10,20,30,50]}
for model in models:
 summary=json.loads((root/f'robustness/{model}/summary.json').read_text());assert set(summary)==expected
run_name='phobert-human-augmentation-v2-seed42';weights=root/f'runs/{run_name}/best/model.safetensors'
with safe_open(weights,framework='numpy') as tensors:
 names=list(tensors.keys());assert len(names)>100
 for key in names:assert np.isfinite(tensors.get_tensor(key)).all(),key
weight_sha=hashlib.file_digest(weights.open('rb'),'sha256').hexdigest()
bank=json.loads((root/'data/noise/bank.json').read_text())
complete={'condition':condition,'run_name':run_name,'bank_sha256':sha256_bytes(json.dumps(bank,sort_keys=True).encode()),'test_profiles':sorted(expected),'extra_references':['fgm-eps1-reference'],'noise_version':bank['version'],'status':'pending_recovery_validation','training_kernel':ref,'training_kernel_status':str(status.status),'original_failure':'Missing reference checkpoint: augmentation (optional cancelled pilot; primary evaluation already saved)','completion_provenance':'reconstructed after independently validating all completed primary outputs; original notebook did not emit a completion marker','best_model_sha256':weight_sha,'best_model_bytes':weights.stat().st_size}
dump_json(root/'ROBUSTNESS_COMPLETE.json',complete)
verified,_=verify_job(root,condition)
complete.update(status='complete_recovered',independent_verification=verified)
dump_json(root/'ROBUSTNESS_COMPLETE.json',complete)
# Ship human test labels separately for CPU-side independent verification.
load_raw(ROOT)['test'][['sample_id','label','text_raw']].to_parquet(root/'official_test_labels.parquet',index=False)
shutil.copy2(folder/'original_training.log',root/'original_training.log')
source=next((ROOT/f'artifacts/robustness/{condition}-source').glob('*.ipynb'));shutil.copy2(source,root/'original_training_source.ipynb')
upload=ROOT/'artifacts/robustness/recovery-upload';upload.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(upload/'recovery.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
 for path in sorted(root.rglob('*')):
  if path.is_file():archive.write(path,str(Path('robustness_project')/path.relative_to(root)))
# Private same-account archival storage; do not assert a new license over source data.
dump_json(upload/'dataset-metadata.json',{'id':'ndhoang2310/vihsd-robustness-v2-recovery','title':'ViHSD robustness v2 verified recovery','licenses':[{'name':'other'}],'description':'Private recovery of own training outputs. ViLexNorm portions retain CC BY-NC-SA 4.0; other source rights remain with original authors. Original training error and provenance included.'})
code='''from pathlib import Path
import json,zipfile,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
import shutil
archives=list(Path('/kaggle/input').rglob('recovery.zip'))
root=Path('/kaggle/working/robustness_project')
if archives:
    with zipfile.ZipFile(archives[0]) as z:z.extractall('/kaggle/working')
else:
    source=next(Path('/kaggle/input').rglob('ROBUSTNESS_COMPLETE.json')).parent
    shutil.copytree(source,root,dirs_exist_ok=True)
done=json.loads((root/'ROBUSTNESS_COMPLETE.json').read_text())
assert done['status']=='complete_recovered'
weights=root/'runs'/done['run_name']/'best/model.safetensors'
assert hashlib.file_digest(weights.open('rb'),'sha256').hexdigest()==done['best_model_sha256']
gold=pd.read_parquet(root/'official_test_labels.parquet').set_index('sample_id')
rows=[]
for model in [done['condition'],'baseline']+done['extra_references']:
 summary=json.loads((root/'robustness'/model/'summary.json').read_text())
 assert len(summary)==11
 for profile,m in summary.items():
  p=pd.read_parquet(root/'robustness'/model/f'{profile}_predictions.parquet')
  assert len(p)==6680 and p.sample_id.is_unique
  assert np.array_equal(p.label,gold.loc[p.sample_id].label)
  prob=p[['p_clean','p_offensive','p_hate']].to_numpy()
  assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-5)
  assert np.array_equal(prob.argmax(1),p.prediction)
  f1=f1_score(p.label,p.prediction,labels=[0,1,2],average='macro',zero_division=0)
  assert abs(f1-m['macro_f1'])<1e-12
  assert abs(accuracy_score(p.label,p.prediction)-m['accuracy'])<1e-12
  assert confusion_matrix(p.label,p.prediction,labels=[0,1,2]).tolist()==m['confusion_matrix']
  rows.append({'model':model,'profile':profile,'macro_f1':f1,'accuracy':m['accuracy']})
result={'passed':True,'profiles_recomputed':len(rows),'training_occurred_in':done['training_kernel'],'training_platform_status':done['training_kernel_status'],'recovery':'CPU verification and archival export only; no retraining','model_sha256':done['best_model_sha256']}
(root/'KAGGLE_RECOVERY_VERIFIED.json').write_text(json.dumps(result,indent=2))
display(pd.DataFrame(rows))
print(json.dumps(result,indent=2))
'''
notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('''# ViHSD v2 — verified recovery of completed training

The GPU training ran for **five epochs** in `ndhoang2310/vihsd-real-noise-augmentation-v2/1`. It saved all primary, baseline and FGM epsilon1 evaluations, then failed when loading an unavailable **optional pilot augmentation reference**. The original platform status remains ERROR. This CPU notebook independently verifies and exports the completed outputs; it does not retrain or conceal the original error. Weight checksum, original source/log, five-epoch history, predictions and reconstruction provenance are included. Synthetic toxic labels remain unverified by humans.
'''),nbformat.v4.new_code_cell(code)])
notebook.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
compile(code,'recovery','exec');dest=ROOT/'kaggle/robustness-augmentation-v2-recovery';dest.mkdir(parents=True,exist_ok=True)
nbformat.write(notebook,dest/'recovery.ipynb')
dump_json(dest/'kernel-metadata.json',{'id':'ndhoang2310/vihsd-real-noise-augmentation-v2','title':'ViHSD Real Noise Augmentation-V2','code_file':'recovery.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':False,'enable_internet':False,'dataset_sources':['ndhoang2310/vihsd-robustness-v2-recovery'],'kernel_sources':[],'competition_sources':[]})
state=json.loads((ROOT/'reports/ROBUSTNESS_RUNS.json').read_text());state['jobs'][condition].update(status='ERROR_OPTIONAL_REFERENCE',training_complete=True,primary_evaluation_complete=True,verified=True,verification=verified,recovery_kernel='ndhoang2310/vihsd-real-noise-augmentation-v2',recovery_version_pending=True);dump_json(ROOT/'reports/ROBUSTNESS_RUNS.json',state)
print(json.dumps({'verified':verified,'upload':str(upload),'recovery_notebook':str(dest),'weight_sha256':weight_sha},indent=2))
