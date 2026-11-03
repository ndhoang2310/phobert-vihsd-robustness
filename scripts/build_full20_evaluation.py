"""Freeze five owner-local evaluation notebooks for the nine selected checkpoints."""
import json,hashlib
from pathlib import Path
import nbformat
from vihsd.full20_eval import PROFILES,view_sha
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def build():
 target=ROOT/'reports/FULL20_EVALUATION_RUNS.json'
 if target.exists():raise FileExistsError('Never overwrite an existing evaluation campaign')
 train=json.loads((ROOT/'reports/FULL20_RUNS.json').read_text());assert train['campaign_status']=='TRAINING_COMPLETE'
 bank=json.loads((ROOT/'data/noise_v2/bank.json').read_text());protocol=train['protocol']
 assert hashlib.sha256(json.dumps(bank,sort_keys=True).encode()).hexdigest()==protocol['noise_bank_sha256']
 reference=ROOT/'artifacts/robustness/augmentation-v2/robustness_project/data/noise_views'
 hashes={p:view_sha(pd.read_parquet(reference/p/'view.parquet')) for p in PROFILES[1:]}
 (ROOT/'reports/FULL20_EVALUATION_VIEW_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n')
 source={str(p.relative_to(ROOT/'src')):p.read_text() for p in sorted((ROOT/'src/vihsd').glob('*.py'))}
 sha=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
 bootstrap=nbformat.read(ROOT/'notebooks/vihsd_phobert_kaggle.ipynb',as_version=4).cells[2].source.replace('"vihsd_project"','"full20_evaluation"')
 grouped={}
 for j in train['jobs']:
  assert j['status']=='COMPLETE' and j['verified']
  job={**j,'best_epoch':j['verification']['best_epoch'],'best_validation_macro_f1':j['verification']['best_validation_macro_f1'],'model_revision':protocol['model_revision']}
  grouped.setdefault(j['bridge_alias'],[]).append(job)
 groups=[]
 for alias,jobs in grouped.items():
  owner=jobs[0]['kernel'].split('/')[0];assert all(j['kernel'].split('/')[0]==owner for j in jobs)
  folder=ROOT/f'kaggle/full20-evaluation-{alias}';folder.mkdir(parents=True,exist_ok=True)
  snapshot='SOURCE_SNAPSHOT = '+repr(source)+'\nCONFIG = '+repr(protocol)+'\nJOBS = '+repr(jobs)+'\nBANK = '+repr(bank)+'\nEXPECTED_VIEWS = '+repr(hashes)+'\n'
  snapshot+='for relative,content in SOURCE_SNAPSHOT.items():\n    path=ROOT/"src"/relative\n    path.parent.mkdir(parents=True,exist_ok=True)\n    path.write_text(content,encoding="utf-8")\nsys.path.insert(0,str(ROOT/"src"))\n'
  snapshot+=f'print("EVALUATION_SOURCE_SHA256",{sha!r})\n'
  cells=[nbformat.v4.new_markdown_cell('# ViHSD Full20 evaluation — '+alias+'\n\n'+(ROOT/'reports/FULL20_EVALUATION_PROTOCOL.md').read_text()),nbformat.v4.new_code_cell(bootstrap),nbformat.v4.new_code_cell(snapshot),nbformat.v4.new_code_cell('from vihsd.full20_eval import run_evaluation\nRESULT=run_evaluation(ROOT,CONFIG,JOBS,BANK,EXPECTED_VIEWS)\nprint((ROOT/"EVALUATION_COMPLETE.json").read_text())')]
  nb=nbformat.v4.new_notebook(cells=cells);nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'};nb.metadata.source_sha256=sha
  for c in cells:
   if c.cell_type=='code':compile(c.source,'evaluation-notebook','exec')
  nbformat.validate(nb);nbformat.write(nb,folder/'evaluate.ipynb')
  metadata={'id':f'{owner}/vihsd-full20-evaluation-{alias}','title':f'ViHSD Full20 Evaluation {alias}','code_file':'evaluate.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'dataset_sources':[],'competition_sources':[],'kernel_sources':[j['kernel']+'/'+str(j['version']) for j in jobs]}
  (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
  groups.append({'group':alias,'bridge_alias':alias,'kernel':metadata['id'],'folder':str(folder),'status':'PREPARED','jobs':jobs,'source_sha256':sha,'session_timeout_seconds':10800})
 state={'protocol':'full20-evaluation-v1','groups':groups,'view_hashes':hashes,'bank_sha256':protocol['noise_bank_sha256'],'source_sha256':sha,'external_kernels':train['external_kernels'],'status':'PREPARED','training_performed':False,'planned_model_profiles':99}
 target.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps({'groups':len(groups),'models':sum(len(g['jobs']) for g in groups),'profiles':99,'source_sha256':sha}))
if __name__=='__main__':build()
