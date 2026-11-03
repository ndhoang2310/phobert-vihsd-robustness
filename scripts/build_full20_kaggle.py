"""Build a separate full-budget campaign without overwriting older experiments."""
import hashlib,json
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[1]
ASSIGNMENTS={
 'clean-s42':('nkhung','danielngnkh'),
 'fgm-s42':('phu','phnguyncng243'),
 'augmentation-s42':('linh','linh2709'),
 'clean-s123':('hoang2','ndhoangg'),
 'fgm-s123':('hoang1','ndhoang2310'),
 'augmentation-s123':('hoang1','ndhoang2310'),
 'clean-s2026':('phu','phnguyncng243'),
 'fgm-s2026':('linh','linh2709'),
 'augmentation-s2026':('nkhung','danielngnkh'),
}
EXTERNAL={
 'nkhung':['danielngnkh/p3-ddl4-ex2-dsp-repfull'],
 'phu':['phnguyncng243/p3-ddl4-ex1-dsp-lite'],
 'linh':['linh2709/p3-ddl4-ex4-dsp-dual'],
 'hoang2':['ndhoangg/p3-ddl4-ex3-dsp-ffm'],
 'hoang1':[],
}

def build():
 destination=ROOT/'reports/FULL20_RUNS.json'
 if destination.exists():raise FileExistsError('Full20 campaign already exists; never rebuild a launched snapshot')
 protocol=json.loads((ROOT/'configs/full20.json').read_text())
 assert protocol['epochs']==20 and protocol['early_stopping_patience'] is None
 bank=json.loads((ROOT/'data/noise_v2/bank.json').read_text())
 bank_sha=hashlib.sha256(json.dumps(bank,sort_keys=True).encode()).hexdigest()
 assert bank_sha==protocol['noise_bank_sha256']
 sources={str(p.relative_to(ROOT/'src')):p.read_text() for p in sorted((ROOT/'src/vihsd').glob('*.py'))}
 source_sha=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
 bootstrap=nbformat.read(ROOT/'notebooks/vihsd_phobert_kaggle.ipynb',as_version=4).cells[2].source.replace('"vihsd_project"','"full20_project"')
 description=(ROOT/'reports/FULL20_PROTOCOL.md').read_text();jobs=[]
 for seed in protocol['training_seeds']:
  for condition in protocol['conditions']:
   job=f'{condition}-s{seed}';alias,owner=ASSIGNMENTS[job]
   folder=ROOT/f'kaggle/full20-{job}';folder.mkdir(parents=True,exist_ok=True)
   snapshot='SOURCE_SNAPSHOT = '+repr(sources)+'\nBANK = '+repr(bank)+'\nCONFIG = '+repr({**protocol,'seed':seed})+'\n'
   snapshot+='for relative,content in SOURCE_SNAPSHOT.items():\n    path=ROOT/"src"/relative\n    path.parent.mkdir(parents=True,exist_ok=True)\n    path.write_text(content,encoding="utf-8")\nsys.path.insert(0,str(ROOT/"src"))\n'
   snapshot+=f'print("SOURCE_SHA256",{source_sha!r})\n'
   cells=[nbformat.v4.new_markdown_cell(f'# ViHSD Full20 — {condition}, seed {seed}\n\n'+description),nbformat.v4.new_code_cell(bootstrap),nbformat.v4.new_code_cell(snapshot),nbformat.v4.new_code_cell(f'from vihsd.convergence import run_training\nSUMMARY=run_training(ROOT,CONFIG,{condition!r},BANK)\nassert SUMMARY["epochs_completed"]==20 and SUMMARY["stop_reason"]=="max_epochs"\nprint((ROOT/"TRAINING_COMPLETE.json").read_text())')]
   notebook=nbformat.v4.new_notebook(cells=cells)
   notebook.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
   notebook.metadata.update(source_sha256=source_sha,noise_bank_sha256=bank_sha)
   for cell in cells:
    if cell.cell_type=='code':compile(cell.source,'full20-notebook','exec')
   nbformat.validate(notebook);nbformat.write(notebook,folder/'train.ipynb')
   metadata={'id':f'{owner}/vihsd-full20-{job}','title':f'ViHSD Full20 {condition.title()} S{seed}','code_file':'train.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'dataset_sources':[],'kernel_sources':[],'competition_sources':[]}
   (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
   jobs.append({'job':job,'condition':condition,'seed':seed,'kernel':metadata['id'],'bridge_alias':alias,'folder':str(folder),'status':'PREPARED','source_sha256':source_sha,'noise_bank_sha256':bank_sha,'session_timeout_seconds':{'clean':7200,'fgm':14400,'augmentation':9600}[condition]})
 state={'protocol':protocol,'source_sha256':source_sha,'noise_bank_sha256':bank_sha,'jobs':jobs,'external_kernels':EXTERNAL,'excluded_accounts':{'huu':'user reports no GPU access'},'campaign_status':'PREPARED','test_evaluated':False,'credential_transport':'existing kaggle-account bridge','max_sessions_per_account':2,'account_allocation_authorized_by':'user requested allocation across available accounts on 2026-10-31','quota_status':'Unknown remaining hours; GPU submissions and run status provide server-side availability checks'}
 destination.write_text(json.dumps(state,indent=2)+'\n')
 print(json.dumps({'jobs_prepared':len(jobs),'source_sha256':source_sha,'assignments':ASSIGNMENTS},indent=2))

if __name__=='__main__':build()
