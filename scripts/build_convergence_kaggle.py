"""Freeze all nine private training notebooks, with zero checkpoint-input dependencies."""
import argparse,hashlib,json
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[1]


def build(username):
    if not username or '/' in username:raise ValueError('A Kaggle username is required')
    protocol=json.loads((ROOT/'configs/convergence.json').read_text())
    bank=json.loads((ROOT/'data/noise_v2/bank.json').read_text())
    bank_sha=hashlib.sha256(json.dumps(bank,sort_keys=True).encode()).hexdigest()
    assert bank_sha==protocol['noise_bank_sha256']
    sources={str(p.relative_to(ROOT/'src')):p.read_text() for p in sorted((ROOT/'src/vihsd').glob('*.py'))}
    source_sha=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    bootstrap=nbformat.read(ROOT/'notebooks/vihsd_phobert_kaggle.ipynb',as_version=4).cells[2].source.replace('"vihsd_project"','"convergence_project"')
    description=(ROOT/'reports/CONVERGENCE_PROTOCOL.md').read_text()
    jobs=[]
    # Finish matched seed blocks, rather than all seeds of the faster condition first.
    for seed in protocol['training_seeds']:
        for condition in protocol['conditions']:
            job=f'{condition}-s{seed}';folder=ROOT/f'kaggle/convergence-{job}';folder.mkdir(parents=True,exist_ok=True)
            config={**protocol,'seed':seed}
            snapshot='SOURCE_SNAPSHOT = '+repr(sources)+'\nBANK = '+repr(bank)+'\nCONFIG = '+repr(config)+'\n'
            snapshot+='''for relative,content in SOURCE_SNAPSHOT.items():
    path=ROOT/'src'/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(content,encoding='utf-8')
sys.path.insert(0,str(ROOT/'src'))
'''
            snapshot+=f'print("SOURCE_SHA256",{source_sha!r})\n'
            cells=[nbformat.v4.new_markdown_cell(f'# ViHSD convergence — {condition}, seed {seed}\n\n'+description),nbformat.v4.new_code_cell(bootstrap),nbformat.v4.new_code_cell(snapshot),nbformat.v4.new_code_cell(f'from vihsd.convergence import run_training\nSUMMARY=run_training(ROOT,CONFIG,{condition!r},BANK)\nprint((ROOT/"TRAINING_COMPLETE.json").read_text())')]
            notebook=nbformat.v4.new_notebook(cells=cells);notebook.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'};notebook.metadata.update(source_sha256=source_sha,noise_bank_sha256=bank_sha)
            for cell in cells:
                if cell.cell_type=='code':compile(cell.source,'notebook','exec')
            nbformat.validate(notebook);nbformat.write(notebook,folder/'train.ipynb')
            slug=f'vihsd-convergence-{job}'
            metadata={'id':f'{username}/{slug}','title':f'ViHSD Convergence {condition.title()} S{seed}','code_file':'train.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'dataset_sources':[],'kernel_sources':[],'competition_sources':[]}
            (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
            jobs.append({'job':job,'condition':condition,'seed':seed,'kernel':metadata['id'],'folder':str(folder),'status':'PREPARED','source_sha256':source_sha,'noise_bank_sha256':bank_sha})
    manifest={'protocol':protocol,'source_sha256':source_sha,'noise_bank_sha256':bank_sha,'username':username,'jobs':jobs,'test_evaluated':False}
    destination=ROOT/'reports/CONVERGENCE_RUNS.json'
    if destination.exists():
        old=json.loads(destination.read_text())
        if any(j['status']!='PREPARED' for j in old['jobs']):raise RuntimeError('An existing launched campaign must not be overwritten')
    destination.write_text(json.dumps(manifest,indent=2));print(json.dumps({'jobs_prepared':len(jobs),'source_sha256':source_sha,'username':username},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--username',required=True);build(p.parse_args().username)
