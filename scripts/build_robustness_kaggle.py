"""Build private first-seed FGM and corpus-grounded augmentation notebooks."""
import json,hashlib,nbformat
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build(condition):
    # Reuse the already-verified environment bootstrap, with a separate output root.
    original=nbformat.read(ROOT/'notebooks/vihsd_phobert_kaggle.ipynb',as_version=4)
    bootstrap=original.cells[2].source.replace('"vihsd_project"','"robustness_project"')
    sources={str(p.relative_to(ROOT/'src')):p.read_text() for p in sorted((ROOT/'src/vihsd').glob('*.py'))}
    config=json.loads((ROOT/'configs/baseline.json').read_text());config['model_revision']='01daacda68afe13d83023d16ec647239e344a1e6'
    if condition in ['fgm-small','augmentation-v2']:
        config.update(noise_version=2,evaluation_profile_mode='core')
    if condition=='augmentation-v2':
        config['extra_reference_conditions']=['fgm']
    snapshot='SOURCE_SNAPSHOT = '+repr(sources)+'\n'
    snapshot+='''for relative, content in SOURCE_SNAPSHOT.items():
    target=ROOT/'src'/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(content,encoding='utf-8')
sys.path.insert(0,str(ROOT/'src'))
'''
    snapshot+='CONFIG = '+repr(config)+'\n'
    sha=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    snapshot+=f'print("SOURCE_SHA256",{sha!r})\n'
    protocol=(ROOT/'reports/ROBUSTNESS_PROTOCOL.md').read_text()
    cells=[nbformat.v4.new_markdown_cell(f'# ViHSD — {condition}: research-grounded robustness\n\n{protocol}'),nbformat.v4.new_code_cell(bootstrap),nbformat.v4.new_code_cell(snapshot),nbformat.v4.new_code_cell(f'from vihsd.robustness import run_job\nPROPOSED, BASELINE = run_job(ROOT, CONFIG, {condition!r})'),nbformat.v4.new_code_cell('''from IPython.display import display,Markdown
import pandas as pd
rows=[]
for model,results in [('baseline',BASELINE),('proposed',PROPOSED)]:
    for profile,metrics in results.items():
        rows.append({'model':model,'profile':profile,'macro_f1':metrics['macro_f1'],'accuracy':metrics['accuracy'],'coverage':metrics.get('coverage')})
display(pd.DataFrame(rows))
print((ROOT/'ROBUSTNESS_COMPLETE.json').read_text())
''')]
    n=nbformat.v4.new_notebook(cells=cells);n.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'};n.metadata['source_sha256']=sha
    for c in n.cells:
        if c.cell_type=='code':compile(c.source,'notebook','exec')
    nbformat.validate(n)
    folder=ROOT/'kaggle'/f'robustness-{condition}';folder.mkdir(parents=True,exist_ok=True)
    name=f'vihsd_{condition}.ipynb';nbformat.write(n,folder/name);nbformat.write(n,ROOT/'notebooks'/name)
    metadata={'id':f'ndhoang2310/vihsd-real-noise-{condition}','title':f'ViHSD Real Noise {condition.title()}','code_file':name,'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'dataset_sources':[],'competition_sources':[],'kernel_sources':['ndhoang2310/vihsd-phobert-first-training/2'] + (['ndhoang2310/vihsd-real-noise-fgm/1'] if condition=='augmentation-v2' else [])}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({'condition':condition,'folder':str(folder),'source_sha256':sha}))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('conditions',nargs='*',choices=['fgm','augmentation','fgm-small','augmentation-v2'])
    args=parser.parse_args()
    for condition in (args.conditions or ['fgm-small','augmentation-v2']):build(condition)
