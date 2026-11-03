"""Fresh 20-epoch-horizon training; the official test is never evaluated here."""
from pathlib import Path
import json
import shutil
import pandas as pd
from .data import dump_json,sha256_bytes
from .pipeline import prepare_all
from .training import train
from .robustness import augmentation_factory


def run_training(root, config, condition, bank, resume_source=None):
    if condition not in ['clean','fgm','augmentation']:raise ValueError(condition)
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    digest=sha256_bytes(json.dumps(bank,sort_keys=True).encode())
    if digest!=config['noise_bank_sha256']:raise ValueError('Frozen noise bank differs from protocol')
    # Every condition receives the same declared horizon and initialization policy.
    seed=int(config['seed']);prefix=config.get('run_prefix','phobert-convergence');name=f'{prefix}-{condition}-seed{seed}'
    config={**config,'run_name':name,'fgm_epsilon':config['fgm_epsilon_selected'] if condition=='fgm' else 0.0}
    frames,encoded,tokenizer,resolved=prepare_all(root,config)
    diagnostics=json.loads((root/'data/tokenized/token_diagnostics.json').read_text())
    for split,frame in frames.items():
        frame['original_bpe_length']=diagnostics[split]['lengths']
        frame['original_unk_tokens']=diagnostics[split]['unk_counts']
        frame['original_segmented_units']=frame.text_segmented.str.split().str.len()
    dump_json(root/'data/noise/bank.json',bank)
    dump_json(root/'reports/convergence_protocol.json',{'condition':condition,'seed':seed,'max_epochs':resolved['epochs'],'patience':resolved['early_stopping_patience'],'min_delta':resolved['early_stopping_min_delta'],'selection':'clean validation Macro-F1; strict best, first wins ties','noise_bank_sha256':digest,'test_evaluated':False,'initialization':'fresh pretrained, unless explicitly resuming SAME 20-epoch run','scheduler_horizon_epochs':resolved['epochs']})
    run=root/'runs'/name;resume=None
    if resume_source:
        source=Path(resume_source).resolve()
        if not (source/'last.pt').is_file():raise FileNotFoundError('Resume source must be a trusted run directory with last.pt and best/')
        # Never mutate mounted inputs or overwrite an unrelated local run.
        if run.exists():raise FileExistsError(f'Refusing to overwrite resume target {run}')
        shutil.copytree(source,run);resume=run/'last.pt'
    callback=None
    if condition=='augmentation':
        callback=augmentation_factory(frames,encoded,tokenizer,bank,root,seed_base=4242+seed*1000003,selection_seed=seed)
    best=train(frames,encoded,tokenizer,resolved,run,resume=resume,epoch_records_factory=callback)
    summary=json.loads((run/'training_summary.json').read_text())
    selection=json.loads((best/'selection.json').read_text())
    history=json.loads((run/'history.json').read_text())
    maximum=max(r['validation']['macro_f1'] for r in history)
    assert selection['score']==maximum and selection['epoch']==next(r['epoch'] for r in history if r['validation']['macro_f1']==maximum)
    assert summary['epochs_completed']<=config['epochs'] and summary['stop_reason'] in ['early_stopping','max_epochs']
    if config.get('early_stopping_patience') is None:
        assert summary['epochs_completed']==config['epochs'] and summary['stop_reason']=='max_epochs'
    dump_json(root/'TRAINING_COMPLETE.json',{'condition':condition,'seed':seed,'run_name':name,'noise_bank_sha256':digest,'test_evaluated':False,**summary})
    print('CONVERGENCE_TRAINING_COMPLETE '+json.dumps({'condition':condition,'seed':seed,**summary}),flush=True)
    return summary
