"""Shared Kaggle robustness jobs with evidence-backed views and paired evaluation."""
from __future__ import annotations
import gc
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .data import dump_json, load_raw, normalize_text, segment_text, sha256_bytes
from .noise import fetch_corpus, make_bank, noise_audit, NoiseGenerator, generate_view
from .training import train, predict, metrics, seed_all
from .pipeline import prepare_all


def encode_noisy_view(clean_frame, clean_records, generator, tokenizer, model_dir, seed, rate=None):
    view=generate_view(clean_frame,generator,seed,rate)
    records=list(clean_records)
    if 'original_bpe_length' in clean_frame:
        view['bpe_tokens_before_truncation']=clean_frame.original_bpe_length.to_numpy()
        view['unk_tokens']=clean_frame.original_unk_tokens.to_numpy()
        view['segmented_units']=clean_frame.original_segmented_units.to_numpy()
    changed=np.flatnonzero(view.changed.to_numpy())
    segmented=[]
    for index in changed:
        segmented.append(segment_text(view.iloc[index].text_noisy,model_dir))
    for start in range(0,len(changed),256):
        tokens=tokenizer(segmented[start:start+256],truncation=True,max_length=256,padding=False)
        full=tokenizer(segmented[start:start+256],truncation=False,padding=False,return_attention_mask=False,verbose=False)['input_ids']
        for local,index in enumerate(changed[start:start+256]):
            records[int(index)]={'input_ids':tokens['input_ids'][local], 'attention_mask':tokens['attention_mask'][local], 'labels':int(view.iloc[index].label)}
            if 'bpe_tokens_before_truncation' in view:
                view.at[int(index),'bpe_tokens_before_truncation']=len(full[local])
                view.at[int(index),'unk_tokens']=full[local][1:-1].count(tokenizer.unk_token_id)
                view.at[int(index),'segmented_units']=len(segmented[start+local].split())
    return records,view


def augmentation_factory(frames, encoded, tokenizer, bank, root, seed_base=4242, selection_seed=None):
    generator=NoiseGenerator(bank)
    def records_for_epoch(epoch):
        records,view=encode_noisy_view(frames['train'],encoded['train'],generator,tokenizer,root/'.cache/vncorenlp',seed_base+epoch)
        prefix='augmentation' if selection_seed is None else f'augmentation:{selection_seed}'
        selected=np.array([int(sha256_bytes(f'{prefix}|{sid}|{epoch}'.encode())[:8],16)%2==0 for sid in view.sample_id])
        applied=view.changed.to_numpy() & selected
        for index in np.flatnonzero(~applied):records[int(index)]=encoded['train'][int(index)]
        dump_json(root/'reports/augmentation'/f'epoch_{epoch+1}.json',{'epoch':epoch+1,'n':len(view),'selected':int(selected.sum()),'applied':int(applied.sum()),'eligible':int(view.changed.sum()),'seed':seed_base+epoch,'selection_seed':selection_seed,'bank_sha256':sha256_bytes(json.dumps(bank,sort_keys=True).encode())})
        # Store traces only for applied edits; never imply that this is a human-labeled toxic pair.
        traces=view.loc[applied].copy();traces['edits']=traces.edits.map(json.dumps)
        traces.to_parquet(root/'reports/augmentation'/f'epoch_{epoch+1}_edits.parquet',index=False)
        print(f'AUGMENTATION epoch={epoch+1} applied={applied.sum()}/{len(view)}',flush=True)
        return records
    return records_for_epoch


def locate_baseline():
    paths=list(Path('/kaggle/input').rglob('selection.json'))
    for path in paths:
        if 'phobert-clean-seed42' not in str(path):continue
        selection=json.loads(path.read_text())
        if selection.get('epoch')==3 and abs(selection.get('score',0)-.6863076500763525)<1e-10:
            if (path.parent/'model.safetensors').exists():return path.parent
    raise FileNotFoundError('Attach ndhoang2310/vihsd-phobert-first-training/2 as a kernel source')


def locate_reference(condition, input_root='/kaggle/input'):
    """Validate requested checkpoint inputs before spending GPU time on training."""
    for path in Path(input_root).rglob('run_protocol.json'):
        protocol=json.loads(path.read_text())
        if protocol.get('condition')!=condition:continue
        candidates=list((path.parent.parent/'runs').glob('*/best/model.safetensors'))
        if len(candidates)==1:
            checkpoint=candidates[0].parent
            if (checkpoint/'selection.json').exists():return checkpoint
    raise FileNotFoundError(f'Missing reference checkpoint: {condition}')


def evaluate_profile(model, name, frame, records, tokenizer, baseline_prob, output, nsw_counts=None):
    import torch
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for batch_size in [64,32,16,8]:
        try:
            result,prob=predict(model,records,tokenizer,batch_size,torch.device('cuda'))
            break
        except torch.cuda.OutOfMemoryError:
            gc.collect();torch.cuda.empty_cache()
            if batch_size == 8:raise
    predictions=pd.DataFrame({'sample_id':frame.sample_id.to_numpy(),'label':frame.label.to_numpy(),'prediction':prob.argmax(1)})
    for i,label in enumerate(['clean','offensive','hate']):predictions[f'p_{label}']=prob[:,i]
    predictions.to_parquet(output/f'{name}_predictions.parquet',index=False)
    if baseline_prob is not None:
        previous=baseline_prob.argmax(1)
        result['prediction_flip_rate_vs_original']=float((prob.argmax(1)!=previous).mean())
        result['originally_correct_now_wrong']=int(((previous==frame.label.to_numpy())&(prob.argmax(1)!=frame.label.to_numpy())).sum())
    if nsw_counts is not None:
        result['natural_cohorts']={}
        for marker in [False,True]:
            mask=(nsw_counts>0)==marker
            if mask.any():result['natural_cohorts']['nsw_marker' if marker else 'no_nsw_marker']={'n':int(mask.sum()),**metrics(frame.label.to_numpy()[mask],prob.argmax(1)[mask])}
    dump_json(output/f'{name}_metrics.json',result)
    print('ROBUSTNESS_METRICS '+json.dumps({'profile':name,'macro_f1':result['macro_f1'],'accuracy':result['accuracy']}),flush=True)
    return result,prob


def evaluate_suite(checkpoint, name, frames, encoded, tokenizer, bank, root, model_dir, profiles=None):
    import torch
    from transformers import AutoModelForSequenceClassification
    output=root/'robustness'/name;output.mkdir(parents=True,exist_ok=True)
    model=AutoModelForSequenceClassification.from_pretrained(checkpoint).to('cuda').eval()
    detector=NoiseGenerator(bank)
    markers=frames['test'].text_raw.map(detector.detect).to_numpy()
    summary={}
    clean,original_prob=evaluate_profile(model,'original',frames['test'],encoded['test'],tokenizer,None,output,markers)
    summary['original']=clean
    if name=='baseline':
        assert abs(clean['macro_f1']-.6520143815686111)<1e-10,'Baseline reproduction differs: block comparison'
    specs=[]
    for partition in ['train','unseen']:
        for seed in [201,202,203]:specs.append((f'empirical_{partition}_s{seed}',partition,seed,None,None))
    for rate in [.1,.2,.3,.5]:specs.append((f'stress_{int(rate*100)}_s201','train',201,rate,None))
    # Type-specific diagnostics use observed pairs only; low coverage is explicitly reported.
    for kind in ([] if profiles == 'core' else sorted({r['kind'] for r in bank['rows'] if r['partition']=='train'})):
        specs.append((f'type_{kind}_s201','train',201,.3,kind))
    for profile,partition,seed,rate,kind in specs:
        cache=root/'data/noise_views'/profile;cache.mkdir(parents=True,exist_ok=True)
        view_path=cache/'view.parquet';records_path=cache/'encoded.json'
        if view_path.exists() and records_path.exists():
            view=pd.read_parquet(view_path);records=json.loads(records_path.read_text())
        else:
            selected=bank if kind is None else {**bank,'rows':[r for r in bank['rows'] if r['kind']==kind]}
            records,view=encode_noisy_view(frames['test'],encoded['test'],NoiseGenerator(selected,partition),tokenizer,model_dir,seed,rate)
            saved=view.copy();saved['edits']=saved.edits.map(json.dumps);saved.to_parquet(view_path,index=False)
            dump_json(records_path,records)
        result,prob=evaluate_profile(model,profile,frames['test'],records,tokenizer,original_prob,output)
        result.update({'coverage':float(view.changed.mean()),'changed_rows':int(view.changed.sum()),'actual_rate_mean':float(view.actual_rate.mean()),'delta_macro_f1_vs_original':result['macro_f1']-clean['macro_f1'],'label_preservation_verified':False})
        if 'bpe_tokens_before_truncation' in view:
            result['tokenization']={'mean_bpe_length':float(view.bpe_tokens_before_truncation.mean()),'unk_token_rate':float(view.unk_tokens.sum()/max(1,(view.bpe_tokens_before_truncation-2).sum())),'mean_fragmentation':float(((view.bpe_tokens_before_truncation-2)/view.segmented_units.clip(lower=1)).mean()),'truncation_rate':float((view.bpe_tokens_before_truncation>256).mean()),'delta_segmented_units_mean':float((view.segmented_units.to_numpy()-frames['test'].original_segmented_units.to_numpy()).mean())}
        changed=view.changed.to_numpy()
        if changed.any():result['changed_only']={'n':int(changed.sum()),**metrics(frames['test'].label.to_numpy()[changed],prob.argmax(1)[changed])}
        # Paired changed-only clean reference avoids crediting the generator's no-op samples.
        if changed.any():result['changed_only_original_macro_f1']=metrics(frames['test'].label.to_numpy()[changed],original_prob.argmax(1)[changed])['macro_f1']
        summary[profile]=result;dump_json(output/f'{profile}_metrics.json',result)
    dump_json(output/'summary.json',summary)
    del model;gc.collect();torch.cuda.empty_cache()
    return summary


def run_job(root, config, condition):
    import torch
    root=Path(root).resolve()
    print('ROBUSTNESS_JOB',condition,flush=True)
    baseline=locate_baseline()
    reference_checkpoints={c:locate_reference(c) for c in config.get('extra_reference_conditions',[])}
    frames,encoded,tokenizer,resolved=prepare_all(root,config)
    diagnostics=json.loads((root/'data/tokenized/token_diagnostics.json').read_text())
    for split,frame in frames.items():
        frame['original_bpe_length']=diagnostics[split]['lengths']
        frame['original_unk_tokens']=diagnostics[split]['unk_counts']
        frame['original_segmented_units']=frame.text_segmented.str.split().str.len()
    raw=load_raw(root)
    corpus,manifest=fetch_corpus(root)
    if config.get('noise_version') == 2:
        from .noise_v2 import fetch_in_domain,make_bank_v2
        domain,_=fetch_in_domain(root)
        bank=make_bank_v2(corpus,domain,raw,root/'data/noise')
    else:
        bank=make_bank(corpus,raw,root/'data/noise')
    audit=noise_audit(corpus,bank,raw,root/'reports/noise')
    if not any(r['partition']=='train' for r in bank['rows']):raise ValueError('Empty training bank')
    dump_json(root/'reports/noise/claims.json',{'human_annotated_source':True,'synthetic_toxic_labels_human_verified':False,'profile':'one observed lexical event per eligible comment; corpus-weighted variants','model_test_used_to_design_generator':False})
    print('NOISE_BANK_READY '+json.dumps(audit['bank_counts']),flush=True)
    callbacks=None
    if condition in ['fgm','fgm-small']:
        epsilon = .1 if condition == 'fgm-small' else 1.0
        resolved.update(fgm_epsilon=epsilon,run_name=f'phobert-fgm-eps{epsilon:g}-seed42')
    elif condition in ['augmentation','augmentation-v2']:
        resolved.update(fgm_epsilon=0.0,run_name='phobert-human-augmentation-v2-seed42' if condition == 'augmentation-v2' else 'phobert-observed-augmentation-seed42',augmentation_probability=.5)
        callbacks=augmentation_factory(frames,encoded,tokenizer,bank,root)
    else:raise ValueError(condition)
    dump_json(root/'reports/run_protocol.json',{'condition':condition,'epsilon_predeclared':resolved.get('fgm_epsilon'), 'hyperparameter_sweep_performed':False,'training_seed':42,'noise_version':bank['version'],'source_revision':bank['source_revision'],'bank_sha256':sha256_bytes(json.dumps(bank,sort_keys=True).encode()),'synthetic_label_preservation':'assumed, not human verified'})
    # GPU sanity test on a real microbatch before full training; epsilon-zero path is unit-tested.
    from .training import make_model,make_loader
    from .adversarial import backward_with_fgm
    seed_all(42);model=make_model(resolved).to('cuda');model.train()
    batch=next(iter(make_loader(encoded['train'][:8],tokenizer,8)));batch={k:v.to('cuda') for k,v in batch.items()}
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    with torch.autocast('cuda',dtype=torch.float16):loss=model(**batch).loss
    before=model.get_input_embeddings().weight.detach().clone()
    value,skipped=backward_with_fgm(model,batch,loss,scaler,1,max(resolved['fgm_epsilon'],.1))
    assert torch.equal(before,model.get_input_embeddings().weight),'Embedding not exactly restored'
    assert np.isfinite(value) and not skipped
    dump_json(root/'reports/fgm_gpu_smoke.json',{'passed':True,'loss':value,'embedding_restored_exactly':True,'epsilon_used':max(resolved['fgm_epsilon'],.1)})
    del model,batch,before,scaler;gc.collect();torch.cuda.empty_cache()
    best=train(frames,encoded,tokenizer,resolved,root/'runs'/resolved['run_name'],epoch_records_factory=callbacks)
    # No further model or parameter choice after this point.
    print('ROBUSTNESS_STAGE FINAL_EVALUATION',flush=True)
    proposed=evaluate_suite(best,condition,frames,encoded,tokenizer,bank,root,root/'.cache/vncorenlp',profiles=config.get('evaluation_profile_mode'))
    # Both jobs include the exact frozen baseline, so comparisons use identical generated files.
    reference=evaluate_suite(baseline,'baseline',frames,encoded,tokenizer,bank,root,root/'.cache/vncorenlp',profiles=config.get('evaluation_profile_mode'))
    extra_references=[]
    for reference_condition,found in reference_checkpoints.items():
        alias='fgm-eps1-reference' if reference_condition=='fgm' else 'pilot-augmentation-reference'
        evaluate_suite(found,alias,frames,encoded,tokenizer,bank,root,root/'.cache/vncorenlp',profiles=config.get('evaluation_profile_mode'))
        import shutil
        shutil.copy2(found/'selection.json',root/'robustness'/alias/'selection.json')
        extra_references.append(alias)
    dump_json(root/'ROBUSTNESS_COMPLETE.json',{'condition':condition,'run_name':resolved['run_name'],'bank_sha256':sha256_bytes(json.dumps(bank,sort_keys=True).encode()),'proposed_clean':proposed['original']['macro_f1'],'baseline_clean':reference['original']['macro_f1'],'test_profiles':list(proposed),'extra_references':extra_references,'noise_version':bank['version'],'status':'complete'})
    print('ROBUSTNESS_COMPLETE '+condition,flush=True)
    return proposed,reference
