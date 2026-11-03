"""Frozen full20 checkpoints: validation gate, identical v2 views, no training."""
import gc,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .data import dump_json,normalize_text
from .pipeline import prepare_all
from .training import predict,seed_all,environment
from .noise import NoiseGenerator
from .robustness import encode_noisy_view,evaluate_suite
PROFILES=['original']+[f'empirical_{p}_s{s}' for p in ['train','unseen'] for s in [201,202,203]]+[f'stress_{r}_s201' for r in [10,20,30,50]]

def view_sha(view):
    h=hashlib.sha256()
    for r in view.itertuples(index=False):
        edits=json.loads(r.edits) if isinstance(r.edits,str) else r.edits
        row=[str(r.sample_id),int(r.label),str(r.text_original),str(r.text_noisy),bool(r.changed),edits]
        h.update((json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode())
    return h.hexdigest()

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def locate(job,input_root):
    found=[]
    for marker in Path(input_root).rglob('TRAINING_COMPLETE.json'):
        d=json.loads(marker.read_text())
        if d.get('condition')==job['condition'] and d.get('seed')==job['seed'] and d.get('run_name')==f"phobert-full20-{job['condition']}-seed{job['seed']}":found.append((marker.parent,d))
    if len(found)!=1:raise RuntimeError(f"Expected exactly one mounted completed input for {job['job']}, got {len(found)}")
    root,d=found[0];run=root/'runs'/d['run_name'];best=run/'best'
    assert d['epochs_completed']==20 and d['stop_reason']=='max_epochs' and d['early_stopping']['patience'] is None
    assert d['test_evaluated'] is False and d['noise_bank_sha256']==job['noise_bank_sha256']
    config=json.loads((run/'config.json').read_text());selection=json.loads((best/'selection.json').read_text())
    assert config['protocol_version']=='full20-v1' and config['epochs']==20 and config['early_stopping_patience'] is None
    assert config['model_revision']==job['model_revision'] and config['seed']==job['seed']
    assert selection['epoch']==job['best_epoch'] and abs(selection['score']-job['best_validation_macro_f1'])<1e-12
    assert (best/'model.safetensors').is_file()
    return root,run,best

def prepare_views(root,frames,encoded,tokenizer,bank,expected):
    hashes={}
    for profile in PROFILES[1:]:
        if profile.startswith('empirical_'):
            _,partition,s=profile.split('_');seed=int(s[1:]);rate=None
        else:
            _,rate_string,s=profile.split('_');partition='train';seed=int(s[1:]);rate=int(rate_string)/100
        records,view=encode_noisy_view(frames['test'],encoded['test'],NoiseGenerator(bank,partition),tokenizer,root/'.cache/vncorenlp',seed,rate)
        digest=view_sha(view);assert digest==expected[profile],f'Frozen diagnostic view mismatch: {profile}'
        cache=root/'data/noise_views'/profile;cache.mkdir(parents=True,exist_ok=True)
        saved=view.copy();saved['edits']=saved.edits.map(json.dumps);saved.to_parquet(cache/'view.parquet',index=False)
        dump_json(cache/'encoded.json',records);hashes[profile]=digest
    dump_json(root/'VIEW_HASHES.json',hashes)
    return hashes

def run_evaluation(root,config,jobs,bank,expected_views,input_root='/kaggle/input'):
    import torch
    from transformers import AutoModelForSequenceClassification
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available();seed_all(42)
    assert hashlib.sha256(json.dumps(bank,sort_keys=True).encode()).hexdigest()==config['noise_bank_sha256']
    # Check every private input before preparing data or reading test predictions.
    inputs={j['job']:locate(j,input_root) for j in jobs}
    frames,encoded,tokenizer,resolved=prepare_all(root,config)
    diagnostics=json.loads((root/'data/tokenized/token_diagnostics.json').read_text())
    for split,frame in frames.items():
        frame['original_bpe_length']=diagnostics[split]['lengths'];frame['original_unk_tokens']=diagnostics[split]['unk_counts'];frame['original_segmented_units']=frame.text_segmented.str.split().str.len()
    dump_json(root/'data/noise/bank.json',bank);dump_json(root/'environment.json',environment())
    audits={}
    # All mounted checkpoints must reproduce saved validation predictions before any test metric.
    for job in jobs:
        source,run,best=inputs[job['job']]
        assert json.loads((source/'data/tokenized/validation.json').read_text())==encoded['validation'],'Validation tokenization changed'
        expected=pd.read_parquet(run/'validation'/f"epoch_{job['best_epoch']}.parquet")
        assert np.array_equal(expected.sample_id,frames['validation'].sample_id)
        assert np.array_equal(expected.label,frames['validation'].label)
        model=AutoModelForSequenceClassification.from_pretrained(best).to('cuda').eval()
        result,prob=predict(model,encoded['validation'],tokenizer,32,torch.device('cuda'))
        assert np.array_equal(prob.argmax(1),expected.prediction),'Checkpoint validation predictions differ'
        assert abs(result['macro_f1']-job['best_validation_macro_f1'])<1e-12
        difference=float(np.abs(prob-expected[['p_clean','p_offensive','p_hate']].to_numpy()).max())
        assert difference<2e-5,f'Checkpoint validation probabilities differ: {difference}'
        folder=root/'validation_gate'/job['job'];folder.mkdir(parents=True,exist_ok=True)
        p=expected.copy()
        for i,c in enumerate(['p_clean','p_offensive','p_hate']):p[c]=prob[:,i]
        p.prediction=prob.argmax(1);p.to_parquet(folder/'predictions.parquet',index=False)
        audit={'job':job['job'],'kernel_input':job['kernel']+'/'+str(job['version']),'selected_epoch':job['best_epoch'],'validation_macro_f1':result['macro_f1'],'validation_prob_max_difference':difference,'model_sha256':file_sha(best/'model.safetensors'),'passed':True}
        audits[job['job']]=audit;dump_json(folder/'audit.json',audit)
        del model;gc.collect();torch.cuda.empty_cache()
    hashes=prepare_views(root,frames,encoded,tokenizer,bank,expected_views)
    summaries={}
    for job in jobs:
        summaries[job['job']]=evaluate_suite(inputs[job['job']][2],job['job'],frames,encoded,tokenizer,bank,root,root/'.cache/vncorenlp',profiles='core')
        assert set(summaries[job['job']])==set(PROFILES)
    done={'protocol':'full20-evaluation-v1','training_performed':False,'bank_sha256':config['noise_bank_sha256'],'jobs':[j['job'] for j in jobs],'profiles':PROFILES,'view_hashes':hashes,'validation_gates':audits,'test_already_viewed_in_prior_research':True,'synthetic_labels_human_verified':False}
    dump_json(root/'EVALUATION_COMPLETE.json',done);print('EVALUATION_COMPLETE '+json.dumps(done),flush=True)
    return done
