import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from vihsd.stopping import EarlyStopping
from vihsd import training


def test_stopping_ties_and_small_improvements():
    stop=EarlyStopping(3,.01)
    assert not stop.observe(.5)
    assert not stop.observe(.505)
    assert not stop.observe(.5)
    assert stop.observe(.51)  # <= best + delta does not reset patience
    other=EarlyStopping(3,.01);other.restore(stop.state_dict());assert other.should_stop
    with pytest.raises(ValueError):EarlyStopping(0)
    with pytest.raises(ValueError):stop.observe(float('nan'))
    with pytest.raises(ValueError):EarlyStopping(4,.01).restore(stop.state_dict())


class TinyModel(torch.nn.Module):
    is_gradient_checkpointing=False
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(12,4,padding_idx=0);self.head=torch.nn.Linear(4,3)
    def get_input_embeddings(self):return self.embedding
    def forward(self,input_ids,labels):
        logits=self.head(self.embedding(input_ids).mean(1))
        return SimpleNamespace(loss=torch.nn.functional.cross_entropy(logits,labels),logits=logits)
    def save_pretrained(self,folder):
        Path(folder).mkdir(parents=True,exist_ok=True);torch.save(self.state_dict(),Path(folder)/'pytorch_model.bin')


class TinyTokenizer:
    def save_pretrained(self,folder):pass


def setup_loop(monkeypatch, predictions):
    rows=[{'input_ids':[i+1,2],'labels':i%3} for i in range(4)]
    frame=pd.DataFrame({'sample_id':[f'validation:{i}' for i in range(4)],'label':[0,1,2,0]})
    encoded={'train':rows,'validation':rows};frames={'validation':frame}
    def loader(records,tokenizer,batch_size,shuffle=False,seed=42):
        def collate(batch):return {k:torch.tensor([r[k] for r in batch]) for k in ['input_ids','labels']}
        return torch.utils.data.DataLoader(records,batch_size=batch_size,shuffle=shuffle,generator=torch.Generator().manual_seed(seed),collate_fn=collate)
    sequence=iter(predictions)
    def predict(*args):
        p=next(sequence);probs=np.full((4,3),.05,dtype=np.float32);probs[np.arange(4),p]=.9
        return {**training.metrics(frame.label,p),'loss':.5},probs
    monkeypatch.setattr(training,'make_model',lambda c:TinyModel())
    monkeypatch.setattr(training,'make_loader',loader);monkeypatch.setattr(training,'predict',predict)
    config={'seed':42,'epochs':8,'early_stopping_patience':2,'early_stopping_min_delta':0.,'train_batch_size':2,'gradient_accumulation_steps':2,'eval_batch_size':4,'weight_decay':.01,'learning_rate':.01,'warmup_ratio':.1,'max_grad_norm':1.,'save_steps':1,'save_validation_predictions':True,'fgm_epsilon':.1}
    return frames,encoded,TinyTokenizer(),config


FIRST=[0,0,0,0];BEST=[0,1,2,0];BAD=[0,1,0,0]


def test_training_stops_and_resume_does_not_restart_stopped_run(monkeypatch,tmp_path):
    f,e,t,c=setup_loop(monkeypatch,[FIRST,BEST,BAD,BAD])
    training.train(f,e,t,c,tmp_path,_test_device='cpu')
    history=json.loads((tmp_path/'history.json').read_text());summary=json.loads((tmp_path/'training_summary.json').read_text())
    assert len(history)==4 and summary['stop_reason']=='early_stopping' and summary['best_epoch']==2
    assert summary['global_step']==4 and summary['max_epochs']==8
    assert len(list((tmp_path/'validation').glob('*.parquet')))==4
    saved=torch.load(tmp_path/'last.pt',weights_only=False);assert saved['early_stopping']['bad_epochs']==2
    training.train(f,e,t,c,tmp_path,resume=tmp_path/'last.pt',_test_device='cpu')
    after=torch.load(tmp_path/'last.pt',weights_only=False)
    assert saved['global_step']==after['global_step']
    assert all(torch.equal(saved['model'][k],after['model'][k]) for k in saved['model'])


def test_interrupted_resume_preserves_patience_optimizer_and_rng(monkeypatch,tmp_path):
    full=tmp_path/'full';partial=tmp_path/'resumed'
    f,e,t,c=setup_loop(monkeypatch,[FIRST,BEST,BAD,BAD]);training.train(f,e,t,c,full,_test_device='cpu')
    f,e,t,c=setup_loop(monkeypatch,[FIRST,BEST,BAD])
    def interrupt(epoch):
        if epoch==3:raise InterruptedError('simulate session boundary')
        return e['train']
    with pytest.raises(InterruptedError):training.train(f,e,t,c,partial,epoch_records_factory=interrupt,_test_device='cpu')
    checkpoint=torch.load(partial/'last.pt',weights_only=False);assert checkpoint['early_stopping']['bad_epochs']==1
    f,e,t,c=setup_loop(monkeypatch,[BAD]);training.train(f,e,t,c,partial,resume=partial/'last.pt',_test_device='cpu')
    a=torch.load(full/'last.pt',weights_only=False);b=torch.load(partial/'last.pt',weights_only=False)
    assert a['global_step']==b['global_step']==4 and a['early_stopping']==b['early_stopping']
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    assert a['scheduler']==b['scheduler']


def test_disabled_stopping_runs_to_budget_and_ties_keep_first_checkpoint(monkeypatch,tmp_path):
    f,e,t,c=setup_loop(monkeypatch,[FIRST]*20);c.update(epochs=20,early_stopping_patience=None)
    training.train(f,e,t,c,tmp_path,_test_device='cpu')
    summary=json.loads((tmp_path/'training_summary.json').read_text())
    assert summary['epochs_completed']==20 and summary['stop_reason']=='max_epochs' and summary['best_epoch']==1
    assert summary['global_step']==20 and len(list((tmp_path/'validation').glob('*.parquet')))==20
