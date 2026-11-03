import copy
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from vihsd.noise import NoiseGenerator, align_pairs, classify_pair
from vihsd.adversarial import FGM, backward_with_fgm


def bank():
    return {'rows':[
        {'mapping_id':'a','canonical':'được','variant':'đc','count':10,'kind':'abbreviation','source_ids':['gold:1'],'partition':'train'},
        {'mapping_id':'b','canonical':'được','variant':'dc','count':2,'kind':'abbreviation','source_ids':['gold:2'],'partition':'unseen'},
        {'mapping_id':'c','canonical':'rồi','variant':'r','count':5,'kind':'abbreviation','source_ids':['gold:3'],'partition':'train'}]}


def test_gold_alignment_and_no_paraphrase():
    pairs=align_pairs('thích anh cá mập k','thích anh cá mập không')
    assert pairs==[{'canonical':'không','variant':'k','kind':'abbreviation'}]
    assert align_pairs('t là người tốt','tôi là người tốt')==[]
    assert align_pairs('t là người tốt','tôi là người tốt',include_protected=True)[0]['canonical']=='tôi'
    assert align_pairs('xin chào','xin chào')==[]


def test_noise_deterministic_trace_and_unicode():
    gen=NoiseGenerator(bank())
    text='Được rồi 😂 #abc'
    a=gen.generate(text,'test:0',101,.5);b=gen.generate(text,'test:0',101,.5)
    assert a==b and a['changed'] and '😂' in a['text_noisy']
    reconstructed=a['text_original']
    for e in sorted(a['edits'],key=lambda x:x['start'],reverse=True):
        assert reconstructed[e['start']:e['end']]==e['original']
        reconstructed=reconstructed[:e['start']]+e['replacement']+reconstructed[e['end']:]
    assert reconstructed==a['text_noisy']
    assert not a['label_preservation_verified']


def test_no_eligible_spans_and_zero_rate():
    gen=NoiseGenerator(bank())
    assert not gen.generate('abc','id',1,.5)['changed']
    assert not gen.generate('được rồi','id',1,0.)['changed']
    assert gen.generate('được','id',1,.5)['actual_rate']==1.
    assert not gen.generate('https://abc.vn/được @được','id',1,.5)['changed']


def test_bank_partition_separation():
    a=NoiseGenerator(bank());b=NoiseGenerator(bank(),'unseen')
    assert a.generate('được','id',1)['text_noisy']=='đc'
    assert b.generate('được','id',1)['text_noisy']=='dc'


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(12,4,padding_idx=0);self.head=torch.nn.Linear(4,3)
    def get_input_embeddings(self):return self.embedding
    def forward(self,input_ids,labels):
        logits=self.head(self.embedding(input_ids).mean(1));return SimpleNamespace(loss=torch.nn.functional.cross_entropy(logits,labels),logits=logits)


def batch():return {'input_ids':torch.tensor([[1,2,0],[3,4,5]]),'labels':torch.tensor([0,2])}


def test_fgm_uses_current_gradient_not_accumulated():
    torch.manual_seed(42);m=TinyClassifier();s=torch.amp.GradScaler('cpu',enabled=False)
    loss=m(**batch()).loss
    m.embedding.weight.grad=torch.full_like(m.embedding.weight,987.)
    f=FGM(m,.5);r=f.direction(loss,s)
    m.embedding.weight.grad=torch.full_like(m.embedding.weight,-123.)
    r2=f.direction(loss,s)
    assert torch.allclose(r,r2) and torch.allclose(r.norm(),torch.tensor(.5),atol=1e-6)
    assert torch.equal(r[0],torch.zeros_like(r[0]))


def test_fgm_restores_exactly_on_exception():
    m=TinyClassifier();original=m.embedding.weight.detach().clone();f=FGM(m,.5)
    with pytest.raises(RuntimeError):
        with f.attack(torch.ones_like(original)*.01):
            assert not torch.equal(m.embedding.weight,original)
            raise RuntimeError('injected failure')
    assert torch.equal(m.embedding.weight,original)


def test_epsilon_zero_matches_baseline_gradient():
    m=TinyClassifier();n=copy.deepcopy(m);s=torch.amp.GradScaler('cpu',enabled=False)
    (m(**batch()).loss/4).backward()
    backward_with_fgm(n,batch(),n(**batch()).loss,s,4,0.,device_type='cpu')
    for p,q in zip(m.parameters(),n.parameters()):assert torch.equal(p.grad,q.grad)


def test_fgm_finite_backward_and_accumulation():
    m=TinyClassifier();s=torch.amp.GradScaler('cpu',enabled=False);original=m.embedding.weight.detach().clone()
    for _ in range(2):
        value,skipped=backward_with_fgm(m,batch(),m(**batch()).loss,s,2,.5,device_type='cpu')
        assert np.isfinite(value) and not skipped
    assert torch.equal(m.embedding.weight,original)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_noise_preserves_all_uppercase():
    result=NoiseGenerator(bank()).generate('ĐƯỢC RỒI 😂','id',1,.5)
    assert all(e['replacement'].isupper() for e in result['edits'])


def test_sampling_weight_and_source_version_are_explicit():
    b=bank();b['version']='version-a'
    a=NoiseGenerator(b).generate('được rồi','id',101,.5)
    b2=copy.deepcopy(b);b2['version']='version-b'
    assert NoiseGenerator(b2).version!=NoiseGenerator(b).version
    assert a==NoiseGenerator(b).generate('được rồi','id',101,.5)


def test_v2_weight_is_conditional_and_bank_stays_disjoint():
    from vihsd.noise_v2 import make_bank_v2
    # Real source fixture includes two independent supporting contexts and ambiguity.
    c={'train':pd.DataFrame([
        {'original':'đc rồi','normalized':'được rồi','source_id':'gold:1'},
        {'original':'rất đc','normalized':'rất được','source_id':'gold:2'},
        {'original':'được không','normalized':'được không','source_id':'gold:3'},
    ])}
    d={'train':pd.DataFrame(columns=['original','normalized','source_id'])}
    f={s:pd.DataFrame({'sample_id':[s+':0'],'text_raw':['đc trong train' if s=='train' else 'mẫu độc lập']}) for s in ['train','validation','test']}
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        b=make_bank_v2(c,d,f,tmp,calibrate=False)
    row=next(r for r in b['rows'] if r['variant']=='đc')
    assert row['partition']=='train'  # a known surface must not be discarded as a fake unseen form
    assert row['count']==2 and row['canonical_opportunities']==3
    assert row['conditional_support_weight']==pytest.approx(2/3)
    assert row['sampling_weight']>0


def test_noisy_view_records_tokenization_and_preserves_ids():
    from pathlib import Path
    if not Path('.cache/vncorenlp/VnCoreNLP-1.2.jar').exists():pytest.skip('Segmenter resources missing')
    from vihsd.pipeline import ensure_java
    from vihsd.robustness import encode_noisy_view
    from transformers import AutoTokenizer
    ensure_java();tokenizer=AutoTokenizer.from_pretrained('vinai/phobert-base',use_fast=False)
    f=pd.DataFrame({'sample_id':['test:0','test:1'],'text_raw':['được rồi 😂','không có mapping'],'label':[0,1],'split':['test','test'],'original_bpe_length':[6,5],'original_unk_tokens':[1,0],'original_segmented_units':[3,3]})
    ids=tokenizer(f.text_raw.tolist(),padding=False,truncation=True,max_length=256)
    encoded=[{'input_ids':i,'attention_mask':a,'labels':int(l)} for i,a,l in zip(ids['input_ids'],ids['attention_mask'],f.label)]
    records,view=encode_noisy_view(f,encoded,NoiseGenerator(bank()),tokenizer,Path('.cache/vncorenlp'),101)
    assert list(view.sample_id)==list(f.sample_id) and len(records)==2
    assert view.iloc[0].changed and not view.iloc[1].changed
    assert view.iloc[0].unk_tokens>=1 and view.iloc[0].bpe_tokens_before_truncation>=3
    assert records[1] is encoded[1]


def test_reference_discovery_requires_complete_checkpoint(tmp_path):
    from vihsd.robustness import locate_reference
    root=tmp_path/'mounted_project';reports=root/'reports';reports.mkdir(parents=True)
    (reports/'run_protocol.json').write_text(json.dumps({'condition':'fgm'}))
    best=root/'runs/model/best';best.mkdir(parents=True);(best/'model.safetensors').write_bytes(b'fixture')
    with pytest.raises(FileNotFoundError):locate_reference('fgm',tmp_path)
    (best/'selection.json').write_text(json.dumps({'epoch':5,'score':.6}))
    assert locate_reference('fgm',tmp_path)==best


def test_missing_reference_fails_before_data_or_training(monkeypatch,tmp_path):
    import vihsd.robustness as pipeline
    monkeypatch.setattr(pipeline,'locate_baseline',lambda:tmp_path)
    def missing(condition):raise FileNotFoundError(f'Missing reference checkpoint: {condition}')
    monkeypatch.setattr(pipeline,'locate_reference',missing)
    monkeypatch.setattr(pipeline,'prepare_all',lambda *a:pytest.fail('Data preparation ran before reference validation'))
    monkeypatch.setattr(pipeline,'train',lambda *a,**k:pytest.fail('Training ran before reference validation'))
    with pytest.raises(FileNotFoundError,match='augmentation'):
        pipeline.run_job(tmp_path,{'extra_reference_conditions':['augmentation']},'augmentation-v2')
