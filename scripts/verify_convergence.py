"""Independently verify validation predictions, model selection and stopping decisions."""
import json,math,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
from vihsd.stopping import EarlyStopping
from vihsd.data import load_raw,dump_json
ROOT=Path(__file__).resolve().parents[1]


def verify(root, expected_protocol=None):
    root=Path(root);done=json.loads((root/'TRAINING_COMPLETE.json').read_text());assert done['test_evaluated'] is False
    run=root/'runs'/done['run_name'];config=json.loads((run/'config.json').read_text());history=json.loads((run/'history.json').read_text());selection=json.loads((run/'best/selection.json').read_text())
    expected_protocol=expected_protocol or {'epochs':20,'early_stopping_patience':3,'early_stopping_min_delta':0.}
    for key in ['epochs','early_stopping_patience','early_stopping_min_delta']:
        assert config[key]==expected_protocol[key],f'Unexpected {key}'
    if expected_protocol.get('protocol_version'):
        assert config['protocol_version']==expected_protocol['protocol_version']
    if expected_protocol.get('noise_bank_sha256'):
        bank=json.loads((root/'data/noise/bank.json').read_text())
        assert hashlib.sha256(json.dumps(bank,sort_keys=True).encode()).hexdigest()==expected_protocol['noise_bank_sha256']==done['noise_bank_sha256']
    assert config['seed']==done['seed'] and len(history)==done['epochs_completed'] and 1<=len(history)<=20
    raw=load_raw(ROOT)['validation'].set_index('sample_id');stop=EarlyStopping(config['early_stopping_patience'],config['early_stopping_min_delta'])
    for number,record in enumerate(history,1):
        assert record['epoch']==number
        assert not stop.should_stop,'Training continued after patience was exhausted'
        p=pd.read_parquet(run/'validation'/f'epoch_{number}.parquet');assert len(p)==2672 and p.sample_id.is_unique
        assert np.array_equal(p.label,raw.loc[p.sample_id].label)
        probs=p[['p_clean','p_offensive','p_hate']].to_numpy();assert np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all() and np.allclose(probs.sum(1),1,atol=1e-5)
        assert np.array_equal(probs.argmax(1),p.prediction)
        score=f1_score(p.label,p.prediction,labels=[0,1,2],average='macro',zero_division=0)
        assert abs(score-record['validation']['macro_f1'])<1e-12
        assert abs(accuracy_score(p.label,p.prediction)-record['validation']['accuracy'])<1e-12
        assert confusion_matrix(p.label,p.prediction,labels=[0,1,2]).tolist()==record['validation']['confusion_matrix']
        stop.observe(score);assert stop.state_dict()==record['early_stopping']
    assert done['early_stopping']==stop.state_dict()
    assert (done['stop_reason']=='early_stopping' and stop.should_stop) or (done['stop_reason']=='max_epochs' and len(history)==20 and not stop.should_stop)
    if config['early_stopping_patience'] is None:
        assert len(history)==config['epochs'] and done['stop_reason']=='max_epochs'
    best=max(r['validation']['macro_f1'] for r in history);epoch=next(r['epoch'] for r in history if r['validation']['macro_f1']==best)
    assert selection['score']==best and selection['epoch']==epoch==done['best_epoch']
    steps=math.ceil(math.ceil(24046/config['actual_train_batch_size'])/config['actual_gradient_accumulation_steps'])
    assert done['global_step']==steps*len(history)
    return {'passed':True,'condition':done['condition'],'seed':done['seed'],'epochs_completed':len(history),'best_epoch':epoch,'best_validation_macro_f1':best,'stop_reason':done['stop_reason'],'test_evaluated':False}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();print(json.dumps(verify(a.root),indent=2))
