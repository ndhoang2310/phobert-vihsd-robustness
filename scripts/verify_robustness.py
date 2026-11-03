"""Independently validate predictions, shared noise, bank isolation and checkpoint choice."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
from vihsd.data import load_raw,dump_json,normalize_text,sha256_bytes

ROOT=Path(__file__).resolve().parents[1]


def verify_job(root, condition):
    root=Path(root);complete=json.loads((root/'ROBUSTNESS_COMPLETE.json').read_text())
    assert complete['condition']==condition
    raw=load_raw(ROOT)['test'].set_index('sample_id')
    bank=json.loads((root/'data/noise/bank.json').read_text())
    assert sha256_bytes(json.dumps(bank,sort_keys=True).encode())==complete['bank_sha256']
    train={r['variant'] for r in bank['rows'] if r['partition']=='train'};unseen={r['variant'] for r in bank['rows'] if r['partition']=='unseen'}
    assert train.isdisjoint(unseen)
    mapping={r['mapping_id']:r for r in bank['rows']}
    trace_events=0
    for view_path in sorted((root/'data/noise_views').glob('*/view.parquet')):
        profile=view_path.parent.name
        view=pd.read_parquet(view_path)
        assert len(view)==6680 and view.sample_id.is_unique
        for row in view.itertuples():
            assert row.text_original==normalize_text(raw.loc[row.sample_id].text_raw)
            edits=json.loads(row.edits) if isinstance(row.edits,str) else row.edits
            assert bool(edits)==bool(row.changed) and not row.label_preservation_verified
            text=row.text_original
            spans=sorted(edits,key=lambda e:e['start'])
            assert all(a['end']<=b['start'] for a,b in zip(spans,spans[1:]))
            for edit in reversed(spans):
                support=mapping[edit['mapping_id']]
                assert support['partition']==('unseen' if profile.startswith('empirical_unseen') else 'train')
                assert edit['source_ids']==support['source_ids'] and edit['kind']==support['kind']
                assert normalize_text(edit['original']).casefold()==support['canonical']
                assert edit['replacement'].casefold()==support['variant']
                assert text[edit['start']:edit['end']]==edit['original']
                text=text[:edit['start']]+edit['replacement']+text[edit['end']:]
                trace_events+=1
            assert text==row.text_noisy
            assert [c for c in text if ord(c)>0xffff]==[c for c in row.text_original if ord(c)>0xffff]
    run=root/'runs'/complete['run_name'];history=json.loads((run/'history.json').read_text())
    assert [r['epoch'] for r in history]==list(range(1,6)) and history[-1]['step']==3760
    selection=json.loads((run/'best/selection.json').read_text());scores=[r['validation']['macro_f1'] for r in history]
    assert selection['epoch']==int(np.argmax(scores))+1
    assert json.loads((root/'reports/fgm_gpu_smoke.json').read_text())['passed']
    results=[]
    for model in [condition,'baseline']+complete.get('extra_references',[]):
        folder=root/'robustness'/model;summary=json.loads((folder/'summary.json').read_text())
        for profile,record in summary.items():
            p=pd.read_parquet(folder/f'{profile}_predictions.parquet')
            assert len(p)==6680 and p.sample_id.is_unique
            source=raw.loc[p.sample_id]
            assert np.array_equal(p.label.to_numpy(),source.label.to_numpy())
            probs=p[['p_clean','p_offensive','p_hate']].to_numpy()
            assert np.isfinite(probs).all() and np.allclose(probs.sum(1),1,atol=1e-5)
            assert ((probs>=0)&(probs<=1)).all()
            assert np.array_equal(probs.argmax(1),p.prediction)
            f1=f1_score(p.label,p.prediction,labels=[0,1,2],average='macro',zero_division=0)
            accuracy=accuracy_score(p.label,p.prediction)
            assert abs(f1-record['macro_f1'])<1e-12 and abs(accuracy-record['accuracy'])<1e-12
            assert confusion_matrix(p.label,p.prediction,labels=[0,1,2]).tolist()==record['confusion_matrix']
            results.append({'condition':condition,'model':model,'profile':profile,'macro_f1':f1,'accuracy':accuracy,'coverage':record.get('coverage'),'actual_rate_mean':record.get('actual_rate_mean')})
        if model=='baseline':assert abs(summary['original']['macro_f1']-.6520143815686111)<1e-12
    if condition.startswith('augmentation'):
        epochs=[json.loads((root/'reports/augmentation'/f'epoch_{i}.json').read_text()) for i in range(1,6)]
        assert all(e['n']==24046 and 0<e['applied']<=e['selected']<=24046 for e in epochs)
    result={'condition':condition,'passed':True,'best_epoch':selection['epoch'],'profiles_verified':len(results),'test_examples_each':6680,'bank_sha256':complete['bank_sha256'],'replayed_evidence_backed_edit_events':trace_events}
    dump_json(ROOT/f'reports/ROBUSTNESS_VERIFY_{condition}.json',result)
    return result,pd.DataFrame(results)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('condition');p.add_argument('root');args=p.parse_args()
    verified,table=verify_job(args.root,args.condition);print(json.dumps(verified,indent=2))
    table.to_csv(ROOT/f'reports/robustness_results_{args.condition}.csv',index=False)
