"""Version 2: paired human sources, in-domain support and conditional event weights."""
from __future__ import annotations
import collections,difflib,json,re,urllib.request
from pathlib import Path
import pandas as pd
from .data import dump_json,normalize_text,sha256_bytes
from .noise import align_pairs,classify_pair,PROTECTED,PUNCT,VILEXNORM_REVISION
SMALL_REVISION='621b43c6e682df71214fec8c44836dd7a84ada52'


def fetch_in_domain(root):
    folder=Path(root)/'data/external/small_lexnorm_vihsd';folder.mkdir(parents=True,exist_ok=True)
    frames={};manifest={'revision':SMALL_REVISION,'source':'https://github.com/AnhHoang0529/Small-LexNormViHSD','files':{}}
    for split,name in [('train','train_data.json'),('test','test_data.json')]:
        url=f'https://raw.githubusercontent.com/AnhHoang0529/Small-LexNormViHSD/{SMALL_REVISION}/{name}';path=folder/name
        if not path.exists():
            with urllib.request.urlopen(url,timeout=120) as response:path.write_bytes(response.read())
        rows=json.loads(path.read_text());frames[split]=pd.DataFrame([{'original':' '.join(r['input']),'normalized':' '.join(r['output']),'source_id':f'smalllex:{split}:{r["id"]}'} for r in rows])
        manifest['files'][split]={'url':url,'sha256':sha256_bytes(path.read_bytes()),'rows':len(rows)}
    dump_json(folder/'source_manifest.json',manifest)
    return frames,manifest


def signature(text):return re.sub(r'\s+','',normalize_text(text).casefold())


def make_bank_v2(corpus,domain,frames,output,calibrate=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    eval_signatures={signature(t) for s in ['validation','test'] for t in frames[s].text_raw}
    sources=pd.concat([corpus['train'],domain['train']],ignore_index=True)
    accepted_sources=[];excluded=[];observations=[];identity=collections.Counter()
    for row in sources.itertuples():
        if signature(row.original) in eval_signatures or signature(row.normalized) in eval_signatures:
            excluded.append(row.source_id);continue
        accepted_sources.append(row)
        for pair in align_pairs(row.original,row.normalized,include_protected=True):observations.append({**pair,'source_id':row.source_id})
        a=[t.strip(PUNCT) for t in normalize_text(row.original).casefold().split()]
        b=[t.strip(PUNCT) for t in normalize_text(row.normalized).casefold().split()]
        for tag,i,j,k,l in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes():
            if tag=='equal':
                for n in range(1,5):
                    identity.update(' '.join(a[z:z+n]) for z in range(i,j-n+1))
    counts=collections.Counter((r['canonical'],r['variant']) for r in observations);targets=collections.defaultdict(collections.Counter);evidence=collections.defaultdict(list)
    for (canonical,variant),n in counts.items():targets[variant][canonical]+=n
    for r in observations:evidence[r['canonical'],r['variant']].append(r['source_id'])
    canonical_set={c for c,v in counts};opportunities=collections.Counter()
    for row in accepted_sources:
        tokens=[t.strip(PUNCT) for t in normalize_text(row.normalized).casefold().split()]
        for n in range(1,5):
            for i in range(len(tokens)-n+1):
                phrase=' '.join(tokens[i:i+n])
                if phrase in canonical_set:opportunities[phrase]+=1
    totals=collections.Counter()
    for (c,v),n in counts.items():totals[c]+=n
    joined='\0'.join(normalize_text(t).casefold() for t in frames['train'].text_raw)
    rows=[];rejected=[]
    for (canonical,variant),n in sorted(counts.items()):
        denominator=sum(targets[variant].values())+identity[variant]
        row={'canonical':canonical,'variant':variant,'count':n,'purity':n/max(1,denominator),'kind':classify_pair(canonical,variant),'source_ids':sorted(set(evidence[canonical,variant])),'identity_observations':identity[variant]}
        if canonical in PROTECTED or row['purity']<.95:
            rejected.append({**row,'reason':'protected_or_context_ambiguous_including_unchanged_uses'});continue
        pattern=r'(?<!\w)'+r'\s+'.join(map(re.escape,variant.split()))+r'(?!\w)'
        appears_in_training=bool(re.search(pattern,joined))
        partition='unseen' if int(sha256_bytes(variant.encode())[:8],16)%5==0 and not appears_in_training else 'train'
        if partition=='train' and (n<2 or len(row['source_ids'])<2):
            rejected.append({**row,'reason':'insufficient_distinct_train_evidence'});continue
        row.update(partition=partition,mapping_id=sha256_bytes((canonical+'\0'+variant).encode())[:16],canonical_opportunities=opportunities[canonical],sampling_weight=n/max(opportunities[canonical],totals[canonical],1))
        row['conditional_support_weight']=row['sampling_weight']
        rows.append(row)
    assert {r['variant'] for r in rows if r['partition']=='train'}.isdisjoint({r['variant'] for r in rows if r['partition']=='unseen'})
    bank={'version':'human-dual-source-conditional-v2','source_revision':VILEXNORM_REVISION,'in_domain_revision':SMALL_REVISION,'rows':rows,'excluded_evaluation_overlap_source_ids':excluded,'label_preservation':'assumed after lexical filtering; not reannotated toxic labels','weighting':'count(canonical,variant) / canonical opportunities; identity uses included in ambiguity filter','sources':{'vilexnorm_train':len(corpus['train']),'smalllexnorm_train':len(domain['train'])}}
    if calibrate:
        bank=calibrate_event_types(bank,domain['train'],frames['train'])
    dump_json(output/'bank.json',bank);dump_json(output/'rejected.json',rejected)
    pd.DataFrame(observations).to_csv(output/'aligned_dual_source_evidence.csv',index=False)
    return bank


def calibrate_event_types(bank,human_train,vihsd_train):
    """Learn family multipliers using human TRAIN proportions and target TRAIN opportunities.

    No classifier, validation/test label or held-out source frequency enters fitting.
    """
    import numpy as np
    from .noise import NoiseGenerator
    kinds=['abbreviation','diacritics','elongation','spelling_variant']
    reference=collections.Counter(p['kind'] for row in human_train.itertuples() for p in align_pairs(row.original,row.normalized))
    target=np.array([reference[k] for k in kinds],dtype=float)
    if target.sum()==0:raise ValueError('Human training reference has no aligned events')
    target/=target.sum()
    generator=NoiseGenerator(bank)
    sample=sorted(vihsd_train.itertuples(),key=lambda r:sha256_bytes(r.sample_id.encode()))[:2000]
    matrix=[]
    for row in sample:
        weights=np.zeros(len(kinds))
        for _,_,canonical in generator.candidate_spans(normalize_text(row.text_raw)):
            for mapping in generator.by_canonical[canonical]:
                weights[kinds.index(mapping['kind'])]+=mapping['sampling_weight']
        if weights.sum()>0:matrix.append(weights)
    if not matrix:
        raise ValueError('No eligible calibration examples')
    matrix=np.asarray(matrix)
    before=(matrix/matrix.sum(1,keepdims=True)).mean(0)
    from scipy.optimize import minimize
    from scipy.spatial.distance import jensenshannon
    def objective(log_weights):
        weighted=matrix*np.exp(log_weights)
        actual=(weighted/weighted.sum(1,keepdims=True)).mean(0)
        return float(jensenshannon(target,actual,base=2)**2+.0005*np.mean(log_weights**2))
    fitted=minimize(objective,np.zeros(len(kinds)),method='L-BFGS-B',bounds=[(-3.,3.)]*len(kinds),options={'ftol':1e-12,'gtol':1e-8,'maxiter':200})
    multipliers=np.round(np.exp(fitted.x),6)
    for row in bank['rows']:
        row['conditional_support_weight']=row['sampling_weight']
        row['sampling_weight']=round(row['sampling_weight']*float(multipliers[kinds.index(row['kind'])]),10)
    after_matrix=matrix*multipliers
    after=(after_matrix/after_matrix.sum(1,keepdims=True)).mean(0)
    bank['version']='human-dual-source-calibrated-v2'
    bank['type_calibration']={'reference':'Small-LexNormViHSD released train only','target_domain_sample':'2000 hash-selected ViHSD train comments; no labels used','eligible_comments':len(matrix),'kind_order':kinds,'human_train_prior':np.round(target,8).tolist(),'before_expected_mix':np.round(before,8).tolist(),'after_expected_mix':np.round(after,8).tolist(),'multipliers':multipliers.tolist(),'iterations':int(fitted.nit),'optimizer_converged':bool(fitted.success),'objective':'JSD to human TRAIN event-type prior + 0.0005 mean squared log multipliers','log_multiplier_bounds':[-3,3],'heldout_data_used':False,'limitations':'Conditional on eligible mapping spans, source corpus is NSW-selected; not a universal UGC error process'}
    return bank
