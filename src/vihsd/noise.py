"""Corpus-grounded corruption: every substitution points to human-normalized data.

Synthetic toxic labels remain an explicit assumption. ViLexNorm provides lexical
normalization supervision, not independent toxic labels for newly generated text.
"""
from __future__ import annotations
import collections
import difflib
import hashlib
import json
import random
import re
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from .data import dump_json, normalize_text, sha256_bytes, supplementary_sequence

VILEXNORM_REVISION = '1070c6b9b00830fad278f48d9797df9ae1c942ac'
VERSION = 'vilexnorm-observed-v1'
# Protect register/target identity; no insertion/deletion or arbitrary synonyms.
PROTECTED = {'tôi', 'tao', 'mày', 'bạn', 'mình', 'nó', 'họ', 'ông', 'bà', 'anh', 'chị', 'em', 'chúng ta', 'chúng tôi'}
PUNCT = '.,!?;:"\'“”‘’()[]{}…'


def fetch_corpus(root):
    root = Path(root)
    folder = root / 'data/external/vilexnorm'
    folder.mkdir(parents=True, exist_ok=True)
    frames, manifest = {}, {'revision': VILEXNORM_REVISION, 'license': 'CC-BY-NC-SA-4.0', 'files': {}}
    for split, name in [('train', 'train'), ('validation', 'dev'), ('test', 'test')]:
        url = f'https://raw.githubusercontent.com/ngxtnhi/ViLexNorm/{VILEXNORM_REVISION}/data/{name}.csv'
        path = folder / f'{split}.csv'
        if not path.exists():
            with urllib.request.urlopen(url, timeout=120) as response:
                path.write_bytes(response.read())
        frame = pd.read_csv(path, keep_default_na=False)[['original', 'normalized']]
        if frame[['original', 'normalized']].map(lambda t: not isinstance(t, str) or not normalize_text(t)).any().any():
            raise ValueError(f'Invalid ViLexNorm {split} text')
        frame['source_id'] = [f'vilexnorm:{split}:{i:06d}' for i in range(len(frame))]
        frames[split] = frame
        manifest['files'][split] = {'url': url, 'sha256': sha256_bytes(path.read_bytes()), 'rows': len(frame)}
    dump_json(folder / 'manifest.json', manifest)
    return frames, manifest


def plain(text):
    text = text.replace('đ', 'd').replace('Đ', 'D')
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


def classify_pair(canonical, variant):
    if plain(canonical) == plain(variant):
        return 'diacritics'
    collapsed = re.sub(r'(.)\1+', r'\1', variant)
    if collapsed == re.sub(r'(.)\1+', r'\1', canonical) and len(variant) > len(canonical):
        return 'elongation'
    if len(variant.replace(' ', '')) < len(canonical.replace(' ', '')):
        return 'abbreviation'
    return 'spelling_variant'


def align_pairs(original, normalized, include_protected=False):
    """Anchor alignment, conservative short lexical spans; no paraphrase generation."""
    original = normalize_text(original)
    normalized = normalize_text(normalized)
    a, b = original.split(), normalized.split()
    result = []
    matcher = difflib.SequenceMatcher(a=[x.casefold() for x in a], b=[x.casefold() for x in b], autojunk=False)
    for tag, i, j, k, l in matcher.get_opcodes():
        if tag != 'replace' or not (1 <= j-i <= 4 and 1 <= l-k <= 4):
            continue
        variant = ' '.join(a[i:j]).strip(PUNCT).casefold()
        canonical = ' '.join(b[k:l]).strip(PUNCT).casefold()
        if not canonical or canonical == variant or (canonical in PROTECTED and not include_protected):
            continue
        if not re.fullmatch(r'[^\W\d_]+(?: [^\W\d_]+){0,3}', canonical, re.UNICODE):
            continue
        if not re.fullmatch(r'[^\W_]+(?: [^\W_]+){0,3}', variant, re.UNICODE):
            continue
        if max(len(variant), len(canonical)) > 45:
            continue
        # Reject broad semantic rewrites; keep spelling/compression supported by gold pairs.
        ratio = difflib.SequenceMatcher(a=plain(canonical).replace(' ', ''), b=plain(variant).replace(' ', '')).ratio()
        prefix_abbreviation = plain(canonical).replace(' ', '').startswith(plain(variant).replace(' ', ''))
        if (ratio < .35 and not prefix_abbreviation) or len(variant.replace(' ', '')) > 3 * len(canonical.replace(' ', '')):
            continue
        result.append({'canonical': canonical, 'variant': variant, 'kind': classify_pair(canonical, variant)})
    return result


def make_bank(corpus, vihsd_frames, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Exclude any source pair whose whole comment overlaps evaluation text.
    evaluation_texts = {normalize_text(t).casefold() for s in ['validation', 'test'] for t in vihsd_frames[s].text_raw}
    excluded, observations = [], []
    for row in corpus['train'].itertuples():
        if normalize_text(row.original).casefold() in evaluation_texts or normalize_text(row.normalized).casefold() in evaluation_texts:
            excluded.append(row.source_id)
            continue
        for pair in align_pairs(row.original, row.normalized, include_protected=True):
            observations.append({**pair, 'source_id': row.source_id})
    if not observations:
        raise ValueError('No supported normalization pairs')
    counts = collections.Counter((r['canonical'], r['variant']) for r in observations)
    variant_targets = collections.defaultdict(collections.Counter)
    for (canonical, variant), count in counts.items():
        variant_targets[variant][canonical] += count
    pair_sources = collections.defaultdict(list)
    for row in observations:
        pair_sources[row['canonical'], row['variant']].append(row['source_id'])
    joined_training = '\0'.join(normalize_text(t).casefold() for t in vihsd_frames['train'].text_raw)
    accepted, rejected = [], []
    for (canonical, variant), count in sorted(counts.items()):
        confidence = count / sum(variant_targets[variant].values())
        row = {'canonical': canonical, 'variant': variant, 'count': count, 'purity': confidence,
               'kind': classify_pair(canonical, variant), 'source_ids': sorted(set(pair_sources[canonical, variant]))}
        if canonical in PROTECTED:
            rejected.append({**row, 'reason': 'protected_register_or_target'})
            continue
        if confidence < .95:
            rejected.append({**row, 'reason': 'ambiguous_variant'})
            continue
        # Surface-level grouping prevents a held-out spelling entering through another target.
        bucket = int(sha256_bytes(variant.encode())[:8], 16) % 5
        row['partition'] = 'unseen' if bucket == 0 else 'train'
        if row['partition'] == 'unseen':
            surface_pattern = r'(?<!\w)' + r'\s+'.join(map(re.escape, variant.split())) + r'(?!\w)'
            if re.search(surface_pattern, joined_training):
                rejected.append({**row, 'reason': 'heldout_variant_present_in_vihsd_train'})
                continue
        # Rare observations retained only as held-out probes, never augmented as frequent forms.
        if count < 2 and row['partition'] == 'train':
            rejected.append({**row, 'reason': 'single_observation_train'})
            continue
        row['mapping_id'] = sha256_bytes((canonical + '\0' + variant).encode())[:16]
        accepted.append(row)
    train_surfaces = {r['variant'] for r in accepted if r['partition'] == 'train'}
    unseen_surfaces = {r['variant'] for r in accepted if r['partition'] == 'unseen'}
    assert train_surfaces.isdisjoint(unseen_surfaces)
    payload = {'version': VERSION, 'source_revision': VILEXNORM_REVISION, 'rows': accepted,
               'excluded_evaluation_overlap_source_ids': excluded,
               'label_preservation': 'assumed after conservative lexical filtering; not human-verified toxic labels'}
    dump_json(output / 'bank.json', payload)
    dump_json(output / 'rejected.json', rejected)
    pd.DataFrame(observations).to_csv(output / 'aligned_train_evidence.csv', index=False)
    return payload


class NoiseGenerator:
    def __init__(self, bank, partition='train'):
        self.rows = [r for r in bank['rows'] if r['partition'] == partition]
        self.partition = partition
        self.version = bank.get('version', VERSION)
        self.by_canonical = collections.defaultdict(list)
        for row in self.rows:
            self.by_canonical[row['canonical']].append(row)
        self.patterns = [(canonical, re.compile(r'(?<!\w)' + r'\s+'.join(map(re.escape, canonical.split())) + r'(?!\w)', re.IGNORECASE))
                         for canonical in sorted(self.by_canonical, key=lambda x: (-len(x), x))]
        self.variant_pattern = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(x) for x in sorted({r['variant'] for r in self.rows}, key=lambda x: -len(x))) + r')(?!\w)', re.IGNORECASE) if self.rows else None

    def candidate_spans(self, text):
        candidates = []
        for canonical, pattern in self.patterns:
            for match in pattern.finditer(text):
                # Avoid URLs/mentions, and substitutions inside existing nonstandard spans.
                if (match.start() and text[match.start()-1] in '@/#') or '://' in text[max(0, match.start()-12):match.start()]:
                    continue
                candidates.append((match.start(), match.end(), canonical))
        existing = [(m.start(), m.end()) for m in self.variant_pattern.finditer(text)] if self.variant_pattern else []
        accepted = []
        for start, end, canonical in sorted(candidates, key=lambda x: (-(x[1]-x[0]), x[0])):
            if any(start < b and end > a for a,b in existing):
                continue
            if not any(start < b and end > a for a,b,_ in accepted):
                accepted.append((start,end,canonical))
        return sorted(accepted)

    def generate(self, text, sample_id, seed, rate=None):
        text = normalize_text(text)
        rng = random.Random(int(sha256_bytes(f'{self.version}|{sample_id}|{seed}|{self.partition}'.encode())[:16],16))
        spans = self.candidate_spans(text)
        total_units = len(text.split())
        # Empirical mode chooses one observed lexical event, weighted by observed frequency.
        # This preserves event realism without pretending the corpus estimates total UGC noise.
        if rate is None:
            target_units = 1
        else:
            target_units = max(0, int(np.ceil(rate * total_units)))
        # Select eligible spans by observed event counts; this is not a population corruption propensity.
        ordered = []
        pool = list(spans)
        while pool:
            weights = [sum(r.get('sampling_weight',r['count']) for r in self.by_canonical[c]) for _,_,c in pool]
            chosen = rng.choices(range(len(pool)), weights=weights, k=1)[0]
            ordered.append(pool.pop(chosen))
        spans = ordered
        edits, changed_units = [], 0
        for start,end,canonical in spans:
            if changed_units >= target_units:
                break
            choices = self.by_canonical[canonical]
            row = rng.choices(choices, weights=[r.get('sampling_weight',r['count']) for r in choices], k=1)[0]
            source = text[start:end]
            replacement = row['variant']
            if source.isupper(): replacement = replacement.upper()
            elif source[0].isupper(): replacement = replacement[0].upper()+replacement[1:]
            if replacement == source:
                continue
            edits.append({'start':start,'end':end,'original':source,'replacement':replacement,
                          'mapping_id':row['mapping_id'],'kind':row['kind'],'source_ids':row['source_ids']})
            changed_units += len(source.split())
        noisy = text
        for edit in sorted(edits,key=lambda r:r['start'],reverse=True):
            noisy = noisy[:edit['start']]+edit['replacement']+noisy[edit['end']:]
        assert supplementary_sequence(noisy)==supplementary_sequence(text)
        return {'sample_id':sample_id,'text_original':text,'text_noisy':noisy,'seed':seed,'bank_partition':self.partition,
                'requested_rate':rate,'actual_rate':changed_units/max(1,total_units),'changed':bool(edits),
                'edits':edits,'label_preservation_verified':False}

    def detect(self, text):
        if not self.variant_pattern:
            return 0
        return len(list(self.variant_pattern.finditer(text)))


def generate_view(frame, generator, seed, rate=None):
    rows=[]
    for row in frame.itertuples():
        variant=generator.generate(row.text_raw,row.sample_id,seed,rate)
        rows.append({**variant,'label':int(row.label),'split':row.split})
    return pd.DataFrame(rows)


def noise_audit(corpus, bank, frames, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    gen=NoiseGenerator(bank)
    summary={'bank_counts':dict(collections.Counter(r['partition'] for r in bank['rows'])),
             'bank_kinds':dict(collections.Counter(r['kind'] for r in bank['rows'])),
             'excluded_evaluation_overlap_sources':len(bank['excluded_evaluation_overlap_source_ids']),
             'external_rows':{s:len(f) for s,f in corpus.items()},'cohorts':{},'validation_views':{}}
    for split,frame in frames.items():
        counts=frame.text_raw.map(gen.detect)
        summary['cohorts'][split]={'n':len(frame),'nsw_marker_rows':int((counts>0).sum()),'nsw_marker_rate':float((counts>0).mean()),
                                  'definition':'Heuristic frozen train-bank exact surface match; not gold noise density'}
        pd.DataFrame({'sample_id':frame.sample_id,'label':frame.label,'nsw_markers':counts}).to_csv(output/f'{split}_natural_cohorts.csv',index=False)
    for partition in ['train','unseen']:
        view=generate_view(frames['validation'],NoiseGenerator(bank,partition),101)
        view.assign(edits=view.edits.map(json.dumps)).to_parquet(output/f'validation_{partition}_review.parquet',index=False)
        summary['validation_views'][partition]={'n':len(view),'changed':int(view.changed.sum()),'coverage':float(view.changed.mean()),'actual_rate_mean':float(view.actual_rate.mean())}
    heldout={}
    for split in ['validation','test']:
        observations=[p for row in corpus[split].itertuples() for p in align_pairs(row.original,row.normalized)]
        train_pairs={(r['canonical'],r['variant']) for r in bank['rows'] if r['partition']=='train'}
        heldout[split]={'aligned_events':len(observations),'exact_pair_coverage':float(np.mean([(r['canonical'],r['variant']) in train_pairs for r in observations])) if observations else 0.,
                        'observed_kinds':dict(collections.Counter(r['kind'] for r in observations))}
    summary['external_heldout_diagnostics']=heldout
    dump_json(output/'noise_audit.json',summary)
    return summary
