"""Pinned author data, immutable raw text and explicit split audits."""
from __future__ import annotations
import hashlib
import io
import json
import re
import unicodedata
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA_COMMIT = '70fe50c9c94145e3b9be5614a048cbd5b480fa54'
DATA_URL = f'https://raw.githubusercontent.com/sonlam1102/vihsd/{DATA_COMMIT}/data/vihsd.zip'
# Official author repository blob fb17d90cbcb731b173f4f76977b3dfc78420ac7d.
EXPECTED_SHA256 = '1d823c4c86a8aede83555c64da4e1be24146967aac6b30cdaf8603d4910f328d'
SPLITS = {'train': 'train', 'validation': 'dev', 'test': 'test'}
EXPECTED_COUNTS = {'train': 24048, 'validation': 2672, 'test': 6680}
LABELS = ['CLEAN', 'OFFENSIVE', 'HATE']
PREPROCESS_VERSION = 'nfc-space-rdr-utf8-v2'
KNOWN_EMPTY_TRAIN_IDS = {'train:010950', 'train:020880'}


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError('Text must be a string; null must not become "nan"')
    text = unicodedata.normalize('NFC', text)
    # Preserve emoji joiners/variation selectors; remove only actual control chars.
    text = ''.join(' ' if unicodedata.category(c) == 'Cc' else c for c in text)
    return re.sub(r'\s+', ' ', text).strip()


def download(root: str | Path) -> Path:
    raw = Path(root) / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / 'vihsd.zip'
    if not archive.exists():
        tmp = archive.with_suffix('.tmp')
        with urllib.request.urlopen(DATA_URL, timeout=120) as response:
            tmp.write_bytes(response.read())
        tmp.replace(archive)
    content = archive.read_bytes()
    checksum = sha256_bytes(content)
    if EXPECTED_SHA256 != 'TO_BE_PINNED' and checksum != EXPECTED_SHA256:
        raise ValueError(f'Dataset checksum mismatch: {checksum}')
    manifest = {'source_url': DATA_URL, 'source_commit': DATA_COMMIT,
                'sha256': checksum, 'observed_at_utc': datetime.now(timezone.utc).isoformat(),
                'labels': dict(enumerate(LABELS)), 'files': {}}
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for split, source_name in SPLITS.items():
            member = f'vihsd/{source_name}.csv'
            data = z.read(member)
            target = raw / f'{split}.csv'
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f'Refusing to overwrite modified raw file: {target}')
            target.write_bytes(data)
            manifest['files'][split] = {'member': member, 'sha256': sha256_bytes(data)}
    dump_json(raw / 'manifest.json', manifest)
    return raw


def canonicalize(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    required = {'free_text', 'label_id'}
    if not required.issubset(frame.columns):
        raise ValueError(f'{split}: expected columns {sorted(required)}')
    out = frame[['free_text', 'label_id']].rename(columns={'free_text': 'text_raw', 'label_id': 'label'}).copy()
    out.insert(0, 'source_row', range(len(out)))
    out.insert(0, 'split', split)
    out.insert(0, 'sample_id', [f'{split}:{i:06d}' for i in range(len(out))])
    return out


def load_raw(root: str | Path) -> dict[str, pd.DataFrame]:
    raw = download(root)
    return {split: canonicalize(pd.read_csv(raw / f'{split}.csv', keep_default_na=False), split)
            for split in SPLITS}


def audit(frames: dict[str, pd.DataFrame], output: str | Path, enforce_counts=True) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    exclusions = []
    summaries = {}
    all_rows = []
    for split, frame in frames.items():
        valid_label = frame.label.isin([0, 1, 2])
        valid_text = frame.text_raw.map(lambda x: isinstance(x, str) and bool(normalize_text(x)))
        for row in frame.loc[~valid_label | ~valid_text].itertuples():
            issue = {'sample_id': row.sample_id, 'invalid_label': not bool(valid_label.loc[row.Index]), 'invalid_text': not bool(valid_text.loc[row.Index])}
            if split == 'train' and row.sample_id in KNOWN_EMPTY_TRAIN_IDS and issue['invalid_text'] and not issue['invalid_label']:
                exclusions.append({**issue, 'reason': 'Empty author-provided train text; exclude from training only'})
            else:
                errors.append(issue)
        summaries[split] = {'n': len(frame), 'label_counts': {str(k): int((frame.label == k).sum()) for k in range(3)},
                            'invalid_label': int((~valid_label).sum()), 'invalid_text': int((~valid_text).sum()),
                            'replacement_char_rows': int(frame.text_raw.map(lambda x: '\ufffd' in str(x)).sum()),
                            'raw_underscore_rows': int(frame.text_raw.map(lambda x: '_' in str(x)).sum())}
        if enforce_counts and len(frame) != EXPECTED_COUNTS[split]:
            errors.append({'split': split, 'reason': f'Expected {EXPECTED_COUNTS[split]}, got {len(frame)}'})
        f = frame.copy()
        f['raw_hash'] = f.text_raw.map(lambda x: sha256_bytes(str(x).encode()))
        f['normalized_hash'] = f.text_raw.map(lambda x: sha256_bytes(normalize_text(x).encode()) if isinstance(x, str) else '')
        all_rows.append(f)
    combined = pd.concat(all_rows, ignore_index=True)
    duplicate_stats = {}
    for mode in ['raw', 'normalized']:
        key = mode + '_hash'
        duplicates = combined[combined.duplicated(key, keep=False)].copy()
        duplicates.to_csv(output / f'duplicates_{mode}.csv', index=False)
        groups = combined.groupby(key)
        cross_hashes = groups.split.nunique()
        conflicts = groups.label.nunique()
        cross = combined[combined[key].isin(cross_hashes[cross_hashes > 1].index)]
        cross.to_csv(output / f'cross_split_{mode}.csv', index=False)
        combined[combined[key].isin(conflicts[conflicts > 1].index)].to_csv(output / f'label_conflicts_{mode}.csv', index=False)
        duplicate_stats[mode] = {'duplicate_rows': len(duplicates), 'duplicate_groups': int(duplicates[key].nunique()),
                                 'cross_split_groups': int((cross_hashes > 1).sum()),
                                 'cross_split_rows': len(cross), 'conflicting_label_groups': int((conflicts > 1).sum())}
    summary = {'splits': summaries, 'duplicates': duplicate_stats, 'blocking_errors': errors, 'training_exclusions': exclusions,
               'policy': 'Preserve official rows/splits; report duplicates and conflicts. Never silently drop rows.'}
    dump_json(output / 'audit.json', summary)
    dump_json(output / 'training_exclusions.json', exclusions)
    if errors:
        dump_json(output / 'blocking_errors.json', errors)
        raise ValueError(f'Data audit failed; see {output}/blocking_errors.json')
    return summary


def preserve_java_unicode(segmenter):
    """Avoid pyjnius String(str) truncating supplementary Unicode to 16 bits.

    Java String(byte[], UTF-8) roundtrips emoji and other non-BMP characters.
    Patching this constructor preserves VnCoreNLP's official word_segment logic.
    """
    java_string = segmenter.javaclass_String
    segmenter.javaclass_String = lambda text: java_string(list(text.encode('utf-8')), 'UTF-8')
    return segmenter


def supplementary_sequence(text):
    return [c for c in text if ord(c) > 0xFFFF]


def prepare(root: str | Path, segmenter_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """One sequential JVM, cached by full data/model/preprocessing fingerprint."""
    import importlib.metadata
    import py_vncorenlp
    from tqdm.auto import tqdm
    root = Path(root).resolve()
    frames = load_raw(root)
    audit(frames, root / 'reports/audit')
    frames['train'] = frames['train'].loc[~frames['train'].sample_id.isin(KNOWN_EMPTY_TRAIN_IDS)].copy().reset_index(drop=True)
    model_dir = Path(segmenter_dir or root / '.cache/vncorenlp').resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    if not (model_dir / 'VnCoreNLP-1.2.jar').exists():
        py_vncorenlp.download_model(save_dir=str(model_dir))
    resources = sorted(p for p in model_dir.rglob('*') if p.is_file())
    resource_hashes = {str(p.relative_to(model_dir)): sha256_bytes(p.read_bytes()) for p in resources}
    signature = {'data_sha256': sha256_bytes((root / 'data/raw/vihsd.zip').read_bytes()),
                 'preprocessing': PREPROCESS_VERSION, 'py_vncorenlp': importlib.metadata.version('py-vncorenlp'),
                 'resource_hashes': resource_hashes}
    signature_hash = sha256_bytes(json.dumps(signature, sort_keys=True).encode())
    target = root / 'data/processed' / signature_hash[:16]
    metadata = target / 'manifest.json'
    if metadata.exists() and all((target / f'{s}.parquet').exists() for s in SPLITS):
        if json.loads(metadata.read_text()) == signature:
            return {s: pd.read_parquet(target / f'{s}.parquet') for s in SPLITS}
    target.mkdir(parents=True, exist_ok=True)
    segmenter = get_segmenter(model_dir)
    result = {}
    for split, frame in frames.items():
        frame = frame.copy()
        frame['text_normalized'] = frame.text_raw.map(normalize_text)
        segmented = []
        for row in tqdm(frame.itertuples(), total=len(frame), desc=f'Segment {split}'):
            try:
                text = ' '.join(segmenter.word_segment(row.text_normalized)).strip()
                if not text:
                    raise ValueError('Empty segmenter output')
                if supplementary_sequence(row.text_normalized) != supplementary_sequence(text):
                    raise ValueError('Supplementary Unicode changed during segmentation')
            except Exception as exc:
                raise RuntimeError(f'Segmentation failed for {row.sample_id}') from exc
            segmented.append(text)
        frame['text_segmented'] = segmented
        frame['preprocess_version'] = PREPROCESS_VERSION
        frame.to_parquet(target / f'{split}.parquet', index=False)
        result[split] = frame
    dump_json(metadata, signature)
    dump_json(root / 'data/processed/latest.json', {'path': str(target), 'fingerprint': signature_hash})
    sample = pd.concat([result['train'].query('label == @label').sample(30, random_state=42) for label in range(3)])
    sample.to_csv(root / 'reports/preprocessing_train_examples.csv', index=False)
    return result


_SEGMENTER = None

def get_segmenter(model_dir):
    """Share one JVM-backed segmenter across clean/noisy views and augmentation."""
    global _SEGMENTER
    if _SEGMENTER is not None:
        return _SEGMENTER
    import os
    import py_vncorenlp
    previous_cwd = Path.cwd()
    try:
        _SEGMENTER = py_vncorenlp.VnCoreNLP(max_heap_size='-Xmx1g', annotators=['wseg'], save_dir=str(Path(model_dir).resolve()))
    finally:
        os.chdir(previous_cwd)
    preserve_java_unicode(_SEGMENTER)
    return _SEGMENTER


def segment_text(text, model_dir):
    normalized = normalize_text(text)
    segmented = ' '.join(get_segmenter(model_dir).word_segment(normalized)).strip()
    if not segmented or supplementary_sequence(normalized) != supplementary_sequence(segmented):
        raise ValueError('Empty segmentation or supplementary Unicode changed')
    return segmented
