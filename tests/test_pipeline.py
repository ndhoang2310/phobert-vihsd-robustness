import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from vihsd.data import canonicalize, normalize_text, audit, load_raw, EXPECTED_SHA256, sha256_bytes
from vihsd.training import metrics, balanced_indices, make_loader, tokenize_frames


def make_frames():
    return {s: canonicalize(pd.DataFrame({'free_text': [f'{s} văn bản {i}' for i in range(6)],
                                         'label_id': [0, 1, 2, 0, 1, 2]}), s)
            for s in ['train', 'validation', 'test']}


def test_normalization_preserves_ugc():
    assert normalize_text('  KHÔNG\tđẹpppp 😅 👨\u200d👩\u200d👧\u200d👦\n#abc https://x.vn ') == 'KHÔNG đẹpppp 😅 👨\u200d👩\u200d👧\u200d👦 #abc https://x.vn'
    assert normalize_text('Việt') == 'Việt'
    with pytest.raises(ValueError): normalize_text(None)


def test_adapter_identity():
    frames = make_frames()
    assert frames['train'].sample_id.tolist() == [f'train:{i:06d}' for i in range(6)]
    assert set(frames['train'].sample_id).isdisjoint(frames['test'].sample_id)
    with pytest.raises(ValueError): canonicalize(pd.DataFrame({'text': ['x']}), 'train')


def test_audit_duplicates_and_conflicts(tmp_path):
    frames = make_frames()
    frames['test'].loc[0, 'text_raw'] = frames['train'].loc[0, 'text_raw']
    frames['test'].loc[0, 'label'] = 2
    result = audit(frames, tmp_path, enforce_counts=False)
    assert result['duplicates']['raw']['cross_split_groups'] == 1
    assert result['duplicates']['raw']['conflicting_label_groups'] == 1
    assert (tmp_path / 'cross_split_normalized.csv').exists()


@pytest.mark.parametrize('column,value', [('text_raw', ''), ('text_raw', None), ('label', 3), ('label', -1)])
def test_invalid_rows_block(tmp_path, column, value):
    frames = make_frames()
    frames['train'].loc[0, column] = value
    with pytest.raises(ValueError): audit(frames, tmp_path, enforce_counts=False)
    assert json.loads((tmp_path / 'audit.json').read_text())['blocking_errors']


def test_known_empty_training_exclusion_is_explicit(tmp_path):
    frames = make_frames()
    frames['train'].loc[0, 'sample_id'] = 'train:010950'
    frames['train'].loc[0, 'text_raw'] = ''
    result = audit(frames, tmp_path, enforce_counts=False)
    assert len(result['training_exclusions']) == 1
    assert not result['blocking_errors']
    assert len(frames['train']) == 6  # audit never drops rows


def test_macro_f1_not_micro():
    result = metrics([0, 0, 1, 2], [0, 0, 0, 0])
    assert result['accuracy'] == .5
    assert result['macro_f1'] == pytest.approx((2 / 3) / 3)
    assert list(result['per_class'])[:3] == ['CLEAN', 'OFFENSIVE', 'HATE']
    assert np.asarray(result['confusion_matrix']).shape == (3, 3)


def test_balanced_subset():
    frame = make_frames()['train']
    indices = balanced_indices(frame, 1)
    assert len(indices) == 3
    assert set(frame.loc[indices].label) == {0, 1, 2}
    assert np.array_equal(indices, balanced_indices(frame, 1))


def test_real_data_integrity(tmp_path):
    archive = Path('data/raw/vihsd.zip')
    if not archive.exists(): pytest.skip('Official data not downloaded')
    assert sha256_bytes(archive.read_bytes()) == EXPECTED_SHA256
    frames = load_raw('.')
    report = audit(frames, tmp_path)
    assert len(frames['train']) == 24048
    assert len(frames['test']) == 6680
    assert len(report['training_exclusions']) == 2
    assert not report['blocking_errors']


def test_real_tokenization_and_padding(tmp_path):
    from transformers import AutoTokenizer, RobertaConfig, RobertaForSequenceClassification
    import torch
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base', use_fast=False)
    frames = make_frames()
    for frame in frames.values(): frame['text_segmented'] = ['Xin chào .', 'không ' * 300, 'đẹpppp 😅', 'học_sinh', 'a', 'b']
    encoded, diagnostics = tokenize_frames(frames, tokenizer, 256, tmp_path)
    assert len(encoded['train'][1]['input_ids']) == 256
    assert encoded['train'][1]['input_ids'][-1] == tokenizer.eos_token_id
    assert diagnostics['train']['lengths'][1] > 256
    batch = next(iter(make_loader(encoded['train'], tokenizer, 3)))
    assert batch['labels'].dtype == torch.long
    assert (batch['attention_mask'][batch['input_ids'] == tokenizer.pad_token_id] == 0).all()
    config = RobertaConfig(vocab_size=tokenizer.vocab_size, hidden_size=24, num_hidden_layers=1,
                          num_attention_heads=2, intermediate_size=32, max_position_embeddings=258,
                          pad_token_id=tokenizer.pad_token_id, num_labels=3)
    model = RobertaForSequenceClassification(config)
    result = model(**batch)
    assert result.logits.shape == (3, 3)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert model.roberta.embeddings.word_embeddings.weight.grad is not None


def test_unicode_bridge_uses_utf8_bytes():
    from vihsd.data import preserve_java_unicode, supplementary_sequence
    class Wrapper:
        javaclass_String = staticmethod(lambda data, encoding: bytes(data).decode(encoding))
    wrapper = preserve_java_unicode(Wrapper())
    text = 'Việt 😂 🥰 👨\u200d👩\u200d👧\u200d👦 𠀀'
    assert wrapper.javaclass_String(text) == text
    assert supplementary_sequence(text) == ['😂', '🥰', '👨', '👩', '👧', '👦', '𠀀']


def test_real_java_segmentation_preserves_emoji():
    import subprocess,sys
    if not Path('.cache/vncorenlp/VnCoreNLP-1.2.jar').exists():
        pytest.skip('Segmenter resources not downloaded')
    code = '''
from pathlib import Path
from vihsd.pipeline import ensure_java
from vihsd.data import preserve_java_unicode, supplementary_sequence
ensure_java()
import py_vncorenlp
segmenter = py_vncorenlp.VnCoreNLP(annotators=['wseg'], save_dir=str(Path('.cache/vncorenlp').resolve()))
preserve_java_unicode(segmenter)
for text in ['😂🥰🥰', 'Việt Nam 😅 👨\\u200d👩\\u200d👧\\u200d👦', 'Xin chào 𠀀 bạn']:
    result = ' '.join(segmenter.word_segment(text))
    assert supplementary_sequence(result) == supplementary_sequence(text), (text, result)
print('UNICODE_SEGMENTATION_OK')
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'UNICODE_SEGMENTATION_OK' in result.stdout
