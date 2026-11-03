import json
import pandas as pd
import pytest
from vihsd.full20_eval import view_sha,locate,prepare_views
from vihsd import full20_eval

def test_view_hash_preserves_semantics_and_detects_changed_benchmark():
 frame=pd.DataFrame([{'sample_id':'test:0','label':2,'text_original':'câu 😃','text_noisy':'cau 😃','changed':True,'edits':[{'start':0,'end':3,'original':'câu','replacement':'cau','source_ids':['x']}]}])
 digest=view_sha(frame)
 serialized=frame.copy();serialized['edits']=serialized.edits.map(json.dumps)
 assert view_sha(serialized)==digest
 serialized.loc[0,'text_noisy']='khác 😃'
 assert view_sha(serialized)!=digest

def test_missing_private_input_blocks_before_gpu_or_test(tmp_path):
 with pytest.raises(RuntimeError,match='exactly one'):
  locate({'condition':'fgm','seed':42,'job':'fgm-s42'},tmp_path)

def test_wrong_frozen_view_blocks_before_cache_or_test(monkeypatch,tmp_path):
 frame=pd.DataFrame([{'sample_id':'test:0','label':0,'text_original':'a','text_noisy':'a','changed':False,'edits':[]}])
 monkeypatch.setattr(full20_eval,'NoiseGenerator',lambda *a:None)
 monkeypatch.setattr(full20_eval,'encode_noisy_view',lambda *a:([],frame))
 with pytest.raises(AssertionError,match='view mismatch'):
  prepare_views(tmp_path,{'test':frame},{'test':[]},None,{}, {'empirical_train_s201':'bad'})
 assert not (tmp_path/'VIEW_HASHES.json').exists()
