"""Exercise capacity reservation and duplicate-submission protection without Kaggle."""
import copy,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('full20_dispatch',ROOT/'scripts/launch_full20.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_external_sessions_reserved_and_submissions_not_duplicated(monkeypatch):
 state={'jobs':[{'job':f'{alias}-{n}','bridge_alias':alias,'status':'PREPARED','kernel':f'{alias}/job-{n}','version':1} for alias in ['a','b','c','d','e'] for n in range(2)],'external_kernels':{alias:([f'{alias}/external'] if alias!='e' else []) for alias in ['a','b','c','d','e']},'max_sessions_per_account':2}
 pushed=[]
 monkeypatch.setattr(m,'save',lambda s:None)
 monkeypatch.setattr(m,'remote_status',lambda *a:'RUNNING')
 def submit(s,j):pushed.append(j['job']);j['status']='SUBMITTED'
 monkeypatch.setattr(m,'submit',submit)
 m.step(state)
 assert len(pushed)==6
 assert all(sum(j['bridge_alias']==a and j['status']=='SUBMITTED' for j in state['jobs'])==1 for a in ['a','b','c','d'])
 assert sum(j['bridge_alias']=='e' and j['status']=='SUBMITTED' for j in state['jobs'])==2
 m.step(state)
 assert len(pushed)==6
 # An ambiguous push blocks additional mutations on that account.
 state['jobs'][0]['status']='SUBMISSION_UNKNOWN'
 m.step(state)
 assert len(pushed)==6
