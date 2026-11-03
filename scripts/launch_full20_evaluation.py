"""Bridge-only evaluation dispatcher with automatic download, verification and report."""
import argparse,datetime,fcntl,json,os,re,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];STATE=ROOT/'reports/FULL20_EVALUATION_RUNS.json';OUT=ROOT/'artifacts/full20-evaluation';BRIDGE='/home/hoang/.local/bin/kaggle-account'
class Canceled(RuntimeError):pass

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(s):
 s['updated_at']=now();tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(STATE)
def call(alias,args,label):
 print(now(),alias,label,flush=True)
 p=subprocess.run([BRIDGE,'run',alias,'--',*args],capture_output=True,text=True,timeout=240)
 output=p.stdout+'\n'+p.stderr;(OUT/f'{label}.log').write_text(output);print(output.strip(),flush=True)
 if p.returncode and re.search(r'cancel|hủy|huỷ',output,re.I):raise Canceled(label)
 return p.returncode,output

def status(alias,ref,label):
 code,out=call(alias,['kernels','status',ref],label);m=re.search(r'KernelWorkerStatus\.(COMPLETE|RUNNING|QUEUED|ERROR|CANCEL_ACKNOWLEDGED)',out)
 if code or not m:raise RuntimeError('Remote status unavailable: '+label)
 return m.group(1)

def step(s):
 for g in s['groups']:
  alias=g['bridge_alias'];name=g['group']
  if g['status'] in ['SUBMITTED','RUNNING','QUEUED']:
   remote=status(alias,g['kernel']+'/'+str(g['version']),name+'-status');g['status']='REMOTE_COMPLETE' if remote=='COMPLETE' else remote;save(s)
   if remote=='ERROR':call(alias,['kernels','logs',g['kernel']+'/'+str(g['version'])],name+'-error-log')
  if g['status']=='PREPARED':
   active=0
   for ref in s['external_kernels'][alias]:
    if status(alias,ref,'external-'+alias) in ['RUNNING','QUEUED']:active+=1
   if active>=2:continue
   folder=Path(g['folder']);m=json.loads((folder/'kernel-metadata.json').read_text())
   assert m['id']==g['kernel'] and m['is_private'] and m['enable_gpu'] and 'id_no' not in m
   assert set(m['kernel_sources'])=={j['kernel']+'/'+str(j['version']) for j in g['jobs']}
   assert all(j['kernel'].split('/')[0]==g['kernel'].split('/')[0] for j in g['jobs'])
   g.update(status='SUBMISSION_UNKNOWN',last_attempt=now());save(s)
   code,out=call(alias,['kernels','push','-p',str(folder),'--accelerator','NvidiaTeslaT4','-t',str(g['session_timeout_seconds'])],name+'-push')
   v=re.search(r'Kernel version (\d+) successfully pushed',out);url=re.search(r'https://www\.kaggle\.com/code/([^\s]+)',out)
   if v and url and url.group(1).rstrip('/')==g['kernel']:g.update(status='SUBMITTED',version=int(v.group(1)),submitted_at=now())
   elif code and 'quota' in out.lower():g['status']='WAITING_QUOTA'
   save(s)
  if g['status']=='REMOTE_COMPLETE':
   target=OUT/name
   pattern=r'(EVALUATION_COMPLETE\.json$|VIEW_HASHES\.json$|environment\.json$|data/noise/bank\.json$|data/noise_views/.*/view\.parquet$|validation_gate/.*\.(?:json|parquet)$|robustness/.*\.(?:json|parquet)$)'
   code,out=call(alias,['kernels','output',g['kernel']+'/'+str(g['version']),'-p',str(target),'--file-pattern',pattern,'--quiet','--force','--page-size','200'],name+'-download')
   if code:raise RuntimeError('Evaluation output download failed: '+name)
   from verify_full20_evaluation import verify
   try:check=verify(target/'full20_evaluation',g,s)
   except Exception:
    g.update(status='VERIFICATION_FAILED',last_error=traceback.format_exc());save(s);raise
   g.update(status='COMPLETE',verified=True,verification=check,completed_at=now());save(s)
 all_done=all(g['status']=='COMPLETE' and g.get('verified') for g in s['groups'])
 s.update(status='VERIFIED' if all_done else 'NEEDS_ATTENTION' if any(g['status'] in ['ERROR','VERIFICATION_FAILED','SUBMISSION_UNKNOWN','WAITING_QUOTA','CANCEL_ACKNOWLEDGED'] for g in s['groups']) else 'RUNNING',last_cycle=now());save(s)
 print('EVALUATION_STATE',json.dumps({g['group']:g['status'] for g in s['groups']}),flush=True)
 if all_done:
  from summarize_full20_evaluation import summarize
  summarize(s);s.update(status='COMPLETE',report=str(ROOT/'reports/FULL20_EVALUATION_RESULTS.md'),completed_at=now());save(s)
 return all_done

def main():
 p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--interval',type=int,default=120);p.add_argument('--max-hours',type=float,default=12);args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'worker.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s=json.loads(STATE.read_text());start=time.monotonic()
  s['worker']={'pid':os.getpid(),'status':'ACTIVE','started_at':now(),'interval_seconds':args.interval};save(s)
  try:
   while True:
    if step(s) or args.once:break
    if s['status']=='NEEDS_ATTENTION':break
    if time.monotonic()-start>args.max_hours*3600:s['status']='MONITOR_DEADLINE_REACHED';break
    time.sleep(args.interval)
  except Canceled:s['status']='LOCAL_UNLOCK_CANCELED';raise
  except BaseException:s['worker']['last_error']=traceback.format_exc();raise
  finally:s['worker'].update(status='STOPPED',ended_at=now());save(s)
if __name__=='__main__':main()
