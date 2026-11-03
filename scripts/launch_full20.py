"""Launch/monitor the full20 campaign using only the existing credential bridge.

One locked writer; no credentials, account rotation, remote cancellation or blind
mutation retries. Other projects retain their sessions. A detached worker may
finish the explicitly authorized nine jobs while the desktop session stays up.
"""
import argparse,datetime,fcntl,json,os,re,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'reports/FULL20_RUNS.json'
ARTIFACTS=ROOT/'artifacts/full20'
BRIDGE='/home/hoang/.local/bin/kaggle-account'
ACTIVE={'SUBMITTED','RUNNING','QUEUED','SUBMISSION_UNKNOWN'}
FAILED={'ERROR','CANCEL_ACKNOWLEDGED','SUBMISSION_UNKNOWN','WAITING_QUOTA','NEEDS_REVIEW','VERIFICATION_FAILED'}
class DialogCanceled(RuntimeError):pass

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(state):
 state['updated_at']=now();tmp=STATE.with_suffix('.tmp')
 tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATE)
def call(alias,args,label):
 print(now(),alias,label,flush=True)
 p=subprocess.run([BRIDGE,'run',alias,'--',*args],capture_output=True,text=True,timeout=240)
 output=p.stdout+'\n'+p.stderr
 ARTIFACTS.mkdir(parents=True,exist_ok=True)
 (ARTIFACTS/f'{label}.log').write_text(output)
 print(output.strip(),flush=True)
 if p.returncode and re.search(r'cancel|hủy|huỷ',output,re.I):raise DialogCanceled(label)
 return p.returncode,output

def remote_status(alias,ref,label):
 code,output=call(alias,['kernels','status',ref],label)
 m=re.search(r'KernelWorkerStatus\.(COMPLETE|RUNNING|QUEUED|ERROR|CANCEL_ACKNOWLEDGED)',output)
 if code or not m:raise RuntimeError(f'Cannot establish remote status: {label}; inspect redacted bridge log')
 return m.group(1)

def submit(state,job):
 folder=Path(job['folder']);metadata=json.loads((folder/'kernel-metadata.json').read_text())
 assert metadata['id']==job['kernel'] and metadata['is_private'] and metadata['enable_gpu'] and 'id_no' not in metadata
 assert job['bridge_alias']!='huu'
 # Record before sending to avoid accidental duplicate submissions after a crash.
 job.update(status='SUBMISSION_UNKNOWN',last_attempt=now());save(state)
 code,output=call(job['bridge_alias'],['kernels','push','-p',str(folder),'--accelerator','NvidiaTeslaT4','-t',str(job['session_timeout_seconds'])],job['job']+'-push')
 version=re.search(r'Kernel version (\d+) successfully pushed',output)
 url=re.search(r'https://www\.kaggle\.com/code/([^\s]+)',output)
 if version and url:
  actual=url.group(1).rstrip('/')
  if actual!=job['kernel']:
   job.update(status='NEEDS_REVIEW',unexpected_returned_kernel=actual);save(state)
   raise RuntimeError('Returned owner/slug differs from prepared metadata; no further mutation on this job')
  job.update(status='SUBMITTED',version=int(version.group(1)),submitted_at=now())
 elif code and re.search(r'quota|GPU.*(?:limit|available)|too many.*(?:session|GPU)',output,re.I):
  job.update(status='WAITING_QUOTA',last_error='Server refused GPU capacity/quota; no account rotation')
 else:
  job['last_error']='Ambiguous push: inspect remote state before retry; no blind retry'
 save(state)

def download(state,job):
 target=ARTIFACTS/job['job'];ref=f"{job['kernel']}/{job['version']}"
 pattern=r'(TRAINING_COMPLETE\.json$|convergence_protocol\.json$|data/noise/bank\.json$|runs/.*/(?:history|config|environment|training_summary|selection)\.json$|runs/.*/validation/epoch_\d+\.parquet$)'
 code,output=call(job['bridge_alias'],['kernels','output',ref,'-p',str(target),'--file-pattern',pattern,'--quiet','--force','--page-size','200'],job['job']+'-download')
 if code:raise RuntimeError('Output download failed: '+job['job'])
 from verify_convergence import verify
 try:
  result=verify(target/'full20_project',expected_protocol=state['protocol'])
  assert result['condition']==job['condition'] and result['seed']==job['seed'] and result['epochs_completed']==20
 except Exception:
  job.update(status='VERIFICATION_FAILED',last_error=traceback.format_exc());save(state);raise
 job.update(status='COMPLETE',verified=True,verification=result,completed_at=now());save(state)

def step(state):
 # Read-only checks first. A COMPLETED remote job frees its compute slot before download.
 blocked_aliases=set()
 for job in state['jobs']:
  if job['status'] in {'SUBMITTED','RUNNING','QUEUED'}:
   try:
    status=remote_status(job['bridge_alias'],f"{job['kernel']}/{job['version']}",job['job']+'-status')
    job.update(status='REMOTE_COMPLETE' if status=='COMPLETE' else status,last_status_check=now());save(state)
   except DialogCanceled:raise
   except Exception as exc:
    job['last_status_error']=str(exc);blocked_aliases.add(job['bridge_alias']);save(state)
 # Do not spend a free slot while waiting for large output downloads.
 for alias,kernels in state['external_kernels'].items():
  waiting=[j for j in state['jobs'] if j['bridge_alias']==alias and j['status']=='PREPARED']
  if not waiting or alias in blocked_aliases:continue
  external_count=0;observations={}
  try:
   for kernel in kernels:
    status=remote_status(alias,kernel,'external-'+alias+'-'+kernel.split('/')[-1])
    observations[kernel]=status
    if status in {'RUNNING','QUEUED'}:external_count+=1
  except DialogCanceled:raise
  except Exception as exc:
   state.setdefault('external_status_errors',{})[alias]=str(exc);save(state);continue
  state.setdefault('external_observations',{})[alias]={'checked_at':now(),'kernels':observations}
  active_count=sum(j['bridge_alias']==alias and j['status'] in ACTIVE for j in state['jobs'])
  # A quota/ambiguous failure blocks new pushes on this alias, without rotation.
  if any(j['bridge_alias']==alias and j['status'] in FAILED for j in state['jobs']):continue
  free=max(0,state['max_sessions_per_account']-external_count-active_count)
  for job in waiting[:free]:
   submit(state,job)
   if job['status']!='SUBMITTED':break
 for job in state['jobs']:
  if job['status']=='REMOTE_COMPLETE':
   try:download(state,job)
   except DialogCanceled:raise
   except Exception as exc:
    job['last_download_error']=str(exc);save(state)
 complete=sum(j['status']=='COMPLETE' and j.get('verified') for j in state['jobs'])
 failures=[j['job'] for j in state['jobs'] if j['status'] in FAILED]
 state.update(completed_jobs=complete,failed_or_blocked_jobs=failures,campaign_status='TRAINING_COMPLETE' if complete==9 else 'NEEDS_ATTENTION' if failures else 'RUNNING',last_cycle=now())
 save(state)
 print('FULL20_STATE',json.dumps({j['job']:j['status'] for j in state['jobs']}),flush=True)
 return complete==9

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--once',action='store_true');p.add_argument('--interval',type=int,default=300);p.add_argument('--max-hours',type=float,default=18)
 args=p.parse_args();ARTIFACTS.mkdir(parents=True,exist_ok=True)
 with (ARTIFACTS/'worker.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  start=time.monotonic();state=json.loads(STATE.read_text())
  state['worker']={'pid':os.getpid(),'started_at':now(),'mode':'once' if args.once else 'monitor','status':'ACTIVE','interval_seconds':args.interval};save(state)
  try:
   while True:
    done=step(state)
    if done or args.once:break
    # Failures are reported; healthy submitted runs can still be collected.
    if all(j['status'] in {'COMPLETE',*FAILED} for j in state['jobs']):break
    if time.monotonic()-start>args.max_hours*3600:
     state['campaign_status']='MONITOR_DEADLINE_REACHED';break
    time.sleep(args.interval)
  except DialogCanceled:
   state['campaign_status']='LOCAL_UNLOCK_CANCELED';state['worker']['status']='STOPPED_AFTER_CANCELLATION';save(state);raise
  except BaseException:
   state['worker']['last_error']=traceback.format_exc();raise
  finally:
   state['worker'].update(status='STOPPED',ended_at=now());save(state)

if __name__=='__main__':main()
