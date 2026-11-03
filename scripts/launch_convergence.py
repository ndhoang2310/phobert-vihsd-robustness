"""Campaign operations exclusively through the existing local credential bridge.

Every remote command opens the bridge's password/account dialog. This helper never
reads a vault, key, credential file, environment credential, or Kaggle SDK session.
"""
import argparse,datetime,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'reports/CONVERGENCE_RUNS.json'
BRIDGE=Path('/home/hoang/.local/bin/kaggle-account')


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(state):
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(STATE)


def bridge(arguments, alias):
    command=[str(BRIDGE),'run']+([alias] if alias else [])+['--']+arguments
    # The existing bridge owns its GUI and 180s local timeout. No secret is passed here.
    return subprocess.run(command,capture_output=True,text=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['push','status','download'])
    parser.add_argument('--job',required=True)
    parser.add_argument('--alias',help='Only use an alias supplied by the user or obtained from kaggle-account list; omit for GUI selection')
    args=parser.parse_args();state=json.loads(STATE.read_text());job=next(j for j in state['jobs'] if j['job']==args.job)
    if state['username']=='ACCOUNT-PENDING':raise RuntimeError('Build metadata with the verified actual Kaggle owner first')
    folder=Path(job['folder']);metadata=json.loads((folder/'kernel-metadata.json').read_text())
    assert metadata['id']==job['kernel'] and metadata['is_private'] and 'id_no' not in metadata
    if args.operation=='push':
        if job['status'] not in ['PREPARED','WAITING_QUOTA']:raise RuntimeError('Already submitted or uncertain; check status before retrying a push')
        owner=job['kernel'].split('/')[0]
        active=[j for j in state['jobs'] if j['status'] in ['SUBMITTED','RUNNING','QUEUED','SUBMISSION_UNKNOWN'] and j['kernel'].split('/')[0]==owner]
        if len(active)>=2:raise RuntimeError('Two campaign GPU sessions already submitted on this owner; check their status first')
        seed=next(j['seed'] for j in state['jobs'] if j['status']!='COMPLETE')
        if job['seed']!=seed:raise RuntimeError('Complete the current matched seed block before starting another seed')
        seconds={'clean':4800,'augmentation':6000,'fgm':8400}[job['condition']]
        arguments=['kernels','push','-p',str(folder),'--accelerator','NvidiaTeslaT4','-t',str(seconds)]
    else:
        if not job.get('version'):raise RuntimeError('No version recorded; inspect remote state before setting an uncertain version')
        ref=f"{job['kernel']}/{job['version']}"
        if args.operation=='status':arguments=['kernels','status',ref]
        else:
            if job['status']!='REMOTE_COMPLETE':raise RuntimeError('Check for remote COMPLETE before downloading/verifying')
            target=ROOT/f"artifacts/convergence/{job['job']}"
            arguments=['kernels','output',ref,'-p',str(target),'--file-pattern',r'(TRAINING_COMPLETE\.json$|convergence_protocol\.json$|data/noise/bank\.json$|runs/.*/(?:history|config|environment|training_summary|selection)\.json$|runs/.*/validation/epoch_\d+\.parquet$)','--quiet','--force','--page-size','200']
    print('The existing bridge will open a local password dialog; target owner: '+job['kernel'].split('/')[0]+'.',flush=True)
    alias=args.alias or job.get('bridge_alias') or state.get('owner_bridge_aliases',{}).get(job['kernel'].split('/')[0])
    result=bridge(arguments,alias);output=result.stdout+'\n'+result.stderr
    logs=ROOT/'artifacts/convergence';logs.mkdir(parents=True,exist_ok=True)
    (logs/f'{args.job}-{args.operation}.log').write_text(output)
    print(output,flush=True)
    if args.operation=='push':
        version=re.search(r'Kernel version (\d+) successfully pushed',output)
        if version:
            actual=re.search(r'https://www\.kaggle\.com/code/([^\s]+)',output)
            if not actual:raise RuntimeError('Push succeeded but actual owner/slug unavailable; inspect remote state before retry')
            job['kernel']=actual.group(1).rstrip('/');metadata['id']=job['kernel']
            (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
            if alias:job['bridge_alias']=alias
            job.update(status='SUBMITTED',version=int(version.group(1)),submitted_at=now(),session_timeout_seconds=seconds);state['campaign_status']='RUNNING'
        elif 'quota' in output.lower():job['status']='WAITING_QUOTA'
        elif re.search(r'cancel|hủy|huỷ',output,re.I):
            state['last_dialog_canceled']=now();write(state);return result.returncode or 1
        else:
            # A local timeout does not establish whether the remote push succeeded.
            job.update(status='SUBMISSION_UNKNOWN',last_attempt=now());state['campaign_status']='NEEDS_REMOTE_STATUS_CHECK'
    elif args.operation=='status' and result.returncode==0:
        match=re.search(r'(?:KernelWorkerStatus\.)?(COMPLETE|RUNNING|QUEUED|ERROR|CANCEL_ACKNOWLEDGED)',output)
        if match:
            name=match.group(1);job['status']='REMOTE_COMPLETE' if name=='COMPLETE' else name;job['last_status_check']=now()
    elif args.operation=='download' and result.returncode==0:
        from verify_convergence import verify
        job['verification']=verify(target/'convergence_project');job.update(status='COMPLETE',verified=True,completed_at=now())
    if all(j['status']=='COMPLETE' and j.get('verified') for j in state['jobs']):state.update(campaign_status='TRAINING_COMPLETE',completed_at=now(),test_evaluated=False)
    write(state);return result.returncode


if __name__=='__main__':raise SystemExit(main())
