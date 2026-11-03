"""Download bounded reports and predictions, keeping large checkpoints on Kaggle."""
import argparse,json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from vihsd.data import dump_json
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('condition');p.add_argument('--version',default='1');args=p.parse_args()
api=KaggleApi();api.authenticate()
ref=f'ndhoang2310/vihsd-real-noise-{args.condition}/{args.version}'
status=api.kernels_status(ref)
if 'COMPLETE' not in str(status.status):raise RuntimeError(f'{ref}: {status.status}, {status.failure_message}')
folder=ROOT/'artifacts/robustness'/args.condition
pattern=r'(reports/|robustness/|data/noise/bank\.json|data/tokenized/token_diagnostics\.json|data/noise_views/.*/view\.parquet|ROBUSTNESS_COMPLETE\.json|runs/.*/(?:history|selection|config|environment|training_summary)\.json)'
files,token=api.kernels_output(ref,str(folder),file_pattern=pattern,force=True,quiet=True,page_size=200)
assert not token
print('Downloaded',len(files),'files to',folder,flush=True)
from verify_robustness import verify_job
verified,table=verify_job(folder/'robustness_project',args.condition)
table.to_csv(ROOT/f'reports/robustness_results_{args.condition}.csv',index=False)
state=json.load(open(ROOT/'reports/ROBUSTNESS_RUNS.json'));state['jobs'][args.condition].update(status='COMPLETE',verified=True,verification=verified)
dump_json(ROOT/'reports/ROBUSTNESS_RUNS.json',state)
print(json.dumps(verified,indent=2),flush=True)
