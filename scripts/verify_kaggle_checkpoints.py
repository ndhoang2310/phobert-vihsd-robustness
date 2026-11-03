"""Verify downloadable checkpoint bytes with bounded GETs; never print signed URLs."""
import argparse,json,struct
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
from vihsd.data import dump_json
p=argparse.ArgumentParser();p.add_argument('conditions',nargs='+');args=p.parse_args()
api=KaggleApi();api.authenticate();rows=[]
for condition in args.conditions:
 ref=f'ndhoang2310/vihsd-real-noise-{condition}/1';owner,slug,version=api.parse_kernel_string(ref)
 with api.build_kaggle_client() as k:
  request=ApiListKernelSessionOutputRequest();request.user_name=owner;request.kernel_slug=slug;request.version_label=api._kernel_version_label(version);request.page_size=200
  response=k.kernels.kernels_api_client.list_kernel_session_output(request)
  items=list(response.files)
  while response.next_page_token:
   request.page_token=response.next_page_token
   response=k.kernels.kernels_api_client.list_kernel_session_output(request);items.extend(response.files)
  found=[]
  for item in items:
   if not (item.file_name.endswith('/best/model.safetensors') or item.file_name.endswith('/last.pt')):continue
   with requests.get(item.url,headers={'Range':'bytes=0-1023'},timeout=60,stream=True) as r:
    assert r.status_code==206
    prefix=next(r.iter_content(chunk_size=1024));assert len(prefix)==1024
    if item.file_name.endswith('.pt'):assert prefix[:2]==b'PK'
    else:assert 0<struct.unpack('<Q',prefix[:8])[0]<1000000
    found.append({'condition':condition,'path':item.file_name,'status':r.status_code,'content_range':r.headers.get('Content-Range'),'bytes_verified':len(prefix)})
  assert len(found)==2,(condition,len(found));rows+=found
out=Path(__file__).resolve().parents[1]/'reports/ROBUSTNESS_CHECKPOINT_VERIFICATION.json'
if out.exists():
 old=json.loads(out.read_text());rows=[r for r in old if r['condition'] not in args.conditions]+rows
dump_json(out,rows);print(json.dumps(rows,indent=2))
