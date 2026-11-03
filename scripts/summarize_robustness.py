"""Capture progress from live logs; no parameters are selected from test results."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
state=json.loads((ROOT/'reports/ROBUSTNESS_RUNS.json').read_text())
for condition in state['jobs']:
    path=ROOT/f'artifacts/robustness/{condition}-live.log'
    if not path.exists():continue
    steps={};validation={};profile_metrics={};complete=False;by_model={};suite=-1
    model_names=[condition,'baseline']+(['fgm-eps1-reference'] if condition=='augmentation-v2' else [])
    for line in path.read_text(errors='replace').splitlines():
        for prefix,target,key in [('TRAIN_STEP ',steps,'step'),('VALIDATION ',validation,'epoch'),('ROBUSTNESS_METRICS ',profile_metrics,'profile')]:
            if line.startswith(prefix):
                data=json.loads(line[len(prefix):]);target[data[key]]=data
                if prefix=='ROBUSTNESS_METRICS ':
                    if data['profile']=='original':suite+=1
                    model=model_names[suite] if suite<len(model_names) else f'extra-{suite}'
                    by_model.setdefault(model,{})[data['profile']]=data
        if line.strip()==f'ROBUSTNESS_COMPLETE {condition}':complete=True
    job=state['jobs'][condition]
    if steps:
        latest=steps[max(steps)];job['latest_step']=latest
        print(f"{condition}: epoch {latest['epoch']}, step {latest['step']}/3760, loss {latest['train_loss']:.4f}")
    job['validation_history']=[validation[k] for k in sorted(validation)]
    job['evaluation_profiles_seen']=profile_metrics
    job['evaluation_by_model']=by_model
    job['complete_marker']=complete
    for epoch in sorted(validation):print(f"  validation epoch {epoch}: Macro-F1 {validation[epoch]['validation']['macro_f1']:.4f}")
    print('  evaluated model/profile events:',sum(len(v) for v in by_model.values()),'by model:',{k:len(v) for k,v in by_model.items()},'complete:',complete)
(ROOT/'reports/ROBUSTNESS_RUNS.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
