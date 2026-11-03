"""Summarize locally captured Kaggle live logs without disclosing credentials."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
log_path = ROOT / 'artifacts/kaggle-live.log'
lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
steps, validation, test_metrics = {}, {}, None
for line in lines:
    for prefix, target, key in [('TRAIN_STEP ', steps, 'step'), ('VALIDATION ', validation, 'epoch')]:
        if line.startswith(prefix):
            record = json.loads(line[len(prefix):])
            target[record[key]] = record
    if line.startswith('FINAL_TEST_METRICS '):
        test_metrics = json.loads(line[len('FINAL_TEST_METRICS '):])
metadata = ROOT / 'reports/KAGGLE_RUN.json'
state = json.loads(metadata.read_text()) if metadata.exists() else {}
state['observed_at_utc'] = datetime.now(timezone.utc).isoformat()
state['latest_training_step'] = steps[max(steps)] if steps else None
state['validation_history'] = [validation[e] for e in sorted(validation)]
state['test_metrics'] = test_metrics
state['pipeline_complete_marker'] = any(line.strip() == 'RUN_COMPLETE' for line in lines)
if state['pipeline_complete_marker']:
    state['stage_last_observed'] = 'COMPLETE_MARKER_SEEN'
metadata.write_text(json.dumps(state, ensure_ascii=False, indent=2))
if steps:
    latest = state['latest_training_step']
    print(f"Epoch {latest['epoch']}, step {latest['step']}/3760, loss {latest['train_loss']:.4f}, elapsed {latest['elapsed_seconds']:.0f}s")
for epoch in sorted(validation):
    record = validation[epoch]
    print(f"Validation epoch {epoch}: Macro-F1 {record['validation']['macro_f1']:.4f}, accuracy {record['validation']['accuracy']:.4f}")
if test_metrics:
    print(f"TEST Macro-F1 {test_metrics['macro_f1']:.4f}, accuracy {test_metrics['accuracy']:.4f}")
print('Complete marker:', state['pipeline_complete_marker'])
