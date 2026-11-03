"""Independently verify downloaded Kaggle reports/predictions against original labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def verify(root):
    from vihsd.data import load_raw, dump_json
    root = Path(root)
    run = root / 'runs/phobert-clean-seed42'
    history = json.loads((run / 'history.json').read_text())
    assert [r['epoch'] for r in history] == [1, 2, 3, 4, 5]
    assert history[-1]['step'] == 3760
    selection = json.loads((run / 'best/selection.json').read_text())
    scores = [r['validation']['macro_f1'] for r in history]
    assert selection['epoch'] == int(np.argmax(scores)) + 1
    assert abs(selection['score'] - max(scores)) < 1e-12
    checks = json.loads((root / 'reports/checks/checks.json').read_text())
    assert checks['passed'] and checks['overfit_metrics']['accuracy'] >= .95
    assert np.isfinite(checks['smoke_losses']).all()
    p = pd.read_parquet(run / 'evaluation/predictions.parquet')
    official = load_raw(Path(__file__).resolve().parents[1])['test']
    assert len(p) == 6680 and p.sample_id.is_unique
    compared = p.merge(official[['sample_id', 'label']], on='sample_id', suffixes=('_predfile', '_source'), validate='one_to_one')
    assert len(compared) == 6680 and (compared.label_predfile == compared.label_source).all()
    probabilities = p[['p_clean', 'p_offensive', 'p_hate']].to_numpy()
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    assert np.array_equal(p.prediction, probabilities.argmax(axis=1))
    actual = json.loads((run / 'evaluation/metrics.json').read_text())
    independent = {'accuracy': accuracy_score(p.label, p.prediction),
                   'macro_f1': f1_score(p.label, p.prediction, labels=[0, 1, 2], average='macro', zero_division=0),
                   'confusion_matrix': confusion_matrix(p.label, p.prediction, labels=[0, 1, 2]).tolist()}
    assert abs(actual['accuracy'] - independent['accuracy']) < 1e-12
    assert abs(actual['macro_f1'] - independent['macro_f1']) < 1e-12
    assert actual['confusion_matrix'] == independent['confusion_matrix']
    assert (root / 'RUN_COMPLETE.json').exists()
    result = {'passed': True, 'epochs': 5, 'optimizer_steps': 3760, 'test_examples': 6680,
              'best_epoch': selection['epoch'], 'independently_recomputed_test_metrics': independent,
              'checks': ['five epochs', 'best selected on validation', 'smoke and overfit', 'source labels and IDs',
                         'probability validity', 'independent metrics', 'complete marker']}
    dump_json(Path(__file__).resolve().parents[1] / 'reports/OUTPUT_VERIFICATION.json', result)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('artifact_root'); verify(parser.parse_args().artifact_root)
