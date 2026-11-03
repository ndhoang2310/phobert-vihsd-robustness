import argparse
import json
from pathlib import Path
from .data import load_raw, audit, prepare
from .eda import create_eda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['doctor', 'download', 'audit', 'eda', 'prepare', 'train', 'evaluate'])
    parser.add_argument('--root', default='.')
    parser.add_argument('--config', default='configs/baseline.json')
    parser.add_argument('--run-dir')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.command == 'doctor':
        from .pipeline import ensure_java
        from .training import environment
        print(ensure_java()); print(json.dumps(environment(), indent=2)); return
    if args.command == 'download':
        frames = load_raw(root); print({s: len(f) for s, f in frames.items()}); return
    if args.command in ['audit', 'eda']:
        frames = load_raw(root)
        summary = audit(frames, root / 'reports/audit')
        if args.command == 'eda':
            create_eda(frames, root / 'reports/eda', summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2)); return
    config = json.loads(Path(args.config).read_text())
    from .pipeline import prepare_all, run_training
    if args.command == 'prepare':
        prepare_all(root, config)
    elif args.command == 'train':
        run_training(root, config)
    else:
        if not args.run_dir:
            parser.error('--run-dir is required for evaluate')
        from transformers import AutoTokenizer
        from .training import tokenize_frames, evaluate_checkpoint
        run_dir = Path(args.run_dir)
        saved_config = json.loads((run_dir / 'config.json').read_text())
        tokenizer = AutoTokenizer.from_pretrained(run_dir / 'best', use_fast=False)
        from .pipeline import ensure_java
        ensure_java()
        frames = prepare(root)
        encoded, _ = tokenize_frames(frames, tokenizer, saved_config['max_length'], root / 'data/tokenized')
        evaluate_checkpoint(run_dir / 'best', frames, encoded, tokenizer, run_dir / 'evaluation', saved_config['eval_batch_size'])

if __name__ == '__main__':
    main()
