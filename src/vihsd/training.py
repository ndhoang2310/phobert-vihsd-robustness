"""Single-GPU PyTorch baseline; preprocessing and test never enter model selection."""
from __future__ import annotations
import copy
import gc
import json
import math
import os
import platform
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from .data import LABELS, dump_json, sha256_bytes


def metrics(y_true, y_pred):
    return {'accuracy': float(accuracy_score(y_true, y_pred)),
            'macro_f1': float(f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro', zero_division=0)),
            'per_class': classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=LABELS, zero_division=0, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()}


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_config(config):
    from huggingface_hub import model_info
    resolved = copy.deepcopy(config)
    resolved['model_revision'] = model_info(config['model_name'], revision=config.get('model_revision', 'main')).sha
    return resolved


def tokenize_frames(frames, tokenizer, max_length, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    encoded, diagnostics = {}, {}
    for split, frame in frames.items():
        ids, masks, lengths, unks, contents = [], [], [], [], []
        texts = frame.text_segmented.tolist()
        for start in range(0, len(texts), 256):
            batch = tokenizer(texts[start:start+256], add_special_tokens=True, truncation=False, padding=False,
                              return_attention_mask=False, verbose=False)['input_ids']
            for tokens in batch:
                lengths.append(len(tokens))
                content = tokens[1:-1]
                contents.append(len(content))
                unks.append(content.count(tokenizer.unk_token_id))
                # Use the tokenizer's own truncation path to preserve EOS.
            truncated = tokenizer(texts[start:start+256], truncation=True, padding=False, max_length=max_length)
            ids.extend(truncated['input_ids'])
            masks.extend(truncated['attention_mask'])
        encoded[split] = [{'input_ids': i, 'attention_mask': m, 'labels': int(y)}
                          for i, m, y in zip(ids, masks, frame.label)]
        diagnostics[split] = {'lengths': lengths, 'unk_counts': unks, 'content_counts': contents}
        assert len(encoded[split]) == len(frame)
        assert all(2 <= len(x['input_ids']) <= max_length for x in encoded[split])
        # Compact non-executable cache for replay and inspection.
        dump_json(output / f'{split}.json', encoded[split])
    dump_json(output / 'token_diagnostics.json', diagnostics)
    return encoded, diagnostics


def make_model(config):
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(
        config['model_name'], revision=config['model_revision'], num_labels=3,
        id2label=dict(enumerate(LABELS)), label2id={v: k for k, v in enumerate(LABELS)})


def make_loader(records, tokenizer, batch_size, shuffle=False, seed=42):
    import torch
    from transformers import DataCollatorWithPadding
    return torch.utils.data.DataLoader(records, batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
        collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, return_tensors='pt'),
        pin_memory=torch.cuda.is_available())


def predict(model, records, tokenizer, batch_size, device):
    import torch
    was_training = model.training
    model.eval()
    logits, labels, losses = [], [], []
    with torch.inference_mode():
        for batch in make_loader(records, tokenizer, batch_size):
            batch = {k: v.to(device) for k, v in batch.items()}
            result = model(**batch)
            losses.append((float(result.loss), len(batch['labels'])))
            logits.append(result.logits.float().cpu())
            labels.extend(batch['labels'].cpu().tolist())
    all_logits = torch.cat(logits)
    probabilities = all_logits.softmax(dim=-1).numpy()
    result = metrics(labels, probabilities.argmax(axis=1))
    result['loss'] = sum(x*n for x, n in losses) / sum(n for _, n in losses)
    if was_training:
        model.train()
    return result, probabilities


def balanced_indices(frame, per_class, seed=42):
    return np.concatenate([frame.index[frame.label == label].to_series().sample(min(per_class, int((frame.label == label).sum())), random_state=seed).to_numpy() for label in range(3)]).astype(int)


def smoke_and_overfit(frames, encoded, tokenizer, config, output):
    """Real pretrained weights: finite backward, roundtrip, then learn 48 train examples."""
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_all(config['seed'])
    idx = balanced_indices(frames['train'], 16)
    records = [encoded['train'][i] for i in idx]
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loader = make_loader(records, tokenizer, 8, shuffle=True)
    model.train()
    losses = []
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        assert torch.isfinite(loss), 'Nonfinite smoke loss'
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        if step == 2:
            break
    model.eval()
    reference = next(iter(make_loader(records[:4], tokenizer, 4)))
    reference = {k: v.to(device) for k, v in reference.items()}
    with torch.no_grad():
        expected = model(**reference).logits.cpu()
    model.save_pretrained(output / 'roundtrip')
    # Optimizer state and RNG roundtrip: trusted checkpoint generated by this run.
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}, output / 'smoke_resume.pt')
    saved = torch.load(output / 'smoke_resume.pt', map_location=device, weights_only=False)
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng'].cpu())
    np.random.set_state(saved['numpy_rng'])
    random.setstate(saved['python_rng'])
    del saved
    reloaded = type(model).from_pretrained(output / 'roundtrip').to(device).eval()
    with torch.no_grad():
        actual = reloaded(**reference).logits.cpu()
    max_diff = float((expected - actual).abs().max())
    assert max_diff < 1e-5, f'Roundtrip mismatch: {max_diff}'
    del reloaded
    gc.collect()
    # Start the overfit check with a fresh optimizer, in FP32; this model is discarded.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    steps = 0
    result = None
    while steps < 300:
        model.train()
        for batch in make_loader(records, tokenizer, 8, shuffle=True, seed=config['seed'] + steps):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite overfit loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
        if steps % 30 == 0:
            result, _ = predict(model, records, tokenizer, 8, device)
            print(f'OVERFIT step={steps} accuracy={result["accuracy"]:.4f}', flush=True)
            if result['accuracy'] >= .95:
                break
    if result is None:
        result, _ = predict(model, records, tokenizer, 8, device)
    summary = {'smoke_losses': losses, 'roundtrip_max_logit_difference': max_diff,
               'optimizer_state_roundtrip': True, 'overfit_steps': steps, 'overfit_metrics': result,
               'passed': result['accuracy'] >= .95}
    dump_json(output / 'checks.json', summary)
    del model, optimizer
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    # Remove diagnostic model files; only the final baseline needs full weights.
    import shutil
    shutil.rmtree(output / 'roundtrip')
    (output / 'smoke_resume.pt').unlink()
    if not summary['passed']:
        raise RuntimeError('Overfit check failed; full training blocked. See checks.json')
    return summary


def environment():
    import importlib.metadata
    import torch
    versions = {}
    for package in ['torch', 'transformers', 'accelerate', 'pandas', 'numpy', 'py-vncorenlp', 'tokenizers', 'scikit-learn']:
        versions[package] = importlib.metadata.version(package)
    return {'python': platform.python_version(), 'packages': versions,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'visible_gpu_count': torch.cuda.device_count()}


def train(frames, encoded, tokenizer, config, run_dir, resume=None, epoch_records_factory=None, _test_device=None):
    import torch
    from transformers import get_linear_schedule_with_warmup
    from .stopping import EarlyStopping
    stopping = EarlyStopping(config.get('early_stopping_patience'), config.get('early_stopping_min_delta', 0.0))
    if config['epochs'] < 1:raise ValueError('epochs must be positive')
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if _test_device is None and not torch.cuda.is_available():
        raise RuntimeError('Full training requires CUDA; use smoke tests for local CPU checks')
    device = torch.device(_test_device or 'cuda:0')
    seed_all(config['seed'])
    model = make_model(config).to(device)
    batch_size = config['train_batch_size']
    accum = config['gradient_accumulation_steps']
    # Probe longest training records, preserve the intended effective batch if OOM.
    longest = sorted(encoded['train'], key=lambda r: len(r['input_ids']), reverse=True)
    while True:
        try:
            batch = next(iter(make_loader(longest[:batch_size], tokenizer, batch_size)))
            loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
            loss.backward()
            model.zero_grad(set_to_none=True)
            del loss, batch
            break
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            if batch_size <= 2:
                if model.is_gradient_checkpointing:
                    raise
                model.gradient_checkpointing_enable()
            else:
                batch_size //= 2
                accum *= 2
    config = {**config, 'actual_train_batch_size': batch_size, 'actual_gradient_accumulation_steps': accum,
              'effective_batch_size': batch_size * accum, 'gradient_checkpointing': model.is_gradient_checkpointing}
    config['precision'] = ('bf16' if torch.cuda.is_bf16_supported(including_emulation=False) else 'fp16') if device.type == 'cuda' else 'fp32'
    n_batches = math.ceil(len(encoded['train']) / batch_size)
    steps_per_epoch = math.ceil(n_batches / accum)
    total_steps = steps_per_epoch * config['epochs']
    no_decay = ('bias', 'LayerNorm.weight', 'layer_norm.weight')
    parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)], 'weight_decay': config['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], betas=(.9, .999), eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_steps * config['warmup_ratio']), total_steps)
    use_bf16 = config['precision'] == 'bf16'
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda' and not use_bf16)
    start_epoch, skip_batches, global_step, best_score = 0, 0, 0, -1.0
    best_epoch = None
    history = []
    partial_epoch_loss, partial_epoch_n = 0.0, 0
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['config'] != config:
            raise ValueError('Resume config differs from checkpoint')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        start_epoch, skip_batches = saved['next_epoch'], saved['next_batch']
        global_step, best_score, best_epoch = saved['global_step'], saved['best_score'], saved['best_epoch']
        history = saved['history']
        if 'early_stopping' in saved:stopping.restore(saved['early_stopping'])
        else:
            for record in history:stopping.observe(record['validation']['macro_f1'])
        partial_epoch_loss=saved.get('partial_epoch_loss',0.0)
        partial_epoch_n=saved.get('partial_epoch_n',0)
        if stopping.epochs_observed != len(history):raise ValueError('Early-stopping state/history mismatch')
        if best_epoch is not None and (not (run_dir/'best/selection.json').exists() or not any((run_dir/'best'/name).exists() for name in ['model.safetensors','pytorch_model.bin'])):
            raise ValueError('Resume requires the previously selected best checkpoint alongside last.pt')
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved
    dump_json(run_dir / 'config.json', config)
    dump_json(run_dir / 'environment.json', environment())
    started = time.time()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()

    def checkpoint(next_epoch, next_batch):
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'scaler': scaler.state_dict(), 'config': config, 'next_epoch': next_epoch, 'next_batch': next_batch,
                 'global_step': global_step, 'best_score': best_score, 'best_epoch': best_epoch, 'history': history,
                 'early_stopping':stopping.state_dict(),
                 'partial_epoch_loss':epoch_loss if next_batch else 0.0,'partial_epoch_n':epoch_n if next_batch else 0,
                 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        tmp = run_dir / 'last.tmp'
        torch.save(state, tmp)
        tmp.replace(run_dir / 'last.pt')

    print(f'TRAINING_STARTED examples={len(encoded["train"])} total_steps={total_steps} gpu={torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU_TEST"} precision={config["precision"]}', flush=True)
    epoch_loss, epoch_n = partial_epoch_loss, partial_epoch_n
    for epoch in range(start_epoch, config['epochs']):
        if stopping.should_stop:break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss, epoch_n = (partial_epoch_loss, partial_epoch_n) if epoch==start_epoch and skip_batches else (0., 0)
        epoch_records = epoch_records_factory(epoch) if epoch_records_factory is not None else encoded['train']
        if len(epoch_records) != len(encoded['train']):
            raise ValueError('Augmentation must preserve the number of training examples')
        loader = make_loader(epoch_records, tokenizer, batch_size, shuffle=True, seed=config['seed'] + epoch)
        for batch_idx, batch in enumerate(loader):
            if epoch == start_epoch and batch_idx < skip_batches:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            group_start = (batch_idx // accum) * accum
            group_size = min(accum, n_batches - group_start)
            with torch.autocast(device.type, dtype=torch.bfloat16 if use_bf16 else torch.float16, enabled=device.type=='cuda'):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                checkpoint(epoch, group_start)
                raise RuntimeError(f'Nonfinite loss at epoch {epoch}, batch {batch_idx}')
            from .adversarial import backward_with_fgm
            loss_value, attack_skipped = backward_with_fgm(model, batch, loss, scaler, group_size, config.get('fgm_epsilon', 0.0), device_type=device.type, dtype=torch.bfloat16 if use_bf16 else torch.float16)
            n = len(batch['labels'])
            epoch_loss += loss_value * n
            epoch_n += n
            if (batch_idx + 1) % accum == 0 or batch_idx + 1 == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step == 1 or global_step % 50 == 0:
                    record = {'epoch': epoch + 1, 'step': global_step, 'train_loss': epoch_loss / max(epoch_n, 1),
                              'learning_rate': scheduler.get_last_lr()[0], 'elapsed_seconds': time.time() - started}
                    with (run_dir / 'steps.jsonl').open('a') as f:
                        f.write(json.dumps(record) + '\n')
                    print('TRAIN_STEP ' + json.dumps(record), flush=True)
                if global_step % config['save_steps'] == 0:
                    checkpoint(epoch, batch_idx + 1)
        validation, validation_prob = predict(model, encoded['validation'], tokenizer, config['eval_batch_size'], device)
        if config.get('save_validation_predictions',False):
            folder=run_dir/'validation';folder.mkdir(parents=True,exist_ok=True)
            prediction=frames['validation'][['sample_id','label']].copy().reset_index(drop=True)
            prediction['prediction']=validation_prob.argmax(1)
            for index,label in enumerate(LABELS):prediction[f'p_{label.lower()}']=validation_prob[:,index]
            prediction.to_parquet(folder/f'epoch_{epoch+1}.parquet',index=False)
        record = {'epoch': epoch + 1, 'step': global_step, 'train_loss': epoch_loss / max(epoch_n, 1),
                  'validation': validation, 'elapsed_seconds': time.time() - started,
                  'peak_gpu_memory_gb': torch.cuda.max_memory_allocated() / 1024**3 if device.type=='cuda' else 0.0}
        stopping.observe(validation['macro_f1'])
        record['early_stopping']=stopping.state_dict()
        history.append(record)
        dump_json(run_dir / 'history.json', history)
        print('VALIDATION ' + json.dumps(record), flush=True)
        if validation['macro_f1'] > best_score:
            best_score, best_epoch = validation['macro_f1'], epoch + 1
            model.save_pretrained(run_dir / 'best')
            tokenizer.save_pretrained(run_dir / 'best')
            dump_json(run_dir / 'best/selection.json', {'split': 'validation', 'metric': 'macro_f1', 'score': best_score, 'epoch': best_epoch})
        checkpoint(epoch + 1, 0)
        if stopping.should_stop:
            print('EARLY_STOPPING '+json.dumps({'epoch':epoch+1,'best_epoch':best_epoch,**stopping.state_dict()}),flush=True)
            break
    dump_json(run_dir / 'training_summary.json', {'best_validation_macro_f1': best_score, 'best_epoch': best_epoch,
                                               'global_step': global_step, 'elapsed_seconds': time.time() - started,
                                               'epochs_completed':len(history),'max_epochs':config['epochs'],
                                               'stop_reason':'early_stopping' if stopping.should_stop else 'max_epochs',
                                               'early_stopping':stopping.state_dict()})
    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return run_dir / 'best'


def evaluate_checkpoint(checkpoint, frames, encoded, tokenizer, output, batch_size=16):
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    from transformers import AutoModelForSequenceClassification
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()
    result, probabilities = predict(model, encoded['test'], tokenizer, batch_size, device)
    predictions = frames['test'][['sample_id', 'label']].copy().reset_index(drop=True)
    predictions['prediction'] = probabilities.argmax(axis=1)
    for i, label in enumerate(LABELS):
        predictions[f'p_{label.lower()}'] = probabilities[:, i]
    assert len(predictions) == len(frames['test']) and predictions.sample_id.is_unique
    assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1), 1, atol=1e-5)
    predictions.to_parquet(output / 'predictions.parquet', index=False)
    dump_json(output / 'metrics.json', result)
    cm = np.asarray(result['confusion_matrix'])
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(output / 'confusion_matrix.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, mat, fmt, title in [(axes[0], cm, 'd', 'Số mẫu'), (axes[1], cm / np.maximum(cm.sum(1, keepdims=True), 1), '.2f', 'Chuẩn hóa theo nhãn thật')]:
        sns.heatmap(mat, annot=True, fmt=fmt, xticklabels=LABELS, yticklabels=LABELS, ax=ax, cmap='Blues')
        ax.set(xlabel='Dự đoán', ylabel='Nhãn thật', title=title)
    fig.tight_layout(); fig.savefig(output / 'confusion_matrix.png', dpi=160); plt.close(fig)
    errors = predictions[predictions.label != predictions.prediction]
    selected = []
    for true in range(3):
        for predicted in range(3):
            if true != predicted:
                candidates = errors[(errors.label == true) & (errors.prediction == predicted)]
                selected.append(candidates.sample(min(10, len(candidates)), random_state=42))
    pd.concat(selected).merge(frames['test'][['sample_id', 'text_raw', 'text_segmented']], on='sample_id').to_csv(output / 'error_review.csv', index=False)
    history_path = output.parent / 'history.json'
    if history_path.exists():
        history = json.loads(history_path.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot([r['epoch'] for r in history], [r['train_loss'] for r in history], label='train')
        axes[0].plot([r['epoch'] for r in history], [r['validation']['loss'] for r in history], label='validation')
        axes[0].set(xlabel='Epoch', ylabel='Loss'); axes[0].legend()
        axes[1].plot([r['epoch'] for r in history], [r['validation']['macro_f1'] for r in history], marker='o')
        axes[1].set(xlabel='Epoch', ylabel='Validation Macro-F1')
        fig.tight_layout(); fig.savefig(output / 'training_curves.png', dpi=160); plt.close(fig)
    print('FINAL_TEST_METRICS ' + json.dumps(result), flush=True)
    return result
