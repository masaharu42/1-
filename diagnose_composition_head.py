"""Diagnose an existing supervised VAE head using fixed training-alloy features."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from data import ELEMENTS, load_split


def ridge_predict(x_train, y_train, x_target, alpha):
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (x_train - mean) / scale
    target = (x_target - mean) / scale
    intercept = y_train.mean(axis=0)
    coef = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y_train - intercept))
    return intercept + target @ coef


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', required=True)
    p.add_argument('--head-predictions', required=True)
    p.add_argument('--run-config', required=True)
    p.add_argument('--split', default='data/split.json')
    p.add_argument('--output', required=True)
    p.add_argument('--alphas', nargs='+', type=float, default=[0.1, 10.0])
    args = p.parse_args()
    if any(not np.isfinite(a) or a <= 0 for a in args.alphas) or len(set(args.alphas)) != len(args.alphas):
        raise ValueError('Ridge alphas must be distinct positive finite values')
    split = load_split(args.split)
    payload = json.loads(Path(args.features).read_text(encoding='utf-8'))
    cfg = json.loads(Path(args.run_config).read_text(encoding='utf-8'))
    eval_cfg = json.loads((Path(args.head_predictions).parent / 'evaluation_config.json').read_text(encoding='utf-8'))
    if payload['split'] != split or tuple(payload['composition_elements']) != ELEMENTS:
        raise ValueError('Feature split or element order mismatch')
    for partition in ('train', 'val', 'test'):
        if sorted(cfg[partition + '_ids']) != sorted(split[partition]):
            raise ValueError('Training config split mismatch')
    if cfg['manifest_sha256'] != payload['manifest_sha256']:
        raise ValueError('Training config and feature manifest mismatch')
    if eval_cfg['partition'] != 'train' or eval_cfg['windows'] != payload['windows']:
        raise ValueError('Head evaluation must use training alloys and same window count')
    if eval_cfg['source_vae_sha256'] != payload['vae_sha256']:
        raise ValueError('Head and features must use the identical VAE checkpoint')
    records = [r for r in payload['records'] if r['partition'] == 'train']
    ids = [int(r['alloy_id']) for r in records]
    if len(set(ids)) != len(ids) or set(ids) != set(split['train']) or len(ids) < 3:
        raise ValueError('Training feature IDs are missing or duplicated')
    x = np.asarray([r['latent'] for r in records], dtype=float)
    y = np.asarray([r['composition'] for r in records], dtype=float)
    if x.ndim != 2 or x.shape[1] != 16 or y.shape != (len(ids), 9) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid 16-dimensional features or composition')
    with Path(args.head_predictions).open(encoding='utf-8-sig', newline='') as f:
        raw_head = list(csv.DictReader(f))
    head = {(int(r['alloy_id']), r['element']): r for r in raw_head}
    if len(head) != len(raw_head) or set(head) != {(i, e) for i in ids for e in ELEMENTS}:
        raise ValueError('Head predictions missing, duplicated, or contain unexpected IDs')
    h = np.empty_like(y)
    for row, alloy_id in enumerate(ids):
        for col, element in enumerate(ELEMENTS):
            record = head[alloy_id, element]
            if abs(float(record['recorded_wt_pct']) - y[row, col]) > 1e-4:
                raise ValueError('Head labels differ from feature labels')
            h[row, col] = float(record['predicted_wt_pct'])
    if not np.isfinite(h).all(): raise ValueError('Non-finite head predictions')
    methods = {'均值_训练内': np.broadcast_to(y.mean(0), y.shape),
               '中位数_训练内': np.broadcast_to(np.median(y, axis=0), y.shape), '成分头_训练内': h}
    for alpha in args.alphas:
        methods[f'岭回归alpha{alpha:g}_训练内'] = ridge_predict(x, y, x, alpha)
    loo = {'均值_LOO': np.empty_like(y), '中位数_LOO': np.empty_like(y)}
    for alpha in args.alphas: loo[f'岭回归alpha{alpha:g}_LOO'] = np.empty_like(y)
    for i in range(len(ids)):
        mask = np.arange(len(ids)) != i
        loo['均值_LOO'][i] = y[mask].mean(0)
        loo['中位数_LOO'][i] = np.median(y[mask], axis=0)
        for alpha in args.alphas:
            loo[f'岭回归alpha{alpha:g}_LOO'][i] = ridge_predict(x[mask], y[mask], x[i:i+1], alpha)[0]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise FileExistsError('Use a new empty output directory')
    predictions = []
    for label, array in {**methods, **loo}.items():
        for i, alloy_id in enumerate(ids):
            for j, element in enumerate(ELEMENTS):
                predictions.append({'model': label, 'alloy_id': alloy_id, 'element': element,
                                    'recorded_wt_pct': y[i, j], 'predicted_wt_pct': array[i, j],
                                    'abs_error_percentage_points': abs(array[i, j]-y[i, j])})
    with (out / '逐合金预测.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=predictions[0]); writer.writeheader(); writer.writerows(predictions)
    summary = []
    for element in ELEMENTS:
        for label in {**methods, **loo}:
            errors = [r['abs_error_percentage_points'] for r in predictions if r['element'] == element and r['model'] == label]
            summary.append({'element': element, 'model': label, 'n': len(errors), 'mae_percentage_points': np.mean(errors), 'max_abs_error_percentage_points': max(errors)})
        for suffix, labels in [('训练内', list(methods)), ('LOO_编码器已见训练图', list(loo))]:
            values = [next(r['mae_percentage_points'] for r in summary if r['element'] == element and r['model'] == label) for label in labels]
            fig, ax = plt.subplots(figsize=(8, 4)); bars = ax.bar(range(len(values)), values, color='#4c84a3'); ax.bar_label(bars, fmt='%.2f', padding=3)
            ax.set_xticks(range(len(values)), [label.split('_')[0] if not label.startswith('岭') else label.split('_')[0].replace('岭回归', 'Ridge ') for label in labels], rotation=15)
            # ASCII plot labels avoid server font dependence; filenames identify the result.
            ax.set_xticklabels([{'均值':'Mean', '中位数':'Median', '成分头':'VAE head'}.get(label.split('_')[0], label.split('_')[0].replace('岭回归', 'Ridge ')) for label in labels])
            ax.set_ylabel('MAE (wt. percentage points)'); ax.set_title(f'{element}: '+('in-sample' if suffix=='训练内' else 'head LOO, encoder has seen images')); ax.set_ylim(0,max(max(values)*1.2,0.1)); fig.tight_layout(); fig.savefig(out / f'{element}_{suffix}.png', dpi=150); plt.close(fig)
    with (out / '逐元素误差汇总.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0]); writer.writeheader(); writer.writerows(summary)
    std = np.asarray(cfg['train_std_wt_percent'], float); active = np.asarray(cfg['composition_active_mask'], bool)
    params = cfg['args']; scale = np.maximum(std, 0.1)
    if params.get('main_elements_scale', 0): scale[:3] = params['main_elements_scale']
    if params.get('composition_loss_mode', 'uniform') == 'main3':
        coefficients = np.where(np.arange(9) < 3, params.get('main_elements_weight', 0.7)/active[:3].sum(), (1-params.get('main_elements_weight', 0.7))/active[3:].sum())
    else: coefficients = np.full(9, 1/active.sum())
    loss_rows = []
    for j, element in enumerate(ELEMENTS):
        contribution = float(coefficients[j] * np.mean(((h[:, j]-y[:, j])/scale[j])**2)) if active[j] else 0.0
        loss_rows.append({'element': element, 'scale_wt_pct': scale[j], 'mean_loss_before_lambda': contribution, 'mean_loss_after_lambda': contribution * params['composition_weight']})
    with (out/'固定窗口_成分损失分解.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=loss_rows[0]);writer.writeheader();writer.writerows(loss_rows)
    info = {'split': split, 'window_count': payload['windows'], 'alphas': args.alphas, 'features_sha256': hashlib.sha256(Path(args.features).read_bytes()).hexdigest(), 'vae_sha256': payload['vae_sha256'], 'warning': 'LOO refits only ridge and baselines; the VAE saw every training alloy, so this is transductive and is not strict held-out-alloy validation. In-sample ridge may overfit. Loss decomposition uses deterministic fixed-window head predictions, not minibatch gradients. Ridge output is unconstrained.'}
    (out/'实验说明.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {len(ids)} training-alloy diagnostics to {out}', flush=True)

if __name__ == '__main__': main()
