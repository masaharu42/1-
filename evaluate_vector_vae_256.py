"""Evaluate a 128D vector VAE and save per-alloy composition and reconstructions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data import AlloyDataset, ELEMENTS, load_split
from vector_vae_256 import VectorVAE256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', default='data/manifest.json')
    parser.add_argument('--split', default='data/split.json')
    parser.add_argument('--partition', choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--windows', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-windows', type=int, default=4)
    args = parser.parse_args()
    if min(args.windows,args.batch_windows) < 1: raise ValueError('windows must be positive')
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    split = load_split(args.split); config = checkpoint['config']
    for part in ('train', 'val', 'test'):
        if sorted(config[part + '_ids']) != sorted(split[part]): raise ValueError('Checkpoint split mismatch')
    if hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest() != config['manifest_sha256']: raise ValueError('Manifest mismatch')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise FileExistsError('Use an empty output directory')
    dataset = AlloyDataset(args.manifest, split[args.partition], 256, args.windows, 'eval')

    if config.get('model_arch') != 'vector_vae_256' or config.get('image_size') != 256: raise ValueError('Expected native 256px VAE checkpoint')
    model = VectorVAE256(int(config['latent_dim'])).to(device); model.load_state_dict(checkpoint['model']); model.eval(); rows = []
    root = Path(args.manifest).resolve().parent
    with torch.no_grad():
        for i, record in enumerate(dataset.records):
            path = (root / record['image']).resolve()
            if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']: raise ValueError('Image checksum mismatch')
            item = dataset[i]; images = item['images'].to(device)
            means = torch.cat([model.encode(chunk)[0] for chunk in images.split(args.batch_windows)]);  reconstruction = model.decode(means[:1]).clamp(-1, 1)
            prediction = model.composition(means.unsqueeze(0))[0].cpu().numpy(); actual = item['composition'].numpy()
            fig, axes = plt.subplots(1, 3, figsize=(10, 4))
            for ax, im, title in zip(axes, [(images[0]+1)/2, (reconstruction[0]+1)/2, (images[0]-reconstruction[0]).abs()/2], ['Input 256x256', 'Vector VAE reconstruction', 'Absolute difference']):
                ax.imshow(im.cpu().permute(1, 2, 0).numpy().clip(0, 1)); ax.set_title(title); ax.axis('off')
            fig.suptitle(f'ID{item["alloy_id"]:02d} first window'); fig.tight_layout(); fig.savefig(out / f'ID{item["alloy_id"]:02d}_向量VAE重建.png', dpi=160); plt.close(fig)
            for j, element in enumerate(ELEMENTS):
                rows.append({'alloy_id': item['alloy_id'], 'element': element, 'recorded_wt_pct': float(actual[j]), 'predicted_wt_pct': float(prediction[j]), 'abs_error_wt_pct': float(abs(prediction[j]-actual[j])), 'active_in_training': config['composition_active_mask'][j]})
            print(f'Evaluated {args.partition} ID{item["alloy_id"]:02d}', flush=True)
    with (out / 'predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    for element in ELEMENTS:
        selected = [row for row in rows if row['element'] == element]
        x = np.array([row['recorded_wt_pct'] for row in selected]); y = np.array([row['predicted_wt_pct'] for row in selected])
        low, high = min(x.min(), y.min()), max(x.max(), y.max()); pad = max((high-low)*0.1, 0.1)
        fig, ax = plt.subplots(figsize=(5, 5)); ax.scatter(x, y); ax.plot([low-pad, high+pad], [low-pad, high+pad], '--', color='gray')
        for row in selected: ax.annotate(str(row['alloy_id']), (row['recorded_wt_pct'], row['predicted_wt_pct']), xytext=(4, 4), textcoords='offset points')
        ax.set(xlabel='Recorded composition (wt.%)', ylabel='Predicted composition (wt.%)', title=f'{element}: {args.partition}')
        fig.tight_layout(); fig.savefig(out / f'{element}_记录值与预测值.png', dpi=160); plt.close(fig)
    (out / 'evaluation_config.json').write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__': main()
