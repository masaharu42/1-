"""Train an alloy-split, composition-supervised 128-dimensional vector VAE."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import AlloyDataset, ELEMENTS, load_split
from vector_vae import VectorVAE


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/manifest.json')
    parser.add_argument('--split', default='data/split.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--alloys-per-batch', type=int, default=2)
    parser.add_argument('--windows', type=int, default=4)
    parser.add_argument('--eval-windows', type=int, default=16)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--kl-weight', type=float, default=0.000001)
    parser.add_argument('--composition-weight', type=float, default=0.0001)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-flip', action='store_true')
    parser.add_argument('--train-rotate', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.alloys_per_batch, args.windows, args.eval_windows, args.eval_every, args.save_every) < 1:
        raise ValueError('Batch and step counts must be positive')
    if args.kl_weight < 0 or args.composition_weight < 0:
        raise ValueError('Loss weights must be nonnegative')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    split = load_split(args.split)
    train = AlloyDataset(args.manifest, split['train'], 256, args.windows, 'train', args.seed, args.train_flip, args.train_rotate)
    val = AlloyDataset(args.manifest, split['val'], 256, args.eval_windows, 'eval', args.seed)
    root = Path(args.manifest).resolve().parent
    for record in train.records + val.records:
        path = (root / record['image']).resolve()
        if not path.is_relative_to(root) or digest(path) != record['sha256']:
            raise ValueError('Image path or checksum mismatch')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Use a new empty output directory')
    std = np.std(train.compositions, axis=0)
    active = torch.tensor(std > 1e-6, device=device)
    if not all(std[ELEMENTS.index(e)] > 1e-6 for e in ('Ni', 'Fe', 'Cr')) or not bool(active[3:].any()):
        raise ValueError('Need varying main and other elements in training')
    scale = torch.tensor(np.maximum(std, 0.1), device=device)
    config = {'args': vars(args), 'train_ids': split['train'], 'val_ids': split['val'], 'test_ids': split['test'], 'manifest_sha256': digest(args.manifest), 'split_sha256': digest(args.split), 'composition_elements': list(ELEMENTS), 'train_std_wt_percent': std.tolist(), 'composition_active_mask': active.cpu().tolist(), 'resize': 'nearest 256 to 128'}
    (out / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    model = VectorVAE().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(train, batch_size=args.alloys_per_batch, shuffle=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False)
    def evaluate(batch, training):
        images = batch['images'].to(device); target = batch['composition'].to(device); b, k = images.shape[:2]
        images = F.interpolate(images.reshape(b*k, 3, 256, 256), size=(128, 128), mode='nearest')
        means, logvar = model.encode(images)
        latent = means + torch.randn_like(means) * torch.exp(0.5 * logvar) if training else means
        reconstruction = model.decode(latent)
        reconstruction_loss = F.l1_loss(reconstruction, images)
        kl = (-0.5 * (1 + logvar - means.square() - logvar.exp())).sum(dim=1).mean()
        pred = model.composition(means.reshape(b, k, 128))
        errors = ((pred - target) / scale).square()
        comp = 0.7 * errors[:, :3].mean() + 0.3 * errors[:, 3:][:, active[3:]].mean()
        total = reconstruction_loss + args.kl_weight * kl + args.composition_weight * comp
        mae = (pred - target).abs().mean(dim=0)
        return total, reconstruction_loss, kl, comp, *mae[:3]
    fields = ['step', 'phase', 'total', 'reconstruction_l1', 'kl', 'composition_loss', 'Ni_mae_wt', 'Fe_mae_wt', 'Cr_mae_wt']
    with (out / 'metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); iterator = iter(train_loader)
        for step in range(1, args.steps + 1):
            try: batch = next(iterator)
            except StopIteration: iterator = iter(train_loader); batch = next(iterator)
            model.train(); optimizer.zero_grad(); values = evaluate(batch, True)
            if not torch.isfinite(values[0]): raise RuntimeError('Non-finite loss')
            values[0].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
            if step == 1 or step % 10 == 0:
                row = dict(zip(fields, [step, 'train'] + [float(v.detach()) for v in values])); writer.writerow(row); f.flush(); print(json.dumps(row), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                model.eval()
                with torch.no_grad(): result = [evaluate(item, False) for item in val_loader]
                means = np.array([[float(x) for x in row] for row in result]).mean(axis=0)
                row = dict(zip(fields, [step, 'val'] + means.tolist())); writer.writerow(row); f.flush(); print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                torch.save({'model': model.state_dict(), 'config': config, 'step': step}, out / f'checkpoint_{step:07d}.pt')
    print('Finished. Test alloy was not loaded.', flush=True)

if __name__ == '__main__': main()
