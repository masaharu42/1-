"""Inspect interpolation and standard-normal samples from the 128D VAE."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import AlloyDataset, load_split
from vector_vae import VectorVAE


def save_image(tensor, path):
    array = ((tensor.detach().cpu().clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--manifest', default='data/manifest.json'); p.add_argument('--split', default='data/split.json')
    p.add_argument('--id-a', type=int, required=True); p.add_argument('--id-b', type=int, required=True)
    p.add_argument('--output', required=True); p.add_argument('--device', default='cuda'); p.add_argument('--seed', type=int, default=42)
    a = p.parse_args(); split = load_split(a.split); ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False); cfg = ck['config']
    for part in ('train', 'val', 'test'):
        if sorted(cfg[part + '_ids']) != sorted(split[part]): raise ValueError('Checkpoint split mismatch')
    if hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest() != cfg['manifest_sha256']: raise ValueError('Manifest mismatch')
    if a.id_a == a.id_b: raise ValueError('Choose two different alloy IDs')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise FileExistsError('Use an empty output directory')
    ds = AlloyDataset(a.manifest, [a.id_a, a.id_b], 256, 1, 'eval'); root = Path(a.manifest).resolve().parent
    for record in ds.records:
        path = (root / record['image']).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']: raise ValueError('Image checksum mismatch')
    dev = torch.device(a.device); model = VectorVAE().to(dev); model.load_state_dict(ck['model']); model.eval(); torch.manual_seed(a.seed)
    with torch.no_grad():
        images = torch.cat([ds[i]['images'] for i in range(2)]).to(dev); images = F.interpolate(images, size=(128, 128), mode='nearest')
        means, _ = model.encode(images); weights = torch.linspace(0, 1, 6, device=dev).unsqueeze(1); z = (1 - weights) * means[0] + weights * means[1]
        prior_latents = torch.randn(4, 128, device=dev)
        interpolated = model.decode(z); prior = model.decode(prior_latents)
        fig, axes = plt.subplots(1, 6, figsize=(15, 3))
        for i, ax in enumerate(axes):
            save_image(interpolated[i], out / f'插值_{i:02d}.png'); torch.save(z[i:i+1].cpu(), out / f'插值潜变量_{i:02d}.pt'); ax.imshow(((interpolated[i].cpu().clamp(-1,1).permute(1,2,0).numpy()+1)/2)); ax.set_title(f'{i/5:.1f}'); ax.axis('off')
        fig.suptitle(f'Posterior mean interpolation ID{a.id_a:02d} to ID{a.id_b:02d}'); fig.tight_layout(); fig.savefig(out / '潜变量插值总览.png', dpi=150); plt.close(fig)
        for i in range(4):
            save_image(prior[i], out / f'标准正态采样_{i:02d}.png')
            torch.save(prior_latents[i:i+1].cpu(), out / f'标准正态潜变量_{i:02d}.pt')
        save_image(images[0], out / f'ID{a.id_a:02d}_输入窗口.png'); save_image(images[1], out / f'ID{a.id_b:02d}_输入窗口.png')
    (out / 'sample_config.json').write_text(json.dumps(vars(a), ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved posterior-mean interpolation and standard-normal prior samples; inspect prior distribution mismatch.', flush=True)

if __name__ == '__main__': main()
