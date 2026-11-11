"""Visualize interpolation between two windows using the native 256px vector VAE."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import AlloyDataset
from vector_vae_256 import VectorVAE256


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, help='Trained vector VAE checkpoint')
    p.add_argument('--manifest', required=True, help='Dataset manifest used in training')
    p.add_argument('--id-a', required=True, type=int)
    p.add_argument('--id-b', required=True, type=int)
    p.add_argument('--window-a', type=int, default=0, help='Zero-based evaluation window index')
    p.add_argument('--window-b', type=int, default=0)
    p.add_argument('--windows', type=int, default=8, help='Number of deterministic grid windows per alloy')
    p.add_argument('--steps', type=int, default=7, help='Number of interpolation points, including endpoints')
    p.add_argument('--device', default='cpu', help='cpu or cuda')
    p.add_argument('--output', required=True, help='New or empty output directory')
    args = p.parse_args()
    if args.id_a == args.id_b:
        p.error('Choose two different alloy IDs')
    if args.windows < 1 or args.steps < 2 or not (0 <= args.window_a < args.windows and 0 <= args.window_b < args.windows):
        p.error('Check --windows, --steps and window indices')

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ckpt['config']
    if config.get('model_arch') != 'vector_vae_256' or config.get('image_size') != 256:
        raise ValueError('Expected a native 256px vector VAE checkpoint')
    if sha256(args.manifest) != config['manifest_sha256']:
        raise ValueError('Manifest does not match the VAE checkpoint')
    dataset = AlloyDataset(args.manifest, [args.id_a, args.id_b], 256, args.windows, 'eval')
    root = Path(args.manifest).resolve().parent
    for record in dataset.records:
        image_path = (root / record['image']).resolve()
        if not image_path.is_relative_to(root) or sha256(image_path) != record['sha256']:
            raise ValueError(f"Image checksum or path mismatch: {record['image']}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Use a new or empty output directory')
    device = torch.device(args.device)
    model = VectorVAE256(int(config['latent_dim'])).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    a = dataset[0]['images'][args.window_a].unsqueeze(0).to(device)
    b = dataset[1]['images'][args.window_b].unsqueeze(0).to(device)
    alphas = np.linspace(0, 1, args.steps)
    with torch.no_grad():
        # Posterior means make this comparison reproducible.
        za = model.encode(a)[0]
        zb = model.encode(b)[0]
        zs = torch.cat([(1 - float(t)) * za + float(t) * zb for t in alphas])
        decoded = model.decode(zs).clamp(-1, 1).cpu()
    def rgb(tensor):
        return ((tensor.cpu().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
    fig, axes = plt.subplots(1, args.steps + 2, figsize=(2.4 * (args.steps + 2), 3))
    axes[0].imshow(rgb(a[0])); axes[0].set_title(f'ID{args.id_a:02d} original')
    for i, (t, image) in enumerate(zip(alphas, decoded)):
        axes[i + 1].imshow(rgb(image)); axes[i + 1].set_title(f'alpha={t:.2f}')
        plt.imsave(output / f'interpolation_{i:02d}.png', rgb(image))
    axes[-1].imshow(rgb(b[0])); axes[-1].set_title(f'ID{args.id_b:02d} original')
    for ax in axes: ax.axis('off')
    fig.suptitle('VAE latent interpolation: A → B (posterior means)')
    fig.tight_layout()
    fig.savefig(output / 'interpolation_overview.png', dpi=150)
    plt.close(fig)
    np.savez_compressed(output / 'latents.npz', z_a=za.cpu().numpy(), z_b=zb.cpu().numpy(), z_interpolated=zs.cpu().numpy(), alphas=alphas)
    metadata = vars(args) | {'checkpoint_sha256': sha256(args.checkpoint), 'manifest_sha256': sha256(args.manifest), 'latent_dim': int(config['latent_dim']), 'latent_choice': 'encoder posterior mean', 'partitions': {part: config.get(part + '_ids') for part in ('train', 'val', 'test')}}
    (output / 'run_info.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved results to {output.resolve()}')


if __name__ == '__main__':
    main()
