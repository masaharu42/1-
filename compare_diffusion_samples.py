"""Make one paired comparison figure and RGB MAE per generated sample."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=np.uint8).copy()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples-dir',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();source=Path(args.samples_dir);out=Path(args.output)
    cfg=json.loads((source/'sample_config.json').read_text(encoding='utf-8'))
    target=read_image(source/'reference_crop.png');coarse=read_image(source/'coarse.png');samples=sorted(source.glob('refined_*.png'))
    if not samples:raise ValueError('No refined images')
    if target.shape!=coarse.shape:raise ValueError('Target and coarse sizes differ')
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Choose a new empty output directory')
    rows=[]
    for pth in samples:
        generated=read_image(pth)
        if generated.shape!=target.shape:raise ValueError('Generated size differs')
        mae=lambda im:float(np.abs(im.astype(np.float32)-target.astype(np.float32)).mean())
        row={'sample':pth.name,'seed_base':cfg['seed'],'vae_rgb_mae_0_255':mae(coarse),'diffusion_rgb_mae_0_255':mae(generated)};rows.append(row)
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax,img,title in zip(axes,[target,coarse,generated],['Input (256x256)',f'VAE condition (MAE {row["vae_rgb_mae_0_255"]:.1f})',f'Diffusion (MAE {row["diffusion_rgb_mae_0_255"]:.1f})']):
            ax.imshow(img,interpolation='nearest');ax.set_title(title);ax.axis('off')
        fig.suptitle(f'{pth.stem}: paired pixel MAE; not a crystallographic metric');fig.tight_layout();fig.savefig(out/f'{pth.stem}_原图_VAE_扩散对照.png',dpi=160);plt.close(fig)
    with (out/'逐图RGB误差.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    (out/'来源说明.json').write_text(json.dumps({'source':str(source),'sample_config':cfg,'warning':'RGB MAE is not a grain boundary or orientation validity metric; conditional reconstruction requires a reference image.'},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved',len(rows),'paired comparisons to',out,flush=True)
if __name__=='__main__':main()
