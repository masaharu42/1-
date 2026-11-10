"""Compare correct and mismatched reconstruction conditions; no training."""
import argparse, csv, hashlib, json, time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler
from data import load_split
from train_diffusion import load_condition_vae, make_unet, resize_for_diffusion, vq


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_crop(manifest, record, crop, device):
    root = Path(manifest).resolve().parent
    path = (root / record['image']).resolve()
    if not path.is_relative_to(root) or sha(path) != record['sha256']:
        raise ValueError('Image path or checksum mismatch: ' + str(path))
    with Image.open(path) as im:
        im = im.convert('RGB'); w, h = im.size
        if min(w, h) < crop: raise ValueError('Image too small: ' + str(path))
        left, top = (w-crop)//2, (h-crop)//2
        a = np.asarray(im.crop((left, top, left+crop, top+crop)), dtype=np.float32).copy()
    return torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to(device)/127.5-1


def show(t):
    return ((t.detach().float().cpu().squeeze(0).permute(1,2,0).numpy()+1)/2).clip(0,1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vae-checkpoint',required=True);p.add_argument('--diffusion-checkpoint',required=True)
    p.add_argument('--manifest',default='data/manifest.json');p.add_argument('--split',default='data/split.json')
    p.add_argument('--ids',type=int,nargs='+',default=[1,18,19]);p.add_argument('--wrong-id',type=int,default=2)
    p.add_argument('--output',required=True);p.add_argument('--sample-steps',type=int,default=50)
    p.add_argument('--noise-repeats',type=int,default=5);p.add_argument('--samples',type=int,default=2)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda')
    a=p.parse_args(); start=time.monotonic(); device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    split=load_split(a.split); manifest=json.loads(Path(a.manifest).read_text(encoding='utf-8-sig'))
    records={int(r['alloy_id']):r for r in manifest['records']}
    if a.wrong_id not in split['train'] or a.wrong_id in a.ids:raise ValueError('Wrong-condition ID must be a different training alloy')
    if len(set(a.ids))!=len(a.ids) or any(i not in records for i in a.ids+[a.wrong_id]):raise ValueError('Unknown or duplicate IDs')
    if min(a.samples,a.noise_repeats,a.sample_steps)<1:raise ValueError('Counts must be positive')
    dc=torch.load(a.diffusion_checkpoint,map_location='cpu',weights_only=False);vc=torch.load(a.vae_checkpoint,map_location='cpu',weights_only=False)
    if dc['split']!=split or any(sorted(vc['config'][k+'_ids'])!=sorted(split[k]) for k in split):raise ValueError('Split mismatch')
    mh=sha(a.manifest);vh=sha(a.vae_checkpoint)
    if dc['vae_sha256']!=vh or dc['manifest_sha256']!=mh or vc['config']['manifest_sha256']!=mh:raise ValueError('Checkpoint provenance mismatch')
    size=int(dc['config']['patch_size']);steps=int(dc['config']['timesteps'])
    if a.sample_steps>steps:raise ValueError('Sample steps exceed schedule')
    vae,crop=load_condition_vae(vc,device,size);unet=make_unet(size,int(dc['config']['base_channels'])).to(device).eval();unet.load_state_dict(dc['unet'])
    ddpm=DDPMScheduler(num_train_timesteps=steps,beta_schedule='squaredcos_cap_v2',prediction_type='epsilon')
    ddim=DDIMScheduler(num_train_timesteps=steps,beta_schedule='squaredcos_cap_v2',prediction_type='epsilon',clip_sample=False);ddim.set_timesteps(a.sample_steps,device=device)
    times=sorted(set(round(x*(steps-1)) for x in [.05,.25,.5,.75,.95]))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Choose an empty output directory')
    rows=[]; summaries=[]
    with torch.no_grad():
      wrong_src=load_crop(a.manifest,records[a.wrong_id],crop,device);wrong=vq(vae,resize_for_diffusion(wrong_src,size))
      for alloy_id in a.ids:
        part=next((k for k in split if alloy_id in split[k]),None)
        if part is None:raise ValueError('ID not in split')
        print(f'ID{alloy_id:02d} ({part}): starting; elapsed {time.monotonic()-start:.1f}s',flush=True)
        full=load_crop(a.manifest,records[alloy_id],crop,device);x=resize_for_diffusion(full,size);coarse=vq(vae,x)
        folder=out/f'ID{alloy_id:02d}_{part}';folder.mkdir()
        fig,axs=plt.subplots(1,3,figsize=(12,4))
        for ax,img,title in zip(axs,[show(full),show(x),show(coarse)],['Original 256 center crop','Nearest downsampled target','VAE condition']):ax.imshow(img,interpolation='nearest');ax.set_title(title);ax.axis('off')
        fig.tight_layout();fig.savefig(folder/'01_原图缩放与VAE条件.png',dpi=170);plt.close(fig)
        # Same central field of view at the two native pixel resolutions.
        original_rgb=show(full); target_rgb=show(x); extent=0.25
        def central(a):
            h,w=a.shape[:2];dy=max(1,round(h*extent/2));dx=max(1,round(w*extent/2))
            return a[h//2-dy:h//2+dy,w//2-dx:w//2+dx]
        fig,axs=plt.subplots(1,2,figsize=(9,4))
        for ax,img,title in zip(axs,[original_rgb,target_rgb],['Original center detail','Nearest resized center detail']):
            ax.imshow(central(img),interpolation='nearest');ax.set_title(title);ax.axis('off')
        fig.suptitle('Same physical field: inspect dark boundaries visually');fig.tight_layout();fig.savefig(folder/'01b_晶界局部放大.png',dpi=200);plt.close(fig)
        for t in times:
          for rep in range(a.noise_repeats):
            gen=torch.Generator(device=device).manual_seed(a.seed+rep)
            noise=torch.randn(x.shape,generator=gen,device=device);tt=torch.tensor([t],device=device,dtype=torch.long);noisy=ddpm.add_noise(x,noise,tt)
            correct=unet(torch.cat([noisy,coarse],1),tt).sample.float();bad=unet(torch.cat([noisy,wrong],1),tt).sample.float()
            good_mse=F.mse_loss(correct,noise.float()).item();bad_mse=F.mse_loss(bad,noise.float()).item()
            rows.append({'id':alloy_id,'partition':part,'wrong_id':a.wrong_id,'t':t,'repeat':rep,'correct_mse':good_mse,'wrong_mse':bad_mse,'delta_wrong_minus_correct':bad_mse-good_mse,'prediction_difference_mse':F.mse_loss(correct,bad).item()})
        sub=[r for r in rows if r['id']==alloy_id];means=[np.mean([r['delta_wrong_minus_correct'] for r in sub if r['t']==t]) for t in times];sd=[np.std([r['delta_wrong_minus_correct'] for r in sub if r['t']==t]) for t in times]
        fig,ax=plt.subplots(figsize=(6,4));ax.errorbar(times,means,yerr=sd,fmt='o-',capsize=4);ax.axhline(0,color='gray');ax.set(xlabel='Diffusion timestep t',ylabel='wrong MSE − correct MSE',title=f'ID{alloy_id:02d}: condition sensitivity');fig.tight_layout();fig.savefig(folder/'02_条件敏感度.png',dpi=160);plt.close(fig)
        for n in range(a.samples):
          gen=torch.Generator(device=device).manual_seed(a.seed+n);initial=torch.randn(x.shape,generator=gen,device=device);outs=[]
          for condition in [coarse,wrong]:
            image=initial.clone()
            for t in ddim.timesteps:image=ddim.step(unet(torch.cat([image,condition],1),t).sample,t,image,eta=0).prev_sample
            outs.append(image.clamp(-1,1))
          fig,axs=plt.subplots(1,4,figsize=(13,4))
          for ax,img,title in zip(axs,[x,coarse,*outs],['Target resized','VAE condition','Correct condition','Mismatched condition']):ax.imshow(show(img),interpolation='nearest');ax.set_title(title);ax.axis('off')
          fig.suptitle(f'ID{alloy_id:02d}, wrong ID{a.wrong_id:02d}, seed {a.seed+n}');fig.tight_layout();fig.savefig(folder/f'03_同噪声条件对照_seed{a.seed+n}.png',dpi=160);plt.close(fig)
          summaries.append({'id':alloy_id,'partition':part,'seed':a.seed+n,'vae_rgb_mae_0_255':float((coarse-x).abs().mean()*127.5),'correct_rgb_mae_0_255':float((outs[0]-x).abs().mean()*127.5),'wrong_rgb_mae_0_255':float((outs[1]-x).abs().mean()*127.5),'condition_output_difference_0_255':float((outs[0]-outs[1]).abs().mean()*127.5)})
        print(f'ID{alloy_id:02d} complete; elapsed {time.monotonic()-start:.1f}s',flush=True)
    for name,data in [('条件敏感度逐次.csv',rows),('配对图像误差.csv',summaries)]:
      with (out/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
    info={'arguments':vars(a),'split':split,'vae_sha256':vh,'diffusion_sha256':sha(a.diffusion_checkpoint),'manifest_sha256':mh,'diffusion_checkpoint_step':dc.get('step'),'elapsed_seconds':time.monotonic()-start,'crop':'center 256; nearest to model size, same as sample.py','interpretation':'RGB MAE is paired pixel difference, not crystallographic validity; condition delta is descriptive, not causation. ID19 and ID14 previously unblinded.'}
    (out/'运行记录.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Results:',out,flush=True)
if __name__=='__main__':main()
