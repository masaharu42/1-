"""Trace DDIM predicted clean images and pre-export clipping; no training."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diffusers import DDIMScheduler
from data import load_split
from train_diffusion import load_condition_vae,make_unet,resize_for_diffusion,vq
from diagnose_diffusion_conditions import sha,load_crop,show


def color_statistics(tensor):
    """Raw clean-image prediction and final export statistics, in [-1,1]."""
    x=tensor.detach().float(); channels=x.mean(dim=(0,2,3)); clipped=x.clamp(-1,1);clipped_channels=clipped.mean(dim=(0,2,3))
    return {'raw_min':float(x.min()),'raw_max':float(x.max()),'outside_fraction':float(((x < -1)|(x > 1)).float().mean()),'pixels_any_outside_fraction':float(((x < -1)|(x > 1)).any(dim=1).float().mean()),**{f'raw_{name}_255':float((v+1)*127.5) for name,v in zip('RGB',channels)},**{f'clipped_{name}_255':float((v+1)*127.5) for name,v in zip('RGB',clipped_channels)}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--vae-checkpoint',required=True);p.add_argument('--diffusion-checkpoint',required=True);p.add_argument('--manifest',default='data/manifest.json');p.add_argument('--split',default='data/split.json');p.add_argument('--ids',nargs='+',type=int,default=[18,19]);p.add_argument('--output',required=True);p.add_argument('--sample-steps',type=int,default=50);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda');a=p.parse_args()
    start=time.monotonic();device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if a.sample_steps<2 or len(a.ids)!=len(set(a.ids)):raise ValueError('Use at least two sampling steps and unique IDs')
    split=load_split(a.split);manifest=json.loads(Path(a.manifest).read_text(encoding='utf-8-sig'));records={int(r['alloy_id']):r for r in manifest['records']}
    if any(i not in records or not any(i in split[k] for k in split) for i in a.ids):raise ValueError('Unknown IDs or IDs outside split')
    dh=sha(a.diffusion_checkpoint);dc=torch.load(a.diffusion_checkpoint,map_location='cpu',weights_only=False)
    if sha(a.diffusion_checkpoint)!=dh:raise RuntimeError('Diffusion checkpoint changed while loading; use a fixed copy')
    vh=sha(a.vae_checkpoint);vc=torch.load(a.vae_checkpoint,map_location='cpu',weights_only=False)
    if sha(a.vae_checkpoint)!=vh:raise RuntimeError('VAE checkpoint changed while loading')
    if dc['split']!=split or any(sorted(vc['config'][k+'_ids'])!=sorted(split[k]) for k in split):raise ValueError('Split mismatch')
    mh=sha(a.manifest)
    if dc['vae_sha256']!=vh or dc['manifest_sha256']!=mh or vc['config']['manifest_sha256']!=mh:raise ValueError('Checkpoint provenance mismatch')
    size=int(dc['config']['patch_size']);steps=int(dc['config']['timesteps'])
    if a.sample_steps>steps:raise ValueError('Sample steps exceed training schedule')
    vae,crop=load_condition_vae(vc,device,size);unet=make_unet(size,int(dc['config']['base_channels'])).to(device).eval();unet.load_state_dict(dc['unet'])
    scheduler=DDIMScheduler(num_train_timesteps=steps,beta_schedule='squaredcos_cap_v2',prediction_type='epsilon',clip_sample=False);scheduler.set_timesteps(a.sample_steps,device=device)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Choose a new empty output directory')
    all_rows=[];baseline=[]
    indices=sorted(set(round(v*(a.sample_steps-1)) for v in [0,.2,.4,.6,.8,1]))
    with torch.no_grad():
      for alloy_id in a.ids:
        part=next(k for k in split if alloy_id in split[k]);full=load_crop(a.manifest,records[alloy_id],crop,device);target=resize_for_diffusion(full,size);coarse=vq(vae,target)
        folder=out/f'ID{alloy_id:02d}_{part}';folder.mkdir()
        for label,tensor in [('target',target),('vae_condition',coarse)]:baseline.append({'id':alloy_id,'partition':part,'image':label,**color_statistics(tensor)})
        gen=torch.Generator(device=device).manual_seed(a.seed);x=torch.randn(target.shape,generator=gen,device=device);snapshots=[];rows=[]
        print(f'ID{alloy_id:02d} ({part}): tracing {a.sample_steps} DDIM steps',flush=True)
        for idx,t in enumerate(scheduler.timesteps):
            pred=unet(torch.cat([x,coarse],1),t).sample
            result=scheduler.step(pred,t,x,eta=0);x0=result.pred_original_sample.float();x=result.prev_sample
            stats=color_statistics(x0);row={'id':alloy_id,'partition':part,'sample_index':idx+1,'diffusion_timestep':int(t.item()),**stats};rows.append(row);all_rows.append(row)
            if idx in indices:snapshots.append((idx+1,int(t.item()),show(x0)))
            if (idx+1)%10==0 or idx+1==a.sample_steps:print(f'ID{alloy_id:02d} step {idx+1}/{a.sample_steps} elapsed {time.monotonic()-start:.1f}s',flush=True)
        fig,axes=plt.subplots(1,2+len(snapshots),figsize=(3*(2+len(snapshots)),3.5))
        for ax,img,title in zip(axes,[show(target),show(coarse)]+[s[2] for s in snapshots],['Real image','VAE condition']+[f'pred x0\nstep {s[0]} (t={s[1]})' for s in snapshots]):ax.imshow(img,interpolation='nearest');ax.set_title(title);ax.axis('off')
        fig.suptitle(f'ID{alloy_id:02d}: clean-image estimates; previews clipped to [-1,1]');fig.tight_layout();fig.savefig(folder/'01_预测原图_去噪轨迹.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4))
        for ch,color in zip('RGB',['#c44141','#379451','#326dc1']):
            ax.plot([r['sample_index'] for r in rows],[r[f'clipped_{ch}_255'] for r in rows],label=f'predicted {ch}',color=color)
            target_channel=color_statistics(target)[f'clipped_{ch}_255'];ax.axhline(target_channel,color=color,linestyle='--',alpha=.55)
        ax.set(xlabel='DDIM sampling step',ylabel='Mean channel (0–255 after clip)',title='Predicted clean-image RGB; dashed = target');ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(folder/'02_RGB均值随采样变化.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4));ax.plot([r['sample_index'] for r in rows],[r['outside_fraction']*100 for r in rows],label='Channel values outside [-1,1]');ax.plot([r['sample_index'] for r in rows],[r['pixels_any_outside_fraction']*100 for r in rows],label='Pixels with any channel outside');ax.set(xlabel='DDIM sampling step',ylabel='Percent before export clipping',title='Out-of-range predicted clean-image values');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(folder/'03_超出范围比例.png',dpi=160);plt.close(fig)
        # Final image uses the exact save convention of sample.py.
        from sample import save_image
        save_image(x[0],folder/'04_最终生成图.png')
        print(f'ID{alloy_id:02d} complete; elapsed {time.monotonic()-start:.1f}s',flush=True)
    for name,rows in [('逐步颜色与截断统计.csv',all_rows),('真实图与条件图统计.csv',baseline)]:
      with (out/name).open('w',encoding='utf-8-sig',newline='') as f:writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    info={'arguments':vars(a),'split':split,'diffusion_checkpoint_step':dc.get('step'),'diffusion_checkpoint_sha256':dh,'vae_sha256':vh,'manifest_sha256':mh,'diffusion_excluded_train_ids':dc.get('config',{}).get('exclude_train_ids',[]),'timesteps':[int(t.item()) for t in scheduler.timesteps],'elapsed_seconds':time.monotonic()-start,'definition':'pred_original_sample is the scheduler clean-image estimate at each noisy step, not an actual physical intermediate structure. Previews are clipped for display; raw statistics are recorded before clipping. Dashed lines use target image only for offline evaluation.'}
    (out/'运行记录.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved results to',out,flush=True)
if __name__=='__main__':main()
