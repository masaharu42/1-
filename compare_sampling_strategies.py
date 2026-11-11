"""Compare DDIM step count, x0 clipping, and VAE-start SDEdit without retraining."""
import argparse,csv,json,math,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diffusers import DDIMScheduler
from data import load_split,AlloyDataset
from train_diffusion import load_condition_vae,make_unet,resize_for_diffusion,vq
from diagnose_diffusion_conditions import sha,load_crop,show
from diagnose_diffusion_color_trajectory import color_statistics
from sample import save_image


def make_scheduler(train_steps,inference_steps,clip,device):
    scheduler=DDIMScheduler(num_train_timesteps=train_steps,beta_schedule='squaredcos_cap_v2',prediction_type='epsilon',clip_sample=clip,clip_sample_range=1.0)
    scheduler.set_timesteps(inference_steps,device=device)
    return scheduler


def choose_sdedit_timestep(scheduler,strength):
    """Strength sets the fraction of the 50-step schedule that will be used."""
    if not 0 < strength <= 1: raise ValueError('SDEdit strengths must be in (0,1]')
    n=len(scheduler.timesteps);count=max(1,min(n,round(strength*n)))
    return n-count


def sample_case(unet,coarse,noise,train_steps,case,device):
    sched=make_scheduler(train_steps,case['steps'],case['clip_x0'],device)
    start=0
    if case.get('strength') is not None:
        start=choose_sdedit_timestep(sched,case['strength'])
        t_start=sched.timesteps[start].reshape(1).to(device)
        image=sched.add_noise(coarse,noise,t_start)
    else:
        image=noise.clone()
    for t in sched.timesteps[start:]:
        pred=unet(torch.cat([image,coarse],dim=1),t).sample
        image=sched.step(pred,t,image,eta=0).prev_sample
        if not torch.isfinite(image).all():raise RuntimeError(f'Non-finite sampling output in {case["label"]} at timestep {int(t)}')
    return image,{'start_timestep':int(sched.timesteps[start].item()),'actual_denoise_steps':len(sched.timesteps)-start}


def dark_line_scores(target, generated, threshold=25, tolerance=2):
    """RGB dark pixels are only a boundary proxy, not EBSD ground truth."""
    def mask(image):return (((image.clamp(-1,1)+1)*127.5)<threshold).all(dim=1,keepdim=True)
    truth=mask(target);pred=mask(generated);n_truth=int(truth.sum());n_pred=int(pred.sum());kernel=2*tolerance+1
    near_truth=F.max_pool2d(truth.float(),kernel,1,tolerance)>0
    near_pred=F.max_pool2d(pred.float(),kernel,1,tolerance)>0
    precision=float((pred & near_truth).sum())/n_pred if n_pred else None
    recall=float((truth & near_pred).sum())/n_truth if n_truth else None
    f1=2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else (0. if precision is not None and recall is not None else None)
    return {'dark_threshold_rgb':threshold,'dark_tolerance_px':tolerance,'reference_dark_pixels':n_truth,'generated_dark_pixels':n_pred,'dark_pixel_precision':precision,'dark_pixel_recall':recall,'dark_pixel_f1':f1}


def eval_grid_positions(width,height,crop,k):
    if min(width,height)<crop:raise ValueError('Image smaller than crop')
    side=math.ceil(math.sqrt(k));ys=np.rint(np.linspace(0,height-crop,side)).astype(int);xs=np.rint(np.linspace(0,width-crop,side)).astype(int)
    grid=[(int(y),int(x)) for y in ys for x in xs];indices=np.rint(np.linspace(0,len(grid)-1,k)).astype(int)
    points=[grid[j] for j in indices]
    if len(set(points))!=len(points):raise ValueError('Evaluation grid has duplicate windows; reduce --windows')
    return points


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vae-checkpoint',required=True);p.add_argument('--diffusion-checkpoint',required=True);p.add_argument('--manifest',default='data/manifest.json');p.add_argument('--split',default='data/split.json')
    p.add_argument('--ids',type=int,nargs='+',default=[18,19]);p.add_argument('--seeds',type=int,nargs='+',default=[42,43]);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--window-mode',choices=['center','grid'],default='center');p.add_argument('--windows',type=int,default=8,help='Fixed evaluation windows per alloy in grid mode');p.add_argument('--experiments',nargs='+',choices=['baseline','more','clip','sdedit'],default=['baseline','more','clip','sdedit']);p.add_argument('--dark-threshold',type=int,default=25);p.add_argument('--dark-tolerance',type=int,default=2)
    p.add_argument('--baseline-steps',type=int,default=50);p.add_argument('--more-steps',type=int,default=100);p.add_argument('--strengths',type=float,nargs='+',default=[.25,.5,.75])
    a=p.parse_args();start_time=time.monotonic();device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if a.windows<1 or not 0<a.dark_threshold<256 or a.dark_tolerance<0 or len(a.experiments)!=len(set(a.experiments)):raise ValueError('Invalid windows, dark-line settings or duplicate experiments')
    if min(a.baseline_steps,a.more_steps)<2 or a.more_steps<=a.baseline_steps:raise ValueError('more-steps must exceed baseline-steps >= 2')
    if len(a.ids)!=len(set(a.ids)) or len(a.seeds)!=len(set(a.seeds)) or len(a.strengths)!=len(set(a.strengths)):raise ValueError('IDs, seeds, strengths must be unique')
    if any(not 0<s<=1 for s in a.strengths):raise ValueError('SDEdit strength must be in (0,1]')
    split=load_split(a.split);info=json.loads(Path(a.manifest).read_text(encoding='utf-8-sig'));records={int(r['alloy_id']):r for r in info['records']}
    if any(i not in records or not any(i in split[k] for k in split) for i in a.ids):raise ValueError('Unknown ID or ID outside split')
    dh=sha(a.diffusion_checkpoint);dc=torch.load(a.diffusion_checkpoint,map_location='cpu',weights_only=False)
    if sha(a.diffusion_checkpoint)!=dh:raise RuntimeError('Diffusion checkpoint changed while loading')
    vh=sha(a.vae_checkpoint);vc=torch.load(a.vae_checkpoint,map_location='cpu',weights_only=False)
    if sha(a.vae_checkpoint)!=vh:raise RuntimeError('VAE checkpoint changed while loading')
    if dc['split']!=split or any(sorted(vc['config'][k+'_ids'])!=sorted(split[k]) for k in split):raise ValueError('Split mismatch')
    mh=sha(a.manifest)
    if dc['vae_sha256']!=vh or dc['manifest_sha256']!=mh or vc['config']['manifest_sha256']!=mh:raise ValueError('Checkpoint provenance mismatch')
    size=int(dc['config']['patch_size']);train_steps=int(dc['config']['timesteps'])
    if a.more_steps>train_steps:raise ValueError('Sampling steps exceed train timesteps')
    vae,crop=load_condition_vae(vc,device,size);unet=make_unet(size,int(dc['config']['base_channels'])).to(device).eval();unet.load_state_dict(dc['unet'])
    all_cases=[{'group':'baseline','label':f'基线_DDIM{a.baseline_steps}_不截断','display':f'DDIM {a.baseline_steps}, no clip','steps':a.baseline_steps,'clip_x0':False}, {'group':'more','label':f'更多步_DDIM{a.more_steps}_不截断','display':f'DDIM {a.more_steps}, no clip','steps':a.more_steps,'clip_x0':False}, {'group':'clip','label':f'逐步截断_DDIM{a.baseline_steps}','display':f'DDIM {a.baseline_steps}, clip x0','steps':a.baseline_steps,'clip_x0':True}]
    all_cases.extend({'group':'sdedit','label':f'VAE加噪_strength{v:g}','display':f'VAE noise strength {v:g}','steps':a.baseline_steps,'clip_x0':False,'strength':v} for v in a.strengths)
    cases=[c for c in all_cases if c['group'] in a.experiments]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Use a new empty output directory')
    rows=[];conditions=[];finished=0;total=len(a.ids)*(a.windows if a.window_mode=='grid' else 1)*len(a.seeds)*len(cases)
    with torch.no_grad():
        for alloy_id in a.ids:
            part=next(k for k in split if alloy_id in split[k])
            center=load_crop(a.manifest,records[alloy_id],crop,device)  # validates source hash
            if a.window_mode=='grid':
                path=Path(a.manifest).resolve().parent / records[alloy_id]['image']
                with Image.open(path) as image: positions=eval_grid_positions(*image.size,crop,a.windows)
                dataset=AlloyDataset(a.manifest,[alloy_id],crop,a.windows,'eval')
                windows=dataset[0]['images']
            else:
                path=Path(a.manifest).resolve().parent / records[alloy_id]['image']
                with Image.open(path) as image: positions=[((image.height-crop)//2,(image.width-crop)//2)]
                windows=center.cpu()
            for window_index,(y,x) in enumerate(positions):
                target=resize_for_diffusion(windows[window_index:window_index+1].to(device),size);coarse=vq(vae,target)
                conditions.extend([{'id':alloy_id,'partition':part,'window_index':window_index,'crop_y':y,'crop_x':x,'kind':name,**dark_line_scores(target,img,a.dark_threshold,a.dark_tolerance),**color_statistics(img)} for name,img in [('real',target),('vae',coarse)]])
                for seed in a.seeds:
                    folder=out/f'ID{alloy_id:02d}_{part}'/f'窗口{window_index:02d}_y{y}_x{x}'/f'seed{seed}';folder.mkdir(parents=True)
                    gen=torch.Generator(device=device).manual_seed(seed);noise=torch.randn(target.shape,generator=gen,device=device);generated=[]
                    for case in cases:
                        result,actual=sample_case(unet,coarse,noise,train_steps,case,device)
                        raw=color_statistics(result);clipped=result.clamp(-1,1);mae=float((clipped-target).abs().mean()*127.5);label=case['label'];save_image(result[0],folder/f'{label}_生成图.png')
                        row={'id':alloy_id,'partition':part,'window_index':window_index,'crop_y':y,'crop_x':x,'seed':seed,'case':label,'mode':'sdedit' if 'strength' in case else 'noise_start','configured_steps':case['steps'],'strength':case.get('strength',''),'clip_x0_each_step':case['clip_x0'],**actual,'RGB_MAE_0_255':mae,**dark_line_scores(target,result,a.dark_threshold,a.dark_tolerance),**raw};rows.append(row)
                        generated.append((case,show(result),actual,mae))
                        fig,axes=plt.subplots(1,3,figsize=(12,4))
                        for ax,img,title in zip(axes,[show(target),show(coarse),show(result)],['Real crop','VAE condition',f'{case["display"]}\nRGB MAE: {mae:.2f}']):ax.imshow(img,interpolation='nearest');ax.set_title(title);ax.axis('off')
                        fig.suptitle(f'ID{alloy_id:02d}, seed={seed}; paired RGB error is not grain-boundary validity');fig.tight_layout();fig.savefig(folder/f'{label}_原图条件生成对照.png',dpi=150);plt.close(fig)
                        finished+=1;elapsed=time.monotonic()-start_time;eta=(total-finished)*elapsed/finished;print(f'{finished}/{total} ID{alloy_id:02d} window={window_index} seed={seed} {label} start_t={actual["start_timestep"]} steps={actual["actual_denoise_steps"]} elapsed={elapsed:.0f}s eta~{eta:.0f}s',flush=True)
                    items=[('Real crop',show(target)),('VAE condition',show(coarse))]+[(c['display'],image) for c,image,_,_ in generated]
                    fig,axes=plt.subplots(math.ceil(len(items)/4),4,figsize=(14,3.5*math.ceil(len(items)/4)),squeeze=False)
                    for ax,(title,img) in zip(axes.flat,items):ax.imshow(img,interpolation='nearest');ax.set_title(title,fontsize=9);ax.axis('off')
                    for ax in list(axes.flat)[len(items):]:ax.axis('off')
                    fig.suptitle(f'ID{alloy_id:02d}, seed {seed}; output display clips to [-1,1]');fig.tight_layout();fig.savefig(folder/'全部采样方法并排.png',dpi=150);plt.close(fig)
    for name,data in [('逐方案结果.csv',rows),('真实与VAE颜色统计.csv',conditions)]:
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
    metadata={'arguments':vars(a),'split':split,'diffusion_checkpoint_step':dc.get('step'),'diffusion_checkpoint_sha256':dh,'vae_sha256':vh,'manifest_sha256':mh,'diffusion_excluded_train_ids':dc.get('config',{}).get('exclude_train_ids',[]),'cases':cases,'elapsed_seconds':time.monotonic()-start_time,'notes':'Dark pixel overlap is a thresholded RGB proxy, not validated grain boundary ground truth; within-tolerance matches need not be one-to-one. Uses same noise tensor for all variants within each ID and seed. Clip applies to predicted x0 each denoising step, not only at export. SDEdit starts by adding noise to the VAE reconstruction, which differs from the training noisy-image distribution. RGB MAE is not a physical validity or inverse-design metric. ID19 already unblinded.'}
    (out/'运行记录.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved results to',out,flush=True)
if __name__=='__main__':main()
