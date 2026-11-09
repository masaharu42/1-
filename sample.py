"""Run image-space DDIM conditioned on a VAE reconstruction.
Reference mode is reconstruction/refinement, not unconditional new alloy generation.
Alternatively --latent accepts a spatial latent [1,4,H/4,W/4] or vector latent [1,128].
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from diffusers import DDIMScheduler
from vector_vae import VectorVAE
from train_diffusion import make_unet, load_condition_vae, resize_for_diffusion

def save_image(t,path):
 arr=((t.detach().float().clamp(-1,1).cpu().permute(1,2,0).numpy()+1)*127.5).round().astype(np.uint8);Image.fromarray(arr).save(path)
def run():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--vae-checkpoint',required=True);p.add_argument('--diffusion-checkpoint',required=True);group=p.add_mutually_exclusive_group(required=True);group.add_argument('--reference',help='real image for reconstruction/refinement');group.add_argument('--latent',help='external latent tensor .pt');p.add_argument('--output',required=True);p.add_argument('--sample-steps',type=int,default=50);p.add_argument('--count',type=int,default=4);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda');args=p.parse_args();device=torch.device(args.device)
 if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use --device cpu for smoke test')
 ck=torch.load(args.diffusion_checkpoint,map_location='cpu',weights_only=False);vc=torch.load(args.vae_checkpoint,map_location='cpu',weights_only=False)
 for k in ['train','val','test']:
  if sorted(map(int,vc['config'][k+'_ids']))!=sorted(map(int,ck['split'][k])):raise ValueError('VAE split mismatch')
 actual_hash=hashlib.sha256(Path(args.vae_checkpoint).read_bytes()).hexdigest()
 if ck.get('vae_sha256')!=actual_hash:raise ValueError('VAE checkpoint does not match the one used in diffusion training')
 size=int(ck['config']['patch_size']);vae,source_patch=load_condition_vae(vc,device,size);unet=make_unet(size,int(ck['config']['base_channels'])).to(device).eval();unet.load_state_dict(ck['unet']);sched=DDIMScheduler(num_train_timesteps=int(ck['config']['timesteps']),beta_schedule='squaredcos_cap_v2',prediction_type='epsilon',clip_sample=False);sched.set_timesteps(args.sample_steps,device=device)
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()):raise FileExistsError(f'{out} is not empty; choose a new --output to preserve previous samples')
 if args.count<1 or not 1<=args.sample_steps<=int(ck['config']['timesteps']):raise ValueError('count must be positive and sample steps must fit the training schedule')
 with torch.no_grad():
  if args.reference:
   img=Image.open(args.reference).convert('RGB');w,h=img.size
   if min(w,h)<source_patch:raise ValueError('reference image smaller than source crop size')
   img=img.crop(((w-source_patch)//2,(h-source_patch)//2,(w-source_patch)//2+source_patch,(h-source_patch)//2+source_patch));arr=np.asarray(img,dtype=np.float32)/127.5-1; x=torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device);x=resize_for_diffusion(x,size);z=vae.encode(x)[0] if isinstance(vae,VectorVAE) else vae.encode(x).latent_dist.mean;save_image(x[0],out/'reference_crop.png')
  else:
   z=torch.load(args.latent,map_location=device,weights_only=True)
   expected=(1,128) if isinstance(vae,VectorVAE) else (1,4,size//4,size//4)
   if not isinstance(z,torch.Tensor) or z.shape!=expected:raise ValueError(f'latent shape must be {expected}')
  coarse=(vae.decode(z) if isinstance(vae,VectorVAE) else vae.decode(z).sample).clamp(-1,1);save_image(coarse[0],out/'coarse.png');gen=torch.Generator(device=device).manual_seed(args.seed)
  for i in range(args.count):
   image=torch.randn((1,3,size,size),generator=gen,device=device)
   for t in sched.timesteps:
    pred=unet(torch.cat([image,coarse],1),t).sample;image=sched.step(pred,t,image,eta=0).prev_sample
   save_image(image[0],out/f'refined_{i:03d}.png');print(f'Generated {i+1}/{args.count}',flush=True)
 (out/'sample_config.json').write_text(json.dumps(vars(args),indent=2,ensure_ascii=False),encoding='utf-8')
if __name__=='__main__':run()
