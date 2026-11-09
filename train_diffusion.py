"""Image-space noise-prediction DDPM conditioned on a frozen VAE reconstruction.
The VAE checkpoint must have exactly the same train/validation/test alloy split.
Validation is deterministic and test IDs are never loaded for training or selection.
"""
from __future__ import annotations
import argparse, csv, json, random, time, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from data import AlloyDataset, load_split
from model import build_vae
from vector_vae import VectorVAE


def parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--manifest',required=True);p.add_argument('--split',required=True);p.add_argument('--vae-checkpoint',required=True);p.add_argument('--output',required=True)
 p.add_argument('--resume',help='Resume last.pt from the same output directory; appends metrics. DataLoader order is not bitwise restored.');p.add_argument('--patch-size',type=int,default=256);p.add_argument('--windows-per-alloy',type=int,default=4);p.add_argument('--batch-size',type=int,default=2,help='number of alloys, each with K windows')
 p.add_argument('--grad-accum',type=int,default=1);p.add_argument('--steps',type=int,default=10000);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--timesteps',type=int,default=1000)
 p.add_argument('--train-flip',action='store_true',help='Apply random pixel flips before deriving the VAE condition')
 p.add_argument('--train-rotate',action='store_true',help='Apply random 90-degree rotations before deriving the VAE condition')
 p.add_argument('--condition-corruption',choices=['none','blur','mask','affine','erode','dilate','mixed'],default='none',help='Experimental damage to VAE condition input; diffusion target remains clean')
 p.add_argument('--corruption-prob',type=float,default=0.2,help='Per-window probability of damaging the condition input')
 p.add_argument('--base-channels',type=int,default=64);p.add_argument('--workers',type=int,default=0);p.add_argument('--seed',type=int,default=42);p.add_argument('--val-every',type=int,default=250);p.add_argument('--save-every',type=int,default=1000);p.add_argument('--log-every',type=int,default=20);p.add_argument('--eval-windows',type=int,default=8);p.add_argument('--amp',choices=['no','fp16','bf16'],default='bf16');p.add_argument('--device',default='cuda');return p

def make_unet(size,base):
 return UNet2DModel(sample_size=size,in_channels=6,out_channels=3,layers_per_block=2,block_out_channels=(base,base*2,base*4),down_block_types=('DownBlock2D','DownBlock2D','DownBlock2D'),up_block_types=('UpBlock2D','UpBlock2D','UpBlock2D'),norm_num_groups=32)

def vq(vae,im):
 # mean avoids random VAE perturbation for validation and ensures stable coarse condition
 with torch.no_grad():
  if isinstance(vae,VectorVAE):return vae.decode(vae.encode(im)[0]).clamp(-1,1)
  return vae.decode(vae.encode(im).latent_dist.mean).sample.clamp(-1,1)

def load_condition_vae(checkpoint,device,patch_size):
 if 'vae' in checkpoint:
  if patch_size!=256:raise ValueError('Spatial VAE requires patch-size 256')
  vae=build_vae();vae.load_state_dict(checkpoint['vae']);source_patch=256
 elif 'model' in checkpoint:
  if patch_size!=128:raise ValueError('128D vector VAE requires patch-size 128')
  vae=VectorVAE();vae.load_state_dict(checkpoint['model']);source_patch=256
 else:raise ValueError('Unsupported VAE checkpoint')
 return vae.to(device).eval().requires_grad_(False),source_patch

def resize_for_diffusion(images,patch_size):
 if images.shape[-2:] == (patch_size,patch_size):return images
 return F.interpolate(images,size=(patch_size,patch_size),mode='nearest')

def corrupt_condition(images,kind,probability):
 """Damage only VAE inputs; the original geometric-transformed target stays clean."""
 if kind=='none' or probability==0:return images
 out=images.clone()
 for index in range(len(images)):
  if torch.rand((),device=images.device).item()>=probability:continue
  operation=kind
  if operation=='mixed':operation=['blur','mask','affine','erode','dilate'][int(torch.randint(5,(),device=images.device).item())]
  image=images[index:index+1]
  if operation=='mask':
   h,w=image.shape[-2:];size=max(2,int(min(h,w)*0.08));y=int(torch.randint(h-size+1,(),device=images.device));x=int(torch.randint(w-size+1,(),device=images.device));out[index,:,y:y+size,x:x+size]=-1
  elif operation=='blur':
   coords=torch.arange(-2,3,device=images.device,dtype=image.dtype);kernel=torch.exp(-coords.square()/2);kernel=kernel/kernel.sum();kernel=(kernel[:,None]*kernel[None,:]).expand(image.shape[1],1,5,5);out[index]=F.conv2d(F.pad(image,(2,2,2,2),mode='replicate'),kernel,groups=image.shape[1])[0]
  elif operation=='affine':
   theta=image.new_tensor([[[1.,0.04,0.],[0.02,1.,0.]]]);grid=F.affine_grid(theta,image.size(),align_corners=False);out[index]=F.grid_sample(image+1,grid,mode='nearest',padding_mode='zeros',align_corners=False)[0]-1
  elif operation=='dilate':out[index]=F.max_pool2d(image,3,stride=1,padding=1)[0]
  elif operation=='erode':out[index]=-F.max_pool2d(-image,3,stride=1,padding=1)[0]
 return out

def validate_splits(checkpoint,split):
 for key in ('train','val','test'):
  ckey=key+'_ids'
  if ckey not in checkpoint.get('config',{}):raise ValueError(f'VAE checkpoint lacks config.{ckey}; cannot verify leakage')
  if sorted(map(int,checkpoint['config'][ckey]))!=sorted(map(int,split[key])):raise ValueError(f'VAE checkpoint {ckey} differs from current split')

def run():
 args=parser().parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 if not args.resume and ((out/'metrics.csv').exists() or (out/'run_config.json').exists()): raise FileExistsError(f'{out} already contains a run; choose a new --output')
 if args.resume and (not (out/'metrics.csv').is_file() or not (out/'run_config.json').is_file()):raise FileNotFoundError('Resume requires metrics.csv and run_config.json in original --output directory')
 if args.resume and Path(args.resume).resolve().parent!=out.resolve():raise ValueError('--resume checkpoint must be in original --output directory')
 if args.steps<1 or min(args.val_every,args.log_every,args.save_every,args.grad_accum,args.batch_size,args.windows_per_alloy,args.eval_windows)<1:raise ValueError('All step counts and batch sizes must be positive')
 if not 0<=args.corruption_prob<=1:raise ValueError('corruption probability must be between 0 and 1')
 random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
 if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; use --device cpu for smoke tests')
 device=torch.device(args.device);amp=args.amp if device.type=='cuda' else 'no';dtype=torch.bfloat16 if amp=='bf16' else torch.float16
 split=load_split(args.split);ck=torch.load(args.vae_checkpoint,map_location='cpu',weights_only=False);validate_splits(ck,split)
 manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
 if ck['config'].get('manifest_sha256')!=manifest_sha256:raise ValueError('VAE checkpoint dataset manifest SHA256 does not match diffusion dataset; cannot validate data provenance')
 vae,source_patch=load_condition_vae(ck,device,args.patch_size)
 train=AlloyDataset(args.manifest,split['train'],patch_size=source_patch,windows_per_alloy=args.windows_per_alloy,mode='train',seed=args.seed,train_flip=args.train_flip,train_rotate=args.train_rotate)
 val=AlloyDataset(args.manifest,split['val'],patch_size=source_patch,windows_per_alloy=args.eval_windows,mode='eval',seed=args.seed)
 dl=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=device.type=='cuda');vl=DataLoader(val,batch_size=1,shuffle=False,num_workers=args.workers)
 if not len(dl) or not len(vl):raise ValueError('Train and validation loaders must be non-empty')
 # Refuse silently changed source images; manifest SHA alone does not protect image contents.
 for record in train.records+val.records:
  path=(Path(args.manifest).resolve().parent/record['image']).resolve()
  if not path.is_relative_to(Path(args.manifest).resolve().parent):raise ValueError('Image path escapes data directory')
  if hashlib.sha256(path.read_bytes()).hexdigest()!=record.get('sha256'):raise ValueError(f'Image checksum mismatch: {path}')
 unet=make_unet(args.patch_size,args.base_channels).to(device);sched=DDPMScheduler(num_train_timesteps=args.timesteps,beta_schedule='squaredcos_cap_v2',prediction_type='epsilon');opt=torch.optim.AdamW(unet.parameters(),lr=args.lr,weight_decay=.01);scaler=torch.amp.GradScaler('cuda',enabled=(amp=='fp16'))
 vae_sha256=hashlib.sha256(Path(args.vae_checkpoint).read_bytes()).hexdigest(); manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(); start=time.time();best=float('inf');step=0;previous_elapsed=0.0;segment=0
 log=out/'metrics.csv'
 if args.resume:
  previous=torch.load(args.resume,map_location='cpu',weights_only=False)
  if previous.get('vae_sha256')!=vae_sha256:raise ValueError('Resume VAE SHA256 mismatch')
  if previous.get('manifest_sha256')!=manifest_sha256:raise ValueError('Resume dataset manifest SHA256 mismatch (or old checkpoint lacks it)')
  if previous.get('split')!=split:raise ValueError('Resume split mismatch')
  required=('patch_size','windows_per_alloy','batch_size','grad_accum','lr','timesteps','base_channels','workers','seed','eval_windows','amp','device')
  old=previous.get('config',{})
  for key in required:
   if old.get(key)!=getattr(args,key):raise ValueError(f'Resume configuration mismatch for {key}: {old.get(key)} vs {getattr(args,key)}')
  for key,default in (('train_flip',False),('train_rotate',False),('condition_corruption','none'),('corruption_prob',0.2)):
   if old.get(key,default)!=getattr(args,key):raise ValueError(f'Resume augmentation mismatch for {key}')
  step=int(previous['step'])
  if step>=args.steps:raise ValueError('--steps must be greater than the checkpoint step')
  with log.open(newline='',encoding='utf-8') as f:
   history=list(csv.DictReader(f));logged=[int(row['step']) for row in history]
  if logged and max(logged)>step:print(f'Preserving pre-interruption log entries through step {max(logged)}; resumed steps after {step} will be repeated in a new segment.',flush=True)
  segment=max([int(previous.get('segment',0))]+[int(row.get('segment') or 0) for row in history])+1
  unet.load_state_dict(previous['unet']);opt.load_state_dict(previous['optimizer']);scaler.load_state_dict(previous['scaler']);best=float(previous.get('best_val',float('inf')));previous_elapsed=float(previous.get('elapsed_sec',0))
  random.setstate(previous['python_rng_state']);np.random.set_state(previous['numpy_rng_state']);torch.set_rng_state(previous['torch_rng_state'])
  if device.type=='cuda' and previous.get('cuda_rng_state') is not None:torch.cuda.set_rng_state_all(previous['cuda_rng_state'])
  print(f'Resuming at step {step}; metrics append; DataLoader position is not restored bitwise.',flush=True)
 else:
  (out/'run_config.json').write_text(json.dumps({**vars(args),'split':split,'vae_config':ck.get('config',{}),'vae_sha256':vae_sha256,'manifest_sha256':manifest_sha256},indent=2,ensure_ascii=False,default=str),encoding='utf-8')
 it=iter(dl);opt.zero_grad(set_to_none=True)
 def evaluate():
  unet.eval(); totals=[];gen=torch.Generator(device=device).manual_seed(args.seed+98765)
  with torch.no_grad():
   for batch in vl:
    im=resize_for_diffusion(batch['images'].flatten(0,1).to(device),args.patch_size);coarse=vq(vae,im);noise=torch.randn(im.shape,generator=gen,device=device);t=torch.randint(0,args.timesteps,(len(im),),generator=gen,device=device);noisy=sched.add_noise(im,noise,t)
    with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp!='no'): pred=unet(torch.cat([noisy,coarse],1),t).sample
    totals.append(F.mse_loss(pred.float(),noise.float()).item())
  unet.train();return float(np.mean(totals))
 def save(path,val_loss):
  state={'unet':unet.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'step':step,'segment':segment,'val_loss':val_loss,'best_val':best,'elapsed_sec':previous_elapsed+time.time()-start,'config':vars(args),'split':split,'vae_checkpoint':str(Path(args.vae_checkpoint).resolve()),'vae_sha256':vae_sha256,'manifest_sha256':manifest_sha256,'python_rng_state':random.getstate(),'numpy_rng_state':np.random.get_state(),'torch_rng_state':torch.get_rng_state()}
  if device.type=='cuda':state['cuda_rng_state']=torch.cuda.get_rng_state_all()
  temp=path.with_suffix(path.suffix+'.tmp');torch.save(state,temp);temp.replace(path)
 with log.open('a' if args.resume else 'w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=['segment','step','train_noise_mse','val_noise_mse','elapsed_sec','lr','gpu_peak_gb'])
  if not args.resume:writer.writeheader()
  unet.train()
  while step<args.steps:
   batch_losses=[]
   for _ in range(args.grad_accum):
    try:batch=next(it)
    except StopIteration:it=iter(dl);batch=next(it)
    im=resize_for_diffusion(batch['images'].flatten(0,1).to(device),args.patch_size);condition=corrupt_condition(im,args.condition_corruption,args.corruption_prob);coarse=vq(vae,condition);noise=torch.randn_like(im);t=torch.randint(0,args.timesteps,(len(im),),device=device);noisy=sched.add_noise(im,noise,t)
    with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp!='no'):pred=unet(torch.cat([noisy,coarse],1),t).sample;loss=F.mse_loss(pred.float(),noise.float())
    if not torch.isfinite(loss): raise RuntimeError('Non-finite loss; stopped before optimizer update')
    scaler.scale(loss/args.grad_accum).backward();batch_losses.append(loss.item())
   scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(unet.parameters(),1.0);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True);step+=1
   v=evaluate() if step%args.val_every==0 or step==args.steps else None
   if step%args.log_every==0 or v is not None:
    record={'segment':segment,'step':step,'train_noise_mse':np.mean(batch_losses),'val_noise_mse':v if v is not None else '', 'elapsed_sec':round(previous_elapsed+time.time()-start,1),'lr':args.lr,'gpu_peak_gb':round(torch.cuda.max_memory_allocated(device)/1024**3,3) if device.type=='cuda' else ''};writer.writerow(record);f.flush();print(json.dumps(record,ensure_ascii=False),flush=True)
   if v is not None and v<best:best=v;save(out/'best.pt',v)
   if step%args.save_every==0 or step==args.steps:save(out/'last.pt',v)
if __name__=='__main__':run()
