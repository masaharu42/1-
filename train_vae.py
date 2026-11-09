"""Train a fresh VAE with optional per-alloy composition supervision.

No final-test samples are ever passed to either training or validation.
"""
import argparse,csv,json,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import AlloyDataset,load_split,ELEMENTS
from model import build_vae,CompositionHead,sobel_magnitude


def arguments():
 p=argparse.ArgumentParser()
 p.add_argument('--manifest',default='data/manifest.json');p.add_argument('--split',default='data/split.json');p.add_argument('--output',required=True)
 p.add_argument('--patch-size',type=int,default=256);p.add_argument('--alloys-per-batch',type=int,default=2);p.add_argument('--windows-per-alloy',type=int,default=4)
 p.add_argument('--eval-windows',type=int,default=16);p.add_argument('--workers',type=int,default=2);p.add_argument('--steps',type=int,default=5000);p.add_argument('--lr',type=float,default=2e-4)
 p.add_argument('--edge-weight',type=float,default=.2);p.add_argument('--kl-weight',type=float,default=1e-6);p.add_argument('--composition-weight',type=float,default=0.0)
 p.add_argument('--composition-loss-mode',choices=['uniform','main3'],default='uniform',help='main3 prioritizes Ni, Fe, Cr; uniform preserves the original loss')
 p.add_argument('--main-elements-weight',type=float,default=0.7,help='Share of composition loss assigned to Ni, Fe, Cr in main3 mode')
 p.add_argument('--main-elements-scale',type=float,default=0.0,help='Fixed wt.%% error scale for Ni, Fe and Cr; 0 retains training standard deviations')
 p.add_argument('--seed',type=int,default=42);p.add_argument('--eval-every',type=int,default=250);p.add_argument('--save-every',type=int,default=500);p.add_argument('--log-every',type=int,default=10)
 p.add_argument('--resume',default=None,help='Resume from a trusted checkpoint within the same output directory.')
 p.add_argument('--train-rotate',action='store_true',help='Random right-angle pixel rotation during training only; orientation and loading axis are not transformed.')
 p.add_argument('--train-flip',action='store_true',help='Enable pixel flips on training windows only; crystallographic invariance not validated.')
 p.add_argument('--corruption-prob',type=float,default=0.0,help='Per-window blur or mask probability; reconstruct clean input.')
 p.add_argument('--precision',choices=['fp32','fp16','bf16'],default='fp16');p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
 return p.parse_args()


def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def composition_loss(pred,target,scale,mask,mode,main_weight):
 errors=((pred-target)/scale).square()
 if mode=='uniform':return errors[:,mask].mean() if mask.any() else errors.new_zeros(())
 main=torch.tensor([e in ('Ni','Fe','Cr') for e in ELEMENTS],device=pred.device,dtype=torch.bool)
 primary=main & mask;other=(~main) & mask
 if int(primary.sum())!=3 or not bool(other.any()):raise ValueError('main3 requires varying Ni, Fe, Cr and at least one varying other element in training')
 return main_weight*errors[:,primary].mean()+(1-main_weight)*errors[:,other].mean()


def main():
 args=arguments()
 if args.patch_size!=256: raise ValueError('Model pilot uses 256x256 crops; change model and documented protocol to use another size.')
 if min(args.alloys_per_batch,args.windows_per_alloy,args.eval_windows,args.steps,args.eval_every,args.save_every,args.log_every)<1: raise ValueError('Positive batch, steps and intervals required')
 if not 0<=args.corruption_prob<=1: raise ValueError('corruption probability must be between 0 and 1')
 if args.composition_weight<0: raise ValueError('composition weight must be nonnegative')
 if not 0<args.main_elements_weight<1:raise ValueError('main elements weight must be strictly between 0 and 1')
 if not np.isfinite(args.main_elements_scale) or args.main_elements_scale<0:raise ValueError('main elements scale must be finite and nonnegative')
 if args.main_elements_scale and args.composition_loss_mode!='main3':raise ValueError('main elements scale requires main3 mode')
 if args.composition_loss_mode=='main3' and args.composition_weight<=0:raise ValueError('main3 requires positive composition weight')
 if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA not available; install CUDA PyTorch on training server or explicitly use --device cpu.')
 if args.device=='cpu' and args.precision!='fp32': raise ValueError('CPU requires --precision fp32')
 seed_all(args.seed);device=torch.device(args.device);sp=load_split(args.split)
 train=AlloyDataset(args.manifest,sp['train'],patch_size=args.patch_size,windows_per_alloy=args.windows_per_alloy,mode='train',seed=args.seed,train_flip=args.train_flip,train_rotate=args.train_rotate)
 val=AlloyDataset(args.manifest,sp['val'],patch_size=args.patch_size,windows_per_alloy=args.eval_windows,mode='eval',seed=args.seed)
 if len(train)<args.alloys_per_batch: raise ValueError('training alloys fewer than batch size')
 # Check file bytes against the precomputed data manifest, including on resume.
 data_root=Path(args.manifest).resolve().parent
 for rec in train.records+val.records:
  path=(data_root/rec['image']).resolve()
  if not path.is_relative_to(data_root): raise ValueError(f'Image path escapes data directory: {path}')
  expected=rec.get('sha256')
  if not expected: raise ValueError(f'Missing image hash in manifest for alloy {rec["alloy_id"]}')
  digest=hashlib.sha256(path.read_bytes()).hexdigest()
  if digest!=expected: raise ValueError(f'Image hash mismatch for alloy {rec["alloy_id"]}: {path}')
 loader=DataLoader(train,batch_size=args.alloys_per_batch,shuffle=True,num_workers=args.workers,pin_memory=device.type=='cuda',drop_last=False)
 vloader=DataLoader(val,batch_size=1,shuffle=False,num_workers=args.workers,pin_memory=device.type=='cuda')
 vae=build_vae().to(device);head=CompositionHead().to(device)
 params=list(vae.parameters())+(list(head.parameters()) if args.composition_weight>0 else [])
 opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=1e-4)
 dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[args.precision]
 scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and args.precision=='fp16')
 # Fixed train-only per-element scaling; constant columns excluded from regression loss.
 values=np.asarray(train.compositions,dtype=np.float32);std=values.std(axis=0);active=std>1e-6
 scale=torch.tensor(np.maximum(std,0.1),device=device);mask=torch.tensor(active,device=device)
 if args.main_elements_scale:
  for element in ('Ni','Fe','Cr'):scale[ELEMENTS.index(element)]=args.main_elements_scale
 if args.composition_loss_mode=='main3' and (not all(active[ELEMENTS.index(e)] for e in ('Ni','Fe','Cr')) or not any(active[i] for i,e in enumerate(ELEMENTS) if e not in ('Ni','Fe','Cr'))):raise ValueError('main3 requires variable Ni, Fe, Cr and at least one variable other element in training')
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 cfg={'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),'split_sha256':hashlib.sha256(Path(args.split).read_bytes()).hexdigest(),'args':vars(args).copy(),'train_ids':sp['train'],'val_ids':sp['val'],'test_ids':sp['test'],'composition_elements':list(ELEMENTS),'train_std_wt_percent':std.tolist(),'composition_active_mask':active.tolist(),'image_scale':'[-1,1]','latent_channels':4,'vae_downsample_factor':4}
 metrics=out/'metrics.csv';fields=['step','phase','total','reconstruction_l1','sobel_l1','kl','composition_loss','composition_mae_wt','lr','elapsed_seconds','gpu_peak_gb','Ni_mae_wt','Fe_mae_wt','Cr_mae_wt','weighted_composition_loss','encoder_grad_image','encoder_grad_composition']
 step=0;prior_elapsed=0.0
 if args.resume:
  ckpt_path=Path(args.resume).resolve()
  if ckpt_path.parent!=out.resolve():raise ValueError('Resume checkpoint must be in --output directory, preserving its logs.')
  if not metrics.is_file() or not (out/'run_config.json').is_file():raise FileNotFoundError('Resume requires existing run_config.json and metrics.csv')
  # Only use trusted, locally produced checkpoints: optimizer and RNG states are pickled.
  state=torch.load(ckpt_path,map_location='cpu',weights_only=False)
  old=state['config']; old_args=old['args']; current=vars(args)
  for key in ('manifest_sha256','split_sha256','train_ids','val_ids','test_ids','composition_elements','train_std_wt_percent','composition_active_mask'):
   if old.get(key)!=cfg.get(key):raise ValueError(f'Resume configuration mismatch: {key}')
  immutable=('patch_size','alloys_per_batch','windows_per_alloy','eval_windows','workers','lr','edge_weight','kl_weight','composition_weight','seed','precision','device','train_flip','train_rotate','corruption_prob')
  for key in immutable:
   if old_args.get(key)!=current[key]:raise ValueError(f'Resume argument mismatch: {key}: {old_args.get(key)} != {current[key]}')
  for key,default in (('composition_loss_mode','uniform'),('main_elements_weight',0.7),('main_elements_scale',0.0)):
   if old_args.get(key,default)!=current[key]:raise ValueError(f'Resume argument mismatch: {key}')
  with metrics.open(newline='',encoding='utf-8') as f:existing_fields=next(csv.reader(f))
  if existing_fields in (fields[:-6],fields[:-3]):fields=existing_fields
  elif existing_fields!=fields:raise ValueError('Unexpected metrics header; refusing to append')
  if 'scaler' not in state:raise ValueError('Checkpoint has no AMP scaler state; cannot safely resume.')
  step=int(state['step'])
  if args.steps<=step:raise ValueError('Requested --steps must exceed checkpoint step.')
  vae.load_state_dict(state['vae']); opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
  if args.composition_weight>0:
   if state['composition_head'] is None:raise ValueError('Checkpoint has no composition head')
   head.load_state_dict(state['composition_head'])
  torch.set_rng_state(state['torch_rng_state'].cpu());np.random.set_state(state['numpy_rng_state']);random.setstate(state['python_rng_state'])
  if device.type=='cuda':
   if 'cuda_rng_state' not in state:raise ValueError('Checkpoint missing CUDA RNG state')
   torch.cuda.set_rng_state_all(state['cuda_rng_state'])
  prior_elapsed=float(state.get('elapsed_seconds',0.0));cfg=old
  with (out/'resume_events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'checkpoint':str(ckpt_path),'checkpoint_step':step,'new_total_steps':args.steps,'note':'Previous log rows after checkpoint remain; DataLoader restarts and crop order is not bitwise identical.'})+'\n')
  print(f'Resuming step {step}; previous log rows are preserved. DataLoader crop order is not guaranteed identical.',flush=True)
 else:
  if any(out.iterdir()):raise FileExistsError(f'{out} is not empty: use a new --output to preserve old runs')
  (out/'run_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
  with metrics.open('w',newline='',encoding='utf-8') as f:csv.DictWriter(f,fieldnames=fields).writeheader()
 start=time.monotonic()
 def calc(batch,training):
  x=batch['images'].to(device,non_blocking=True); c=batch['composition'].to(device,non_blocking=True);B,K=x.shape[:2]; flat=x.reshape(B*K,*x.shape[2:])
  damaged=flat
  if training and args.corruption_prob>0:
   damaged=flat.clone()
   for i in range(len(damaged)):
    if torch.rand((),device=device).item()<args.corruption_prob:
     if torch.rand((),device=device).item()<0.5:
      damaged[i:i+1]=F.avg_pool2d(damaged[i:i+1],kernel_size=3,stride=1,padding=1)
     else:
      h,w=damaged.shape[-2:]; size=max(2,int(min(h,w)*0.08)); yy=int(torch.randint(h-size+1,(),device=device));xx=int(torch.randint(w-size+1,(),device=device));damaged[i,:,yy:yy+size,xx:xx+size]=0
  with torch.autocast(device_type=device.type,dtype=dtype,enabled=device.type=='cuda' and args.precision!='fp32'):
   posterior=vae.encode(damaged).latent_dist;z=posterior.sample() if training else posterior.mode(); recon=vae.decode(z).sample
   rec=F.l1_loss(recon.float(),flat.float());edge=F.l1_loss(sobel_magnitude(recon),sobel_magnitude(flat));kl=posterior.kl().float().mean()
   # Regression from the posterior mean, while the reconstruction still samples z.
   means=posterior.mode().reshape(B,K,*z.shape[1:]);pred=head(means) if args.composition_weight>0 else None
   if pred is not None:
    comp=composition_loss(pred,c,scale,mask,args.composition_loss_mode,args.main_elements_weight)
    mae=(pred-c).abs().mean()
   else:comp=rec.new_zeros(());mae=rec.new_full((),float('nan'))
   loss=rec+args.edge_weight*edge+args.kl_weight*kl+args.composition_weight*comp
  per_element=(pred-c).abs().mean(dim=0) if pred is not None else torch.full((len(ELEMENTS),),float('nan'),device=device)
  return loss,rec,edge,kl,comp,mae,per_element[0],per_element[1],per_element[2]
 def record(step,phase,vals,gradient_norms=None):
  nums=[float(v.detach().cpu()) for v in vals];row=dict(zip(fields[:8],[step,phase]+nums[:6]));row.update(lr=opt.param_groups[0]['lr'],elapsed_seconds=round(prior_elapsed+time.monotonic()-start,2),gpu_peak_gb=round(torch.cuda.max_memory_allocated()/1024**3,3) if device.type=='cuda' else 0)
  if len(fields)>11:row.update(Ni_mae_wt=nums[6],Fe_mae_wt=nums[7],Cr_mae_wt=nums[8])
  if len(fields)>14:
   row['weighted_composition_loss']=nums[4]*args.composition_weight
   if gradient_norms is not None:row.update(encoder_grad_image=gradient_norms[0],encoder_grad_composition=gradient_norms[1])
  with metrics.open('a',newline='',encoding='utf-8') as f:csv.DictWriter(f,fieldnames=fields).writerow(row)
  print(f"{phase:5} step {step:6}/{args.steps} loss={nums[0]:.5f} L1={nums[1]:.5f} edge={nums[2]:.5f} KL={nums[3]:.4f} comp={nums[4]:.4f} comp_MAE={nums[5]:.3f} Ni={nums[6]:.3f} Fe={nums[7]:.3f} Cr={nums[8]:.3f}wt% elapsed={row['elapsed_seconds']:.0f}s peak={row['gpu_peak_gb']:.2f}GB",flush=True)
 def save(step):
  state={'vae':vae.state_dict(),'composition_head':head.state_dict() if args.composition_weight>0 else None,'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'step':step,'elapsed_seconds':prior_elapsed+time.monotonic()-start,'config':cfg,'torch_rng_state':torch.get_rng_state(),'numpy_rng_state':np.random.get_state(),'python_rng_state':random.getstate()}
  if device.type=='cuda':state['cuda_rng_state']=torch.cuda.get_rng_state_all()
  path=out/f'checkpoint_{step:07d}.pt'; version=1
  while path.exists():
   path=out/f'checkpoint_{step:07d}_resume{version}.pt';version+=1
  tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)
 def evaluate(step):
  vae.eval();head.eval();totals=[]
  with torch.no_grad():
   for batch in vloader:totals.append([float(x.detach().cpu()) for x in calc(batch,False)])
  if not totals:raise ValueError('validation split is empty')
  record(step,'val',[torch.tensor(x) for x in np.mean(totals,axis=0)]);vae.train();head.train()
 vae.train();head.train()
 while step<args.steps:
  for batch in loader:
   opt.zero_grad(set_to_none=True);vals=calc(batch,True)
   if not torch.isfinite(vals[0]):raise RuntimeError(f'Non-finite loss at step {step+1}; checkpoint preserved.')
   next_step=step+1;gradient_norms=None
   if args.composition_weight>0 and (next_step%args.log_every==0 or next_step==1) and len(fields)>14:
    shared=vae.encoder.conv_out.weight
    image_loss=vals[1]+args.edge_weight*vals[2]+args.kl_weight*vals[3]
    # Match AMP loss scaling used by backward to avoid fp16 diagnostic underflow.
    image_grad=torch.autograd.grad(scaler.scale(image_loss),shared,retain_graph=True)[0]
    comp_grad=torch.autograd.grad(scaler.scale(args.composition_weight*vals[4]),shared,retain_graph=True)[0]
    diagnostic_scale=scaler.get_scale()
    gradient_norms=(float((image_grad.detach().float()/diagnostic_scale).norm().cpu()),float((comp_grad.detach().float()/diagnostic_scale).norm().cpu()))
   scaler.scale(vals[0]).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(params,1.0);scaler.step(opt);scaler.update();step=next_step
   if step%args.log_every==0 or step==1:record(step,'train',vals,gradient_norms)
   if step%args.eval_every==0 or step==args.steps:evaluate(step)
   if step%args.save_every==0 or step==args.steps:save(step)
   if step>=args.steps:break
 print('Done. Test split was never loaded; use separately for final evaluation.',flush=True)
if __name__=='__main__':main()
