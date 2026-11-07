"""Fit a per-alloy composition head on a frozen, newly trained VAE."""
import argparse,csv,json,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import AlloyDataset,load_split,ELEMENTS
from model import build_vae,CompositionHead

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--vae-checkpoint',required=True);p.add_argument('--manifest',default='data/manifest.json');p.add_argument('--split',default='data/split.json');p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int,default=1000);p.add_argument('--lr',type=float,default=.0001);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--windows',type=int,default=4);p.add_argument('--eval-windows',type=int,default=8);p.add_argument('--eval-every',type=int,default=100);p.add_argument('--seed',type=int,default=42);p.add_argument('--train-flip',action='store_true');p.add_argument('--train-rotate',action='store_true');p.add_argument('--corruption-prob',type=float,default=0.0);a=p.parse_args()
 if min(a.steps,a.batch_size,a.windows,a.eval_windows,a.eval_every)<1:raise ValueError('steps, batch and evaluation sizes must be positive')
 if not 0<=a.corruption_prob<=1:raise ValueError('corruption probability must be between 0 and 1')
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);dev=torch.device(a.device)
 if dev.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use --device cpu')
 sp=load_split(a.split);ck=torch.load(a.vae_checkpoint,map_location='cpu',weights_only=False)
 for k in ('train','val','test'):
  if sorted(map(int,ck['config'][k+'_ids']))!=sorted(sp[k]):raise ValueError('VAE and head splits differ')
 digest=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()
 if ck['config'].get('manifest_sha256')!=digest:raise ValueError('VAE checkpoint and dataset manifest differ')
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if (out/'metrics.csv').exists():raise FileExistsError('Output already has metrics; select a new output directory')
 ds=AlloyDataset(a.manifest,sp['train'],256,a.windows,'train',a.seed,train_flip=a.train_flip,train_rotate=a.train_rotate);vds=AlloyDataset(a.manifest,sp['val'],256,a.eval_windows,'eval',a.seed);dl=DataLoader(ds,batch_size=a.batch_size,shuffle=True,num_workers=0);vl=DataLoader(vds,batch_size=1,shuffle=False,num_workers=0)
 data_root=Path(a.manifest).resolve().parent
 for rec in ds.records+vds.records:
  path=(data_root/rec['image']).resolve()
  if not path.is_relative_to(data_root) or hashlib.sha256(path.read_bytes()).hexdigest()!=rec.get('sha256'):raise ValueError(f'Dataset image integrity check failed for ID {rec["alloy_id"]}')
 vae=build_vae().to(dev);vae.load_state_dict(ck['vae']);vae.eval().requires_grad_(False);head=CompositionHead().to(dev);opt=torch.optim.AdamW(head.parameters(),lr=a.lr);std=np.std(ds.compositions,axis=0);mask=torch.tensor(std>1e-6,device=dev);scale=torch.tensor(np.maximum(std,.1),device=dev);start=time.monotonic()
 def calc(batch,training=False):
  images=batch['images'].to(dev);target=batch['composition'].to(dev);B,K=images.shape[:2];flat=images.reshape(B*K,*images.shape[2:])
  if training and a.corruption_prob>0:
   flat=flat.clone()
   for i in range(len(flat)):
    if torch.rand((),device=dev).item()<a.corruption_prob:
     if torch.rand((),device=dev).item()<0.5:flat[i:i+1]=F.avg_pool2d(flat[i:i+1],3,1,1)
     else:
      h,w=flat.shape[-2:];size=max(2,int(min(h,w)*0.08));y=int(torch.randint(h-size+1,(),device=dev));x=int(torch.randint(w-size+1,(),device=dev));flat[i,:,y:y+size,x:x+size]=0
  with torch.no_grad():z=vae.encode(flat).latent_dist.mode().reshape(B,K,4,64,64)
  pred=head(z);loss=(((pred-target)/scale)[:,mask].square()).mean();mae=(pred-target).abs().mean();return loss,mae
 fields=['step','train_loss','train_mae','val_loss','val_mae','elapsed_s'];it=iter(dl)
 cfg=dict(ck['config']);cfg['composition_active_mask']=(std>1e-6).tolist();cfg['train_std_wt_percent']=std.tolist();cfg['frozen_head_args']=vars(a)
 (out/'run_config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf8')
 with (out/'metrics.csv').open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for step in range(1,a.steps+1):
   try: batch=next(it)
   except StopIteration:it=iter(dl);batch=next(it)
   head.train();opt.zero_grad();loss,mae=calc(batch,True)
   if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
   loss.backward();opt.step();val_loss=val_mae=''
   if step%a.eval_every==0 or step==a.steps:
    head.eval();v=[]
    with torch.no_grad():
     for b in vl:
      l,m=calc(b);v.append((l.item(),m.item()))
    val_loss,val_mae=np.mean(v,axis=0).tolist()
   if step==1 or step%10==0 or val_loss!='':
    row={'step':step,'train_loss':loss.item(),'train_mae':mae.item(),'val_loss':val_loss,'val_mae':val_mae,'elapsed_s':round(time.monotonic()-start,2)};w.writerow(row);f.flush();print(json.dumps(row),flush=True)
 state={'vae':vae.state_dict(),'composition_head':head.state_dict(),'config':cfg,'source_vae_sha256':hashlib.sha256(Path(a.vae_checkpoint).read_bytes()).hexdigest(),'step':a.steps};torch.save(state,out/'frozen_head.pt');print('Finished. Test split not loaded.',flush=True)
if __name__=='__main__':main()
