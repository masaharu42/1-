"""Train native 256px vector VAE on alloy-separated crops."""
import argparse,csv,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import AlloyDataset,ELEMENTS,load_split
from vector_vae_256 import VectorVAE256


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def corrupt(images,prob):
    result=images.clone()
    for i in range(len(result)):
        if torch.rand((),device=result.device).item()>=prob:continue
        if torch.rand((),device=result.device).item()<.5:
            result[i:i+1]=F.avg_pool2d(result[i:i+1],3,1,1)
        else:
            size=max(2,int(images.shape[-1]*.08)); y=int(torch.randint(images.shape[-2]-size+1,(),device=result.device));x=int(torch.randint(images.shape[-1]-size+1,(),device=result.device));result[i,:,y:y+size,x:x+size]=0
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='data/manifest.json');p.add_argument('--split',default='data/split.json');p.add_argument('--output',required=True)
    p.add_argument('--latent-dim',type=int,choices=[128,256],default=256);p.add_argument('--steps',type=int,default=5000);p.add_argument('--alloys-per-batch',type=int,default=4);p.add_argument('--windows',type=int,default=4);p.add_argument('--eval-windows',type=int,default=16);p.add_argument('--eval-batch-windows',type=int,default=4);p.add_argument('--grad-accum',type=int,default=1)
    p.add_argument('--eval-every',type=int,default=250);p.add_argument('--save-every',type=int,default=500);p.add_argument('--log-every',type=int,default=10);p.add_argument('--lr',type=float,default=.0002);p.add_argument('--kl-weight',type=float,default=.000001);p.add_argument('--composition-weight',type=float,default=.0001);p.add_argument('--main-elements-weight',type=float,default=.7);p.add_argument('--main-elements-scale',type=float,default=0.)
    p.add_argument('--train-flip',action='store_true');p.add_argument('--train-rotate',action='store_true');p.add_argument('--corruption-prob',type=float,default=0.);p.add_argument('--precision',choices=['fp32','fp16','bf16'],default='fp16');p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=42);p.add_argument('--resume',default=None)
    a=p.parse_args()
    if min(a.steps,a.alloys_per_batch,a.windows,a.eval_windows,a.eval_batch_windows,a.grad_accum,a.eval_every,a.save_every,a.log_every)<1:raise ValueError('Counts must be positive')
    if min(a.lr,a.kl_weight,a.composition_weight,a.main_elements_scale)<0 or not 0<a.main_elements_weight<1 or not 0<=a.corruption_prob<=1:raise ValueError('Invalid loss or augmentation setting')
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if device.type=='cpu' and a.precision!='fp32':raise ValueError('CPU requires --precision fp32')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    split=load_split(a.split);train=AlloyDataset(a.manifest,split['train'],256,a.windows,'train',a.seed,a.train_flip,a.train_rotate);val=AlloyDataset(a.manifest,split['val'],256,a.eval_windows,'eval',a.seed)
    if len(train)<a.alloys_per_batch:raise ValueError('Batch exceeds training alloys')
    root=Path(a.manifest).resolve().parent
    for rec in train.records+val.records:
        path=(root/rec['image']).resolve()
        if not path.is_relative_to(root) or sha(path)!=rec['sha256']:raise ValueError('Image hash/path mismatch')
    std=train.compositions.std(axis=0);active=std>1e-6
    if not active[:3].all() or not active[3:].any():raise ValueError('Main and other elements must vary')
    scale=np.maximum(std,.1)
    if a.main_elements_scale:scale[:3]=a.main_elements_scale
    scale_t=torch.as_tensor(scale,device=device);active_t=torch.as_tensor(active,device=device)
    config={'model_arch':'vector_vae_256','image_size':256,'latent_dim':a.latent_dim,'args':vars(a).copy(),**{k+'_ids':split[k] for k in split},'manifest_sha256':sha(a.manifest),'split_sha256':sha(a.split),'composition_elements':list(ELEMENTS),'train_std_wt_percent':std.tolist(),'composition_active_mask':active.tolist(),'resize':'none'}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);metrics=out/'metrics.csv';model=VectorVAE256(a.latent_dim).to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    amp=a.precision!='fp32' and device.type=='cuda';dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[a.precision];scaler=torch.amp.GradScaler('cuda',enabled=amp and a.precision=='fp16')
    fields=['step','phase','total','reconstruction_l1','kl','composition_loss','Ni_mae_wt','Fe_mae_wt','Cr_mae_wt','elapsed_seconds','eta_seconds','gpu_peak_gb'];step=0;previous_elapsed=0.
    if a.resume:
        cp=Path(a.resume).resolve()
        if cp.parent!=out.resolve() or not metrics.exists() or not (out/'run_config.json').exists():raise ValueError('Resume requires checkpoint in existing output directory')
        state=torch.load(cp,map_location='cpu',weights_only=False);old=state['config']
        for key in ['model_arch','image_size','latent_dim','train_ids','val_ids','test_ids','manifest_sha256','split_sha256']:
            if old[key]!=config[key]:raise ValueError('Resume mismatch: '+key)
        for key,value in vars(a).items():
            if key not in ('steps','resume') and old['args'].get(key)!=value:raise ValueError('Resume argument mismatch: '+key)
        step=int(state['step']);previous_elapsed=float(state['elapsed_seconds'])
        if step>=a.steps:raise ValueError('steps must exceed checkpoint step')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
        config=old
        print('Resume restores optimizer and RNG; DataLoader crop order can differ.',flush=True)
    else:
        if any(out.iterdir()):raise FileExistsError('Use a new empty output directory')
        (out/'run_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
        with metrics.open('w',newline='',encoding='utf-8') as f:csv.DictWriter(f,fieldnames=fields).writeheader()
    train_loader=DataLoader(train,batch_size=a.alloys_per_batch,shuffle=True);val_loader=DataLoader(val,batch_size=1,shuffle=False);iterator=iter(train_loader);start=time.monotonic();first_step=step
    def calc(batch,training):
        images=batch['images'].to(device);targets=batch['composition'].to(device);b,k=images.shape[:2];flat=images.reshape(b*k,3,256,256);input_images=corrupt(flat,a.corruption_prob) if training and a.corruption_prob else flat
        with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp):
            if training:
                means,logvar=model.encode(input_images);z=means+torch.randn_like(means)*torch.exp(.5*logvar);recon=model.decode(z);rec=F.l1_loss(recon.float(),flat.float())
            else:
                means_parts=[];logs=[];losses=[]
                for chunk in input_images.split(a.eval_batch_windows):
                    mu,lv=model.encode(chunk);reconstruction=model.decode(mu);means_parts.append(mu);logs.append(lv);losses.append((F.l1_loss(reconstruction.float(),chunk.float(),reduction='sum'),chunk.numel()))
                means=torch.cat(means_parts);logvar=torch.cat(logs);rec=sum(x[0] for x in losses)/sum(x[1] for x in losses)
            kl=(-.5*(1+logvar.float()-means.float().square()-logvar.float().exp())).sum(dim=1).mean();pred=model.composition(means.reshape(b,k,a.latent_dim));err=((pred-targets)/scale_t).square();comp=a.main_elements_weight*err[:,:3].mean()+(1-a.main_elements_weight)*err[:,3:][:,active_t[3:]].mean();total=rec+a.kl_weight*kl+a.composition_weight*comp;mae=(pred-targets).abs().mean(dim=0)
        return total,rec,kl,comp,*mae[:3]
    def record(phase,values):
        elapsed=previous_elapsed+time.monotonic()-start;eta=(a.steps-step)*(time.monotonic()-start)/max(1,step-first_step);nums=[float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for v in values];row=dict(zip(fields,[step,phase]+nums+[round(elapsed,1),round(eta,1),round(torch.cuda.max_memory_allocated()/1024**3,3) if device.type=='cuda' else 0]))
        with metrics.open('a',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerow(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    while step<a.steps:
        model.train();opt.zero_grad(set_to_none=True);acc=[]
        for _ in range(a.grad_accum):
            try:batch=next(iterator)
            except StopIteration:iterator=iter(train_loader);batch=next(iterator)
            values=calc(batch,True)
            if not torch.isfinite(values[0]):raise RuntimeError('Non-finite loss')
            scaler.scale(values[0]/a.grad_accum).backward();acc.append([float(v.detach()) for v in values])
        scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update();step+=1
        if step==1 or step%a.log_every==0:record('train',np.mean(acc,axis=0))
        if step%a.eval_every==0 or step==a.steps:
            model.eval()
            with torch.no_grad(): vals=[[float(x) for x in calc(item,False)] for item in val_loader]
            record('val',np.mean(vals,axis=0))
        if step%a.save_every==0 or step==a.steps:
            state={'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'config':config,'step':step,'elapsed_seconds':previous_elapsed+time.monotonic()-start,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
            if device.type=='cuda':state['cuda_rng']=torch.cuda.get_rng_state_all()
            path=out/f'checkpoint_{step:07d}.pt'
            if path.exists():raise FileExistsError('Checkpoint exists; refusing overwrite')
            tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)
    print('Finished; test images were not loaded.',flush=True)
if __name__=='__main__':main()
