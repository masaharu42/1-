"""Evaluate one fixed VAE checkpoint on a chosen alloy partition; no training."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import AlloyDataset,load_split,ELEMENTS
from model import build_vae,CompositionHead

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',default='data/manifest.json');p.add_argument('--split',default='data/split.json');p.add_argument('--partition',choices=['val','test'],required=True);p.add_argument('--output',required=True);p.add_argument('--windows',type=int,default=16);p.add_argument('--device',default='cuda');a=p.parse_args()
 device=torch.device(a.device)
 if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use --device cpu')
 sp=load_split(a.split);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 for k in ['train','val','test']:
  if sorted(map(int,ck['config'][k+'_ids']))!=sorted(sp[k]):raise ValueError('Checkpoint split does not match; evaluation aborted')
 if ck['config'].get('manifest_sha256') != hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest():raise ValueError('Checkpoint and manifest differ')
 o=Path(a.output);o.mkdir(parents=True,exist_ok=True)
 if any(o.iterdir()):raise FileExistsError('Evaluation directory already populated; use a new output path')
 vae=build_vae().to(device);vae.load_state_dict(ck['vae']);vae.eval();head=None
 if ck.get('composition_head') is not None:
  head=CompositionHead().to(device);head.load_state_dict(ck['composition_head']);head.eval()
 ds=AlloyDataset(a.manifest,sp[a.partition],patch_size=256,windows_per_alloy=a.windows,mode='eval');rows=[]
 for rec in ds.records:
  path=Path(a.manifest).resolve().parent/rec['image']
  if hashlib.sha256(path.read_bytes()).hexdigest()!=rec['sha256']:raise ValueError('Evaluation image SHA256 mismatch')
 with torch.no_grad():
  for batch in DataLoader(ds,batch_size=1,shuffle=False,num_workers=0):
   im=batch['images'].to(device);B,K=im.shape[:2];z=vae.encode(im.reshape(B*K,*im.shape[2:])).latent_dist.mode().reshape(B,K,4,64,64)
   recon=vae.decode(z[:,0]).sample.float().clamp(-1,1);original=im[0,0].float().cpu();reconstructed=recon[0].cpu()
   fig,axes=plt.subplots(1,3,figsize=(10,4));images=[(original+1)/2,(reconstructed+1)/2,(original-reconstructed).abs()/2]
   for ax,image,title in zip(axes,images,['Original','VAE reconstruction','Absolute RGB difference']):ax.imshow(image.permute(1,2,0).numpy().clip(0,1));ax.set_title(title);ax.axis('off')
   fig.suptitle(f'ID {int(batch["alloy_id"][0])}: first fixed evaluation window');fig.tight_layout();fig.savefig(o/f'ID{int(batch["alloy_id"][0]):02d}_原图与VAE重建.png',dpi=160);plt.close(fig)
   if head is None:continue
   pred=head(z).cpu().numpy()[0];actual=batch['composition'].numpy()[0]
   for j,e in enumerate(ELEMENTS):rows.append({'alloy_id':int(batch['alloy_id'][0]),'element':e,'recorded_wt_pct':float(actual[j]),'predicted_wt_pct':float(pred[j]),'abs_error_wt_pct':float(abs(pred[j]-actual[j])),'active_in_training':bool(ck['config']['composition_active_mask'][j])})
 if rows:
  with (o/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 for e in (ELEMENTS if rows else []):
  rr=[r for r in rows if r['element']==e];x=np.array([r['recorded_wt_pct'] for r in rr]);y=np.array([r['predicted_wt_pct'] for r in rr]);fig,ax=plt.subplots(figsize=(5,5));ax.scatter(x,y);lo=min(x.min(),y.min());hi=max(x.max(),y.max());pad=max((hi-lo)*.1,.1);ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],'--',color='gray')
  for r in rr:ax.annotate(str(r['alloy_id']),(r['recorded_wt_pct'],r['predicted_wt_pct']),xytext=(4,4),textcoords='offset points')
  ax.set(xlabel='Recorded composition (wt.%)',ylabel='Predicted composition (wt.%)',title=f'{e}: {a.partition}');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(o/f'{e}_记录值与预测值.png',dpi=160);plt.close(fig)
 config=vars(a).copy();config['checkpoint_sha256']=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest();config['source_vae_sha256']=ck.get('source_vae_sha256',config['checkpoint_sha256'])
 (o/'evaluation_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8');print(f'Evaluated {a.partition} IDs {sp[a.partition]}. Test output is not for hyperparameter selection.',flush=True)
 if head is None: print('No composition head in checkpoint; reconstruction only.',flush=True)
if __name__=='__main__':main()
