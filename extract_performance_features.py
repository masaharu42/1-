"""Extract alloy-level VAE latent statistics for a strictly held-out performance pilot."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from data import AlloyDataset, load_split, ELEMENTS
from model import build_vae

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',default='data/manifest.json');p.add_argument('--split',default='data/split.json');p.add_argument('--output',required=True);p.add_argument('--windows',type=int,default=16);p.add_argument('--batch-windows',type=int,default=4);p.add_argument('--device',default='cuda');a=p.parse_args()
 if min(a.windows,a.batch_windows)<1:raise ValueError('Window counts must be positive')
 device=torch.device(a.device)
 if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; specify --device cpu')
 split=load_split(a.split);checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 for part in ('train','val','test'):
  if sorted(map(int,checkpoint['config'][part+'_ids']))!=sorted(split[part]):raise ValueError('VAE split mismatch; refuse to use a model that could have seen the test alloy')
 if checkpoint['config'].get('manifest_sha256')!=sha(a.manifest):raise ValueError('Checkpoint/manifest SHA256 mismatch')
 out=Path(a.output)
 if out.exists():raise FileExistsError(f'{out} already exists; choose a new output file')
 out.parent.mkdir(parents=True,exist_ok=True)
 vae=build_vae().to(device);vae.load_state_dict(checkpoint['vae']);vae.eval();rows=[]
 for part in ('train','val','test'):
  ds=AlloyDataset(a.manifest,split[part],256,a.windows,'eval')
  for idx in range(len(ds)):
   record=ds.records[idx];image=Path(a.manifest).resolve().parent/record['image']
   if sha(image)!=record['sha256']:raise ValueError(f'Image SHA256 mismatch ID {record["alloy_id"]}')
   item=ds[idx];x=item['images'];z_list=[]
   with torch.no_grad():
    for chunk in x.split(a.batch_windows):
     z=vae.encode(chunk.to(device)).latent_dist.mode();mu=z.mean(dim=(-1,-2));sd=z.std(dim=(-1,-2),unbiased=False);z_list.append(torch.cat([mu,sd],dim=-1).cpu().numpy())
   z=np.concatenate(z_list);v=np.concatenate([z.mean(axis=0),z.std(axis=0)]).astype(float)
   if not np.isfinite(v).all():raise ValueError('nonfinite latent features')
   rows.append({'alloy_id':int(item['alloy_id']),'partition':part,'latent':v.tolist(),'composition':item['composition'].tolist()});print(f'{part}: encoded alloy {item["alloy_id"]}',flush=True)
 payload={'version':1,'split':split,'windows':a.windows,'manifest_sha256':sha(a.manifest),'vae_sha256':sha(a.checkpoint),'composition_elements':list(ELEMENTS),'latent_description':'channel-wise spatial mean/std, then across-window mean/std; 16 features','records':rows};out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('Saved',out)
if __name__=='__main__':main()
