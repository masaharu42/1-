"""Alloy-level dataset. An item is one alloy, not an independent crop."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ELEMENTS=('Ni','Fe','Cr','Nb','Ta','Co','Mo','Al','Ti')

def load_split(split_path, fold_key=None):
 raw=json.loads(Path(split_path).read_text(encoding='utf-8-sig'))
 if fold_key is not None:raw=raw[fold_key]
 result={k:[int(i) for i in raw[k]] for k in ('train','val','test')}
 sets=[set(result[k]) for k in result]
 if any(not x for x in sets): raise ValueError('train, val, test must be non-empty')
 if any(len(result[k])!=len(set(result[k])) for k in result) or sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]: raise ValueError('Split IDs overlap or contain duplicates')
 return result

class AlloyDataset(Dataset):
 def __init__(self,manifest_path,ids,patch_size=256,windows_per_alloy=4,mode='train',seed=42,train_flip=False,train_rotate=False):
  self.path=Path(manifest_path).resolve(); info=json.loads(self.path.read_text(encoding='utf-8-sig'))
  if tuple(info['elements'])!=ELEMENTS:raise ValueError('Element order mismatch')
  self.patch_size=int(patch_size);self.windows_per_alloy=int(windows_per_alloy);self.mode=mode;self.seed=int(seed);self.train_flip=bool(train_flip);self.train_rotate=bool(train_rotate)
  if mode not in ('train','eval','val','test'):raise ValueError('mode must be train, eval, val, or test')
  if self.patch_size<=0 or self.windows_per_alloy<=0:raise ValueError('patch_size/windows_per_alloy must be positive')
  mapping={int(x['alloy_id']):x for x in info['records']};idlist=[int(i) for i in ids]
  if len(set(idlist))!=len(idlist) or any(i not in mapping for i in idlist):raise ValueError('Unknown or duplicate alloy IDs')
  self.records=[mapping[i] for i in idlist];self.compositions=np.asarray([x['composition'] for x in self.records],dtype=np.float32)
  if self.compositions.shape!=(len(idlist),len(ELEMENTS)):raise ValueError('Invalid composition vector')
 def __len__(self):return len(self.records)
 def __getitem__(self,index):
  rec=self.records[index];path=(self.path.parent/rec['image']).resolve()
  if not path.is_relative_to(self.path.parent):raise ValueError('Image path escapes dataset directory')
  with Image.open(path) as im: arr=np.asarray(im.convert('RGB'),dtype=np.uint8).copy()
  h,w=arr.shape[:2];s=self.patch_size
  if h<s or w<s:raise ValueError(f'Image {path} ({w}×{h}) is smaller than {s}×{s}; no automatic resizing')
  # Train uses worker-seeded torch RNG; eval uses fixed deterministic grid, no labels/other IDs.
  patches=[]; k=self.windows_per_alloy
  if self.mode=='train':
   coords=[(int(torch.randint(h-s+1,(1,)).item()),int(torch.randint(w-s+1,(1,)).item())) for _ in range(k)]
  else:
   side=math.ceil(math.sqrt(k));ys=np.rint(np.linspace(0,h-s,side)).astype(int);xs=np.rint(np.linspace(0,w-s,side)).astype(int)
   grid=[(int(y),int(x)) for y in ys for x in xs]; chosen=np.rint(np.linspace(0,len(grid)-1,k)).astype(int);coords=[grid[j] for j in chosen]
  for y,x in coords:
   patch=arr[y:y+s,x:x+s]
   if self.mode=='train' and self.train_flip:
    if torch.rand(()).item()<0.5: patch=patch[:,::-1]
    if torch.rand(()).item()<0.5: patch=patch[::-1,:]
   if self.mode=='train' and self.train_rotate:
    patch=np.rot90(patch,k=int(torch.randint(4,(1,)).item()),axes=(0,1))
   patch=np.ascontiguousarray(patch);patches.append(torch.from_numpy(patch).permute(2,0,1).float().div_(127.5).sub_(1))
  return {'images':torch.stack(patches),'composition':torch.from_numpy(self.compositions[index].copy()),'alloy_id':int(rec['alloy_id'])}
