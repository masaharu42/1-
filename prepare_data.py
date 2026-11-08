"""Prepare self-contained, audited raw Euler+GB images and alloy-level splits."""
from __future__ import annotations
import argparse,csv,hashlib,json,re,shutil
from pathlib import Path

ELEMENTS=('Ni','Fe','Cr','Nb','Ta','Co','Mo','Al','Ti')
PROJECT=Path(__file__).resolve().parent
WORKSPACE=PROJECT.parent
DEFAULT_IMAGES=WORKSPACE/'data'/'原始导出数据'/'Euler+gb'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ids(text):return [int(x.strip()) for x in text.split(',') if x.strip()]
def main():
 p=argparse.ArgumentParser(description='Copy original Euler+GB BMPs and create explicitly alloy-level split.')
 p.add_argument('--images-dir',type=Path,default=DEFAULT_IMAGES);p.add_argument('--compositions',type=Path,required=True);p.add_argument('--output',type=Path,default=PROJECT/'data');p.add_argument('--val-ids',default='14');p.add_argument('--test-ids',default='19');p.add_argument('--overwrite',action='store_true',help='Replace prepared data files; source data are never modified.')
 a=p.parse_args();out=a.output.resolve(); manifest=out/'manifest.json'
 if manifest.exists() and not a.overwrite: raise FileExistsError(f'{manifest} exists; use --overwrite only if intended')
 with a.compositions.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 compositions={int(x['alloy_id']):x for x in rows}
 images={}
 for file in a.images_dir.glob('*-GB + AE.bmp'):
  m=re.fullmatch(r'(\d+)(?:-\d+)?-GB \+ AE\.bmp',file.name)
  if not m:continue
  i=int(m.group(1))
  if i in images:raise ValueError(f'Duplicate image for ID {i}')
  images[i]=file
 expected=set(range(1,23))
 if set(images)!=expected or not expected.issubset(compositions): raise ValueError(f'Expected IDs 1–22. Image IDs: {sorted(images)}; composition IDs: {sorted(compositions)}')
 vals,tests=ids(a.val_ids),ids(a.test_ids)
 if not vals or not tests or len(vals)!=len(set(vals)) or len(tests)!=len(set(tests)) or set(vals)&set(tests) or not (set(vals)|set(tests))<=expected: raise ValueError('Validation and test IDs must be nonempty, unique, disjoint and within 1–22.')
 out.mkdir(parents=True,exist_ok=True);image_out=out/'images';image_out.mkdir(exist_ok=True)
 records=[]
 for i in sorted(expected):
  src=images[i];target=image_out/f'{i:02d}.bmp';shutil.copy2(src,target)
  comp=compositions[i];v=[float(comp[x]) for x in ELEMENTS]
  if any(x<0 for x in v) or abs(sum(v)-100)>0.01:raise ValueError(f'Invalid nominal composition in ID {i}: {v}')
  records.append({'alloy_id':i,'image':f'images/{i:02d}.bmp','composition':v,'composition_status':comp.get('composition_status','unknown'),'source_note':comp.get('source_note',''),'source_image_name':src.name,'sha256':sha(target)})
 split={'train':sorted(expected-set(vals)-set(tests)),'val':sorted(vals),'test':sorted(tests)}
 info={'version':1,'image_type':'original Euler+GB BMP','elements':list(ELEMENTS),'composition_units':'wt.%','alloy_22_note':'GH706 标准范围中点，为建模标签而非确认试样实测成分','records':records}
 manifest.write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(out/'split.json').write_text(json.dumps(split,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');shutil.copy2(a.compositions,out/'compositions_source.csv')
 print(f'Prepared {len(records)} original images. Train={split["train"]}; val={split["val"]}; test={split["test"]}.')
 print('ID22 is included but its composition is a nominal standard-range midpoint, not a confirmed measurement.')
if __name__=='__main__':main()
