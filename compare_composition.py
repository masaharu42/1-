"""Compare train-mean and ridge on frozen VAE features; optionally include head predictions."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import ELEMENTS,load_split

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--features',required=True);p.add_argument('--split',default='data/split.json');p.add_argument('--output',required=True);p.add_argument('--alpha',type=float,default=10.0);p.add_argument('--evaluate-test',action='store_true');p.add_argument('--head-predictions',help='predictions.csv from evaluate_vae on the same partition and VAE');a=p.parse_args()
 if a.alpha<=0:raise ValueError('alpha must be positive')
 sp=load_split(a.split);payload=json.loads(Path(a.features).read_text(encoding='utf-8'))
 if payload['split']!=sp or tuple(payload['composition_elements'])!=ELEMENTS:raise ValueError('Split or element order mismatch')
 records=payload['records'];ids=[int(r['alloy_id']) for r in records]
 if len(set(ids))!=len(ids) or set(ids)!=set(sum(sp.values(),[])):raise ValueError('Feature IDs mismatch')
 for r in records:
  if int(r['alloy_id']) not in sp[r['partition']]:raise ValueError('Feature partition mismatch')
 train=[r for r in records if r['partition']=='train'];part='test' if a.evaluate_test else 'val';target=[r for r in records if r['partition']==part]
 x=np.asarray([r['latent'] for r in train],float);y=np.asarray([r['composition'] for r in train],float);xt=np.asarray([r['latent'] for r in target],float)
 if y.shape[1]!=len(ELEMENTS) or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(xt).all():raise ValueError('Invalid features or labels')
 mu=x.mean(0);sd=x.std(0);sd[sd<1e-8]=1;xs=(x-mu)/sd;ts=(xt-mu)/sd;ym=y.mean(0);coef=np.linalg.solve(xs.T@xs+a.alpha*np.eye(xs.shape[1]),xs.T@(y-ym));ridge=ym+ts@coef;active=y.std(0)>1e-6
 head={}
 if a.head_predictions:
  config=json.loads((Path(a.head_predictions).parent/'evaluation_config.json').read_text(encoding='utf-8'))
  if config['source_vae_sha256']!=payload['vae_sha256'] or config['windows']!=payload['windows'] or config['partition']!=part:raise ValueError('Head must use the same VAE, evaluation window count and partition as features')
  with open(a.head_predictions,encoding='utf-8-sig',newline='') as f:
   for r in csv.DictReader(f):head[(int(r['alloy_id']),r['element'])]=r
  expected={(int(r['alloy_id']),e) for r in target for e in ELEMENTS}
  if set(head)!=expected:raise ValueError('Head predictions do not match selected partition and elements')
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()):raise FileExistsError('Use a new empty output directory')
 rows=[]
 for i,r in enumerate(target):
  for j,e in enumerate(ELEMENTS):
   actual=float(r['composition'][j]);h=head.get((int(r['alloy_id']),e));hp=float(h['predicted_wt_pct']) if h else None
   if h and abs(float(h['recorded_wt_pct'])-actual)>1e-4:raise ValueError('Head and feature labels differ')
   rows.append({'合金ID':r['alloy_id'],'元素':e,'记录成分_wt%':actual,'训练均值_wt%':ym[j],'岭回归_wt%':ridge[i,j],'岭回归绝对误差_百分点':abs(ridge[i,j]-actual),'神经网络_wt%':hp if hp is not None else '', '神经网络绝对误差_百分点':abs(hp-actual) if hp is not None else '', '训练集该元素有变化':bool(active[j])})
   values=[actual,ym[j],ridge[i,j]]+([hp] if hp is not None else []);names=['Recorded','Train mean','Ridge']+(['Neural head'] if hp is not None else [])
   fig,ax=plt.subplots(figsize=(6,4));ax.bar(names,values,color=['#555','#999','#3288bd','#e69532'][:len(values)]);ax.set_ylabel('Composition (wt.%)');ax.set_title(f'ID{int(r["alloy_id"]):02d} {e} ({part})');ax.axhline(0,color='black',linewidth=.5);fig.tight_layout();fig.savefig(out/f'ID{int(r["alloy_id"]):02d}_{e}_成分比较.png',dpi=160);plt.close(fig)
 with (out/'九种元素成分比较.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 meta={'split':sp,'partition':part,'alpha':a.alpha,'feature_file_sha256':hashlib.sha256(Path(a.features).read_bytes()).hexdigest(),'head_csv':a.head_predictions,'warning':'Ridge unconstrained; values may be negative or not sum to 100. Single test alloy is not overall accuracy.'}
 (out/'比较配置.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8');print(f'完成 {part}：{len(target)} 个合金，九种元素，保存至 {out}',flush=True)
if __name__=='__main__':main()
