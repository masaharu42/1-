"""Fit simple train-only ridge models. Targets must be explicitly provided and reviewed."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import load_split

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',required=True);p.add_argument('--targets',required=True,help='CSV with alloy_id and explicitly selected numeric target column');p.add_argument('--target-column',required=True);p.add_argument('--split',default='data/split.json');p.add_argument('--output',required=True);p.add_argument('--alpha',type=float,default=10);p.add_argument('--exclude-train-ids',default='',help='Comma-separated training alloy IDs to omit from supervised fitting; VAE may still have seen their images');p.add_argument('--evaluate-test',action='store_true',help='Run once after fixing the method and parameters; no selection on test')
 a=p.parse_args()
 if not math.isfinite(a.alpha) or a.alpha<=0:raise ValueError('alpha must be finite and positive')
 sp=load_split(a.split);exclude={int(x.strip()) for x in a.exclude_train_ids.split(',') if x.strip()};
 if not exclude.issubset(sp['train']):raise ValueError('Excluded IDs must be training IDs')
 train_ids=[i for i in sp['train'] if i not in exclude]
 if len(train_ids)<3:raise ValueError('At least three training alloys required')
 payload=json.loads(Path(a.features).read_text(encoding='utf8'))
 if payload.get('split')!=sp:raise ValueError('features/split mismatch')
 records=payload['records'];ids=[int(v['alloy_id']) for v in records]
 if len(ids)!=len(set(ids)) or set(ids)!=set(sum(sp.values(),[])):raise ValueError('feature IDs do not match split')
 for v in records:
  if v['partition'] not in sp or int(v['alloy_id']) not in sp[v['partition']]:raise ValueError('feature partition mismatch')
 byid={int(v['alloy_id']):v for v in records};needed=train_ids+sp['val']+(sp['test'] if a.evaluate_test else []);labels={}
 with open(a.targets,encoding='utf-8-sig',newline='') as f:
  reader=csv.DictReader(f)
  if not reader.fieldnames or 'alloy_id' not in reader.fieldnames or a.target_column not in reader.fieldnames:raise ValueError('Target CSV requires alloy_id and selected target column')
  seen=set()
  for row in reader:
   i=int(row['alloy_id'])
   if i in seen:raise ValueError(f'Duplicate alloy ID {i} in target table')
   seen.add(i)
   if i not in needed:continue
   try:value=float(row[a.target_column])
   except (TypeError,ValueError):raise ValueError(f'Missing/non-numeric target for alloy {i}')
   if not math.isfinite(value) or value<0:raise ValueError(f'Nonfinite or negative target for alloy {i}')
   labels[i]=value
 if set(needed)-labels.keys():raise ValueError(f'Missing target IDs {sorted(set(needed)-labels.keys())}')
 out=Path(a.output)
 if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a new output directory')
 out.mkdir(parents=True,exist_ok=True)
 rows=[];models={};names={'latent':['latent'],'composition':['composition'],'combined':['latent','composition']}
 for mode,groups in names.items():
  def X(ids):return np.array([sum((byid[i][g] for g in groups),[]) for i in ids],dtype=float)
  x=X(train_ids);y=np.array([labels[i] for i in train_ids]);mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-8]=1;x=(x-mean)/scale;ymean=y.mean();coef=np.linalg.solve(x.T@x+a.alpha*np.eye(x.shape[1]),x.T@(y-ymean))
  models[mode]={'features':groups,'x_mean':mean.tolist(),'x_scale':scale.tolist(),'coef':coef.tolist(),'intercept':float(ymean)}
  print(f'{mode}: fitted {len(y)} training alloys; alpha={a.alpha}',flush=True)
  for part in (['val','test'] if a.evaluate_test else ['val']):
   xp=(X(sp[part])-mean)/scale;pred=xp@coef+ymean
   for i,estimate in zip(sp[part],pred):rows.append({'alloy_id':i,'partition':part,'model':mode,'target':a.target_column,'observed':labels[i],'predicted':float(estimate),'abs_error':float(abs(estimate-labels[i]))})
 for part in (['val','test'] if a.evaluate_test else ['val']):
  for i in sp[part]:rows.append({'alloy_id':i,'partition':part,'model':'training_mean','target':a.target_column,'observed':labels[i],'predicted':float(np.mean([labels[j] for j in train_ids])), 'abs_error':float(abs(np.mean([labels[j] for j in train_ids])-labels[i]))})
 with (out/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 for part in (['val','test'] if a.evaluate_test else ['val']):
  fig,ax=plt.subplots(figsize=(6,5))
  for mode in ('training_mean','latent','composition','combined'):
   rr=[v for v in rows if v['model']==mode and v['partition']==part];ax.scatter([v['observed'] for v in rr],[v['predicted'] for v in rr],label=mode)
  v=[v for v in rows if v['partition']==part];low=min(min(r['observed'],r['predicted']) for r in v);high=max(max(r['observed'],r['predicted']) for r in v);pad=max((high-low)*.05,.01);ax.plot([low-pad,high+pad],[low-pad,high+pad],'k--');ax.set(xlabel='Recorded '+a.target_column,ylabel='Predicted '+a.target_column,title=f'{part}: held-out alloys');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/f'{part}_prediction.png',dpi=160);plt.close(fig)
 provenance={'target_column':a.target_column,'target_sha256':sha(a.targets),'feature_sha256':sha(a.features),'split':sp,'supervised_train_ids':train_ids,'excluded_train_ids':sorted(exclude),'alpha':a.alpha,'models':models,'test_evaluated':a.evaluate_test,'warning':'Exploratory surrogate, fixed alpha. Composition and combined branches use recorded composition at evaluation (known-composition oracle), not available from an unknown generated image. Labels and physical meaning require independent review. No candidate optimization or material validation.'};(out/'model.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n',encoding='utf8');print(f'Saved validation predictions. Test evaluated: {a.evaluate_test}. Composition/combined branches use recorded composition and are known-composition comparisons.',flush=True)
if __name__=='__main__':main()
