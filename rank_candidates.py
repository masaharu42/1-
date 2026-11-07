"""Rank an existing candidate table by two explicitly supplied prediction columns.
This is Pareto filtering, not optimization, reward training, or validation.
"""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def nondominated(x):
 keep=np.ones(len(x),dtype=bool)
 for i in range(len(x)):
  keep[i]=not np.any(np.all(x>=x[i],axis=1)&np.any(x>x[i],axis=1))
 return keep

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidates',required=True);p.add_argument('--objective-x',required=True);p.add_argument('--objective-y',required=True);p.add_argument('--direction-x',choices=['max','min'],default='max');p.add_argument('--direction-y',choices=['max','min'],default='max');p.add_argument('--output',required=True);a=p.parse_args()
 if a.objective_x==a.objective_y:raise ValueError('Objectives must be distinct')
 with Path(a.candidates).open(encoding='utf-8-sig',newline='') as f:
  rd=csv.DictReader(f);data=list(rd);fields=rd.fieldnames
 if not data or not fields or not {'candidate_id',a.objective_x,a.objective_y}.issubset(fields):raise ValueError('CSV needs candidate_id and both explicit objective columns')
 if len({r['candidate_id'] for r in data})!=len(data):raise ValueError('candidate_id must be unique')
 x=np.array([[float(r[a.objective_x]),float(r[a.objective_y])] for r in data],float)
 if not np.isfinite(x).all():raise ValueError('Candidate objectives must be finite')
 transformed=x*np.array([1 if a.direction_x=='max' else -1,1 if a.direction_y=='max' else -1]);front=nondominated(transformed)
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()):raise FileExistsError('Choose an empty output directory')
 with (out/'pareto_candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields+['pareto_front']);w.writeheader()
  for r,k in zip(data,front):w.writerow({**r,'pareto_front':bool(k)})
 fig,ax=plt.subplots(figsize=(7,5));ax.scatter(x[~front,0],x[~front,1],c='#a6a6a6',label='Other');ax.scatter(x[front,0],x[front,1],c='#d45246',label='Pareto front');ax.set(xlabel=a.objective_x,ylabel=a.objective_y,title='Predicted candidate objectives (not validation)');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'pareto_candidates.png',dpi=160);plt.close(fig)
 (out/'config.json').write_text(json.dumps({**vars(a),'warning':'Scores must be predictions available at design time; no experimental verification, uncertainty or validity is inferred.'},indent=2,ensure_ascii=False),encoding='utf8');print(f'{front.sum()} of {len(front)} supplied candidates are nondominated. This is not validation.')
if __name__=='__main__':main()
