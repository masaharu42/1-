"""Plot each logged metric separately from a VAE, frozen-head or diffusion CSV."""
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv');p.add_argument('--output',required=True);a=p.parse_args();o=Path(a.output);o.mkdir(parents=True,exist_ok=True)
with open(a.csv,encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
if not rows:raise SystemExit('No metric rows')
exclude={'step','phase','segment','elapsed_seconds','elapsed_sec','elapsed_s','lr','gpu_peak_gb'};names=[x for x in rows[0] if x not in exclude]
for name in names:
 fig,ax=plt.subplots(figsize=(8,4));present=False
 for phase in (['train','val'] if 'phase' in rows[0] else [None]):
  xs=[];ys=[]
  for r in rows:
   if phase is not None and r.get('phase')!=phase:continue
   try: y=float(r[name]);x=int(r['step'])
   except (ValueError,TypeError,KeyError):continue
   if y==y:xs.append(x);ys.append(y)
  if xs:ax.plot(xs,ys,marker='.',label=phase or name);present=True
 if present:
  ax.set(xlabel='Optimizer step',ylabel=name,title=name);ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(o/(name+'.png'),dpi=160)
 plt.close(fig)
print(f'Plots saved to {o}')
