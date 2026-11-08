import csv,json,subprocess,sys,tempfile,unittest
from pathlib import Path

class CompositionComparisonTest(unittest.TestCase):
 def test_test_labels_do_not_affect_validation_and_nine_outputs(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);split={'train':[1,2,3],'val':[4],'test':[5]};(root/'split.json').write_text(json.dumps(split))
   elements=['Ni','Fe','Cr','Nb','Ta','Co','Mo','Al','Ti']
   def run(test_value,out,test=False):
    records=[]
    for i in range(1,6):
     part=next(k for k,v in split.items() if i in v);v=test_value if i==5 else i
     records.append({'alloy_id':i,'partition':part,'latent':[i*.2,i*.3],'composition':[float(v+j) for j in range(9)]})
    f=root/'features.json';f.write_text(json.dumps({'split':split,'composition_elements':elements,'records':records}))
    cmd=[sys.executable,'compare_composition.py','--features',str(f),'--split',str(root/'split.json'),'--output',str(out)]
    if test:cmd.append('--evaluate-test')
    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
    with (out/'九种元素成分比较.csv').open(encoding='utf-8-sig') as fp:return list(csv.DictReader(fp))
   a=run(1,root/'a');b=run(999,root/'b');self.assertEqual(a,b);self.assertEqual([r['元素'] for r in a],elements)
   c=run(5,root/'c',True);self.assertEqual(len(c),9);self.assertTrue(all(r['合金ID']=='5' for r in c))
