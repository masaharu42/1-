import csv,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path

class PerformanceValidationTest(unittest.TestCase):
 def test_train_and_val_ignore_test_label(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);split={'train':[1,2,3],'val':[4],'test':[5]};sp=root/'split.json';sp.write_text(json.dumps(split));features={'split':split,'records':[{'alloy_id':i,'partition':next(k for k,v in split.items() if i in v),'latent':[i*.1, i*.2],'composition':[i,100-i]} for i in range(1,6)]};fp=root/'features.json';fp.write_text(json.dumps(features));target=root/'targets.csv'
   def run(test_value,out):
    with target.open('w',newline='') as f:
     w=csv.writer(f);w.writerow(['alloy_id','peak']);w.writerows([[i,(test_value if i==5 else i*10)] for i in range(1,6)])
    subprocess.run([sys.executable,'fit_performance.py','--features',str(fp),'--targets',str(target),'--target-column','peak','--split',str(sp),'--output',str(out)],check=True,stdout=subprocess.DEVNULL)
    with (out/'predictions.csv').open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
   a=run(100,root/'a');b=run(999,root/'b');self.assertEqual(a,b);self.assertTrue(all(r['partition']=='val' for r in a))
if __name__=='__main__':unittest.main()

class PerformanceExclusionTest(unittest.TestCase):
 def test_excluded_training_targets_not_required(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);sp={'train':[1,2,3,4],'val':[5],'test':[6]}; (root/'split.json').write_text(json.dumps(sp));(root/'features.json').write_text(json.dumps({'split':sp,'records':[{'alloy_id':i,'partition':next(k for k,v in sp.items() if i in v),'latent':[i*.1,i*.2],'composition':[i,100-i]} for i in range(1,7)]}));
   with (root/'target.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['alloy_id','peak']);w.writerows([[i,i*10] for i in [1,2,3,5]])
   subprocess.run([sys.executable,'fit_performance.py','--features',str(root/'features.json'),'--targets',str(root/'target.csv'),'--target-column','peak','--split',str(root/'split.json'),'--output',str(root/'out'),'--exclude-train-ids','4'],check=True,stdout=subprocess.DEVNULL)
   model=json.loads((root/'out/model.json').read_text());self.assertEqual(model['supervised_train_ids'],[1,2,3]);self.assertEqual(model['excluded_train_ids'],[4])
