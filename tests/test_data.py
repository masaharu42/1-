import hashlib,json,tempfile,unittest
from pathlib import Path
from data import AlloyDataset,load_split

class DataProtocolTest(unittest.TestCase):
 def setUp(self):
  self.manifest=Path('data/manifest.json');self.split=load_split('data/split.json')
 def test_all_22_and_labels(self):
  data=json.loads(self.manifest.read_text(encoding='utf8'));r=data['records'];self.assertEqual({i['alloy_id'] for i in r},set(range(1,23)));self.assertIn('midpoint',r[21]['composition_status']);self.assertEqual(sum(map(len,self.split.values())),22)
  for item in r:
   p=self.manifest.parent/item['image'];self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),item['sha256']);self.assertAlmostEqual(sum(item['composition']),100,places=2)
 def test_eval_deterministic_and_shape(self):
  a=AlloyDataset(self.manifest,self.split['test'],patch_size=256,windows_per_alloy=4,mode='eval',train_flip=True,train_rotate=True);x=a[0];y=a[0];self.assertEqual(tuple(x['images'].shape),(4,3,256,256));self.assertTrue(x['images'].equal(y['images']))
 def test_overlap_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'split.json';f.write_text(json.dumps({'train':[1,2],'val':[2],'test':[3]}));
   with self.assertRaises(ValueError):load_split(f)
if __name__=='__main__':unittest.main()
