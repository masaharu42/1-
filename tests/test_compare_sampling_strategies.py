import unittest
from types import SimpleNamespace
import torch
from compare_sampling_strategies import choose_sdedit_timestep,make_scheduler,sample_case

class ZeroUnet:
 def __call__(self,x,t):return SimpleNamespace(sample=torch.zeros_like(x[:,:3]))

class SamplingStrategyTest(unittest.TestCase):
 def test_strength_selects_suffix_and_reports_actual_timestep(self):
  sched=make_scheduler(1000,50,False,torch.device('cpu'))
  self.assertEqual(choose_sdedit_timestep(sched,.5),25)
  self.assertEqual(int(sched.timesteps[25]),480)
  self.assertEqual(choose_sdedit_timestep(sched,1.),0)
  with self.assertRaises(ValueError):choose_sdedit_timestep(sched,0)
 def test_same_noise_baseline_and_intermediate_clip_differ(self):
  device=torch.device('cpu');coarse=torch.zeros(1,3,4,4);noise=torch.ones_like(coarse)*3
  base={'label':'base','steps':5,'clip_x0':False};clip={'label':'clip','steps':5,'clip_x0':True}
  a,meta=sample_case(ZeroUnet(),coarse,noise,10,base,device);b,_=sample_case(ZeroUnet(),coarse,noise,10,clip,device)
  self.assertEqual(meta['actual_denoise_steps'],5);self.assertGreater(float(a.max()),1);self.assertLessEqual(float(b.max()),1)
 def test_sdedit_records_actual_denoising_steps(self):
  device=torch.device('cpu');coarse=torch.ones(1,3,4,4)*.25;noise=torch.zeros_like(coarse)
  result,meta=sample_case(ZeroUnet(),coarse,noise,100,{'label':'sdedit','steps':10,'clip_x0':False,'strength':.5},device)
  self.assertEqual(meta['actual_denoise_steps'],5);self.assertEqual(meta['start_timestep'],40);self.assertTrue(torch.isfinite(result).all())
if __name__=='__main__':unittest.main()

class MultiwindowAndDarkPixelsTest(unittest.TestCase):
 def test_dark_proxy_tolerance_and_empty_case(self):
  from compare_sampling_strategies import dark_line_scores
  target=torch.ones(1,3,6,6);prediction=torch.ones_like(target)
  target[:,:,2,2]=-1;prediction[:,:,2,3]=-1
  scored=dark_line_scores(target,prediction,25,1)
  self.assertEqual(scored['reference_dark_pixels'],1);self.assertEqual(scored['generated_dark_pixels'],1);self.assertEqual(scored['dark_pixel_f1'],1.)
  missed=dark_line_scores(target,prediction,25,0);self.assertEqual(missed['dark_pixel_f1'],0.)
  none=dark_line_scores(torch.ones_like(target),torch.ones_like(target),25,1);self.assertIsNone(none['dark_pixel_f1'])
 def test_grid_coordinates_match_fixed_dataset(self):
  import numpy as np
  from PIL import Image
  from data import AlloyDataset
  from compare_sampling_strategies import eval_grid_positions
  dataset=AlloyDataset('data/manifest.json',[19],256,8,'eval'); item=dataset[0]
  with Image.open('data/images/19.bmp') as image:
   rgb=np.asarray(image.convert('RGB')).copy();coords=eval_grid_positions(image.width,image.height,256,8)
  self.assertEqual(len(set(coords)),8)
  for index,(y,x) in enumerate(coords):
   expected=torch.from_numpy(rgb[y:y+256,x:x+256].copy()).permute(2,0,1).float().div(127.5).sub(1)
   self.assertTrue(torch.equal(item['images'][index],expected))
