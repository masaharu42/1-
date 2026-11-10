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
