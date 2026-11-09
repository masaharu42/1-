import unittest
import torch
from train_diffusion import corrupt_condition

class DiffusionAugmentationTest(unittest.TestCase):
 def test_disabled_and_clean_target_unchanged(self):
  source=torch.linspace(-1,1,3*32*32).reshape(1,3,32,32);original=source.clone()
  self.assertTrue(torch.equal(corrupt_condition(source,'none',1),source))
  for kind in ('blur','mask','affine','erode','dilate','mixed'):
   output=corrupt_condition(source,kind,1)
   self.assertEqual(output.shape,source.shape)
   self.assertTrue(torch.isfinite(output).all())
   self.assertTrue(torch.equal(source,original))

 def test_zero_probability_preserves_condition(self):
  source=torch.randn(2,3,32,32)
  self.assertTrue(torch.equal(source,corrupt_condition(source,'mask',0)))
