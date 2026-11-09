import unittest
import torch
from train_diffusion import noise_losses

class DiffusionLossTest(unittest.TestCase):
 def test_identical_noise_has_zero_loss(self):
  x=torch.randn(2,3,32,32,requires_grad=True)
  pixel,ssim,edge=noise_losses(x,x,'mse')
  self.assertAlmostEqual(pixel.item(),0,places=6)
  self.assertAlmostEqual(ssim.item(),0,places=5)
  self.assertAlmostEqual(edge.item(),0,places=6)
 def test_nonidentical_noise_has_finite_gradient(self):
  x=torch.randn(2,3,32,32,requires_grad=True);y=torch.randn_like(x)
  pixel,ssim,edge=noise_losses(x,y,'l1');(pixel+0.1*ssim+0.1*edge).backward()
  self.assertTrue(torch.isfinite(x.grad).all())
  self.assertGreater(x.grad.abs().sum().item(),0)
