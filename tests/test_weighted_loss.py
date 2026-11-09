import unittest
import torch
from train_vae import composition_loss

class WeightedLossTest(unittest.TestCase):
 def test_uniform_matches_original(self):
  p=torch.arange(18,dtype=torch.float32).reshape(2,9);t=torch.zeros_like(p);s=torch.arange(1,10,dtype=torch.float32);m=torch.tensor([1,1,1,1,0,1,1,1,1],dtype=torch.bool)
  self.assertTrue(torch.allclose(composition_loss(p,t,s,m,'uniform',.7),(((p-t)/s)[:,m].square()).mean()))
 def test_group_weights_and_gradients(self):
  p=torch.zeros((1,9),requires_grad=True);t=torch.ones_like(p);s=torch.ones(9);m=torch.ones(9,dtype=torch.bool)
  loss=composition_loss(p,t,s,m,'main3',.7);self.assertAlmostEqual(loss.item(),1.0);loss.backward()
  self.assertAlmostEqual(abs(p.grad[0,0].item()),2*.7/3,places=6)
  self.assertAlmostEqual(abs(p.grad[0,3].item()),2*.3/6,places=6)
 def test_absent_main_element_rejected(self):
  p=torch.zeros((1,9));m=torch.ones(9,dtype=torch.bool);m[2]=False
  with self.assertRaises(ValueError):composition_loss(p,p,torch.ones(9),m,'main3',.7)
 def test_equal_main_error_has_equal_contribution_with_shared_scale(self):
  mask=torch.ones(9,dtype=torch.bool);scale=torch.tensor([5.,5.,5.,1.,1.,1.,1.,1.,1.]);target=torch.zeros((1,9))
  losses=[]
  for index in range(3):
   pred=torch.zeros((1,9));pred[0,index]=1
   losses.append(composition_loss(pred,target,scale,mask,'main3',.7).item())
  self.assertAlmostEqual(max(losses),min(losses),places=7)
