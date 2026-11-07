import unittest
import torch
from model import CompositionHead

class ModelTest(unittest.TestCase):
 def test_composition_head_shape_and_closure(self):
  torch.manual_seed(0);head=CompositionHead();y=head(torch.randn(2,3,4,8,8))
  self.assertEqual(tuple(y.shape),(2,9));self.assertTrue(torch.all(y>=0));self.assertTrue(torch.allclose(y.sum(dim=1),torch.tensor([100.,100.]),atol=1e-4))
if __name__=='__main__':unittest.main()
