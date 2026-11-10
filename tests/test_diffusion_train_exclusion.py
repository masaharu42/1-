"""Exclude alloys from diffusion without changing the VAE split."""
import unittest
from train_diffusion import select_diffusion_train_ids

class DiffusionExclusionTest(unittest.TestCase):
 def setUp(self):self.split={'train':[1,2,3,15,20],'val':[14],'test':[19]}
 def test_training_only_exclusion_preserves_original_split(self):
  selected=select_diffusion_train_ids(self.split,[1,15,20])
  self.assertEqual(selected,[2,3]);self.assertEqual(self.split['train'],[1,2,3,15,20]);self.assertEqual(self.split['test'],[19])
 def test_invalid_exclusions_are_rejected(self):
  for excluded in ([19],[14],[99],[1,1],[1,2,3,15,20]):
   with self.subTest(excluded=excluded),self.assertRaises(ValueError):select_diffusion_train_ids(self.split,excluded)
 def test_default_retains_original_training_split(self):
  self.assertEqual(select_diffusion_train_ids(self.split,[]),self.split['train'])
if __name__=='__main__':unittest.main()
