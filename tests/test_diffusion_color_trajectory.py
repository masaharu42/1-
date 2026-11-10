import unittest
import torch
from diagnose_diffusion_color_trajectory import color_statistics

class ColorStatisticsTest(unittest.TestCase):
 def test_records_raw_and_clipped_values_separately(self):
  # One of two pixels has two channels outside the display range.
  x=torch.tensor([[[[-2.,0.]],[[2.,0.]],[[0.,0.]]]])
  s=color_statistics(x)
  self.assertAlmostEqual(s['raw_min'],-2.);self.assertAlmostEqual(s['raw_max'],2.)
  self.assertAlmostEqual(s['outside_fraction'],2/6,places=6)
  self.assertAlmostEqual(s['pixels_any_outside_fraction'],.5)
  self.assertAlmostEqual(s['raw_R_255'],0.);self.assertAlmostEqual(s['clipped_R_255'],63.75)
  self.assertAlmostEqual(s['raw_G_255'],255.);self.assertAlmostEqual(s['clipped_G_255'],191.25)
if __name__=='__main__':unittest.main()
