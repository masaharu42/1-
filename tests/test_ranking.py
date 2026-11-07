import unittest
import numpy as np
from rank_candidates import nondominated
class RankingTest(unittest.TestCase):
 def test_dominance_and_ties(self):
  x=np.array([[1,1],[2,2],[1,3],[2,2],[0,0]],float)
  np.testing.assert_array_equal(nondominated(x),[False,True,True,True,False])
if __name__=='__main__':unittest.main()
