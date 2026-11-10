"""Shape and compatibility contract for native 256px vector VAEs."""
import unittest
import torch
from vector_vae_256 import VectorVAE256
from train_diffusion import vq

class NativeVectorVAETest(unittest.TestCase):
    def test_three_latent_dimensions_and_diffusion_condition(self):
        # Meta tensors check the actual network shapes without allocating the
        # large 512-dimensional fully connected weights on CI machines.
        for latent_dim in (128, 256, 512):
            with self.subTest(latent_dim=latent_dim), torch.device('meta'):
                model=VectorVAE256(latent_dim).eval()
                images=torch.empty(2,3,256,256)
                means,logvar=model.encode(images)
                self.assertEqual(tuple(means.shape),(2,latent_dim))
                self.assertEqual(tuple(logvar.shape),(2,latent_dim))
                self.assertEqual(tuple(model.decode(means).shape),(2,3,256,256))
                self.assertEqual(tuple(model.composition(means.reshape(1,2,latent_dim)).shape),(1,9))
                self.assertEqual(tuple(vq(model,images).shape),(2,3,256,256))
    def test_unknown_latent_size_rejected(self):
        with self.assertRaises(ValueError):VectorVAE256(17)

if __name__=='__main__':unittest.main()
