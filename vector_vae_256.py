"""Vector VAE for native 256px crops; separate from historical 128px model."""
import torch
from torch import nn

class VectorVAE256(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        if latent_dim not in (128, 256, 512): raise ValueError('latent_dim must be 128, 256 or 512')
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,64,4,stride=2,padding=1),nn.ReLU(),nn.Conv2d(64,128,4,stride=2,padding=1),nn.ReLU(),nn.Conv2d(128,128,4,stride=2,padding=1),nn.ReLU(),nn.Flatten())
        self.mean=nn.Linear(128*32*32,latent_dim);self.logvar=nn.Linear(128*32*32,latent_dim)
        self.expand=nn.Linear(latent_dim,128*32*32)
        self.decoder=nn.Sequential(nn.Unflatten(1,(128,32,32)),nn.ConvTranspose2d(128,128,4,stride=2,padding=1),nn.ReLU(),nn.ConvTranspose2d(128,64,4,stride=2,padding=1),nn.ReLU(),nn.ConvTranspose2d(64,32,4,stride=2,padding=1),nn.ReLU(),nn.Conv2d(32,3,3,padding=1),nn.Tanh())
        self.composition_head=nn.Sequential(nn.Linear(latent_dim,128),nn.SiLU(),nn.Linear(128,9))
    def encode(self,x):
        features=self.encoder(x)
        return self.mean(features),self.logvar(features).clamp(-20,10)
    def decode(self,z): return self.decoder(self.expand(z))
    def composition(self,means): return torch.softmax(self.composition_head(means.mean(dim=1)).float(),dim=-1)*100
