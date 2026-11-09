"""128-dimensional vector VAE for the Euler+GB comparison."""
import torch
from torch import nn


class VectorVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLU(), nn.Flatten(),
        )
        self.mean = nn.Linear(128 * 32 * 32, 128)
        self.logvar = nn.Linear(128 * 32 * 32, 128)
        self.expand = nn.Linear(128, 128 * 32 * 32)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 32, 32)),
            nn.ConvTranspose2d(128, 64, 6, stride=2, padding=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2, padding=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 5, padding=2), nn.Tanh(),
        )
        self.composition_head = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 9))

    def encode(self, images):
        features = self.encoder(images)
        return self.mean(features), self.logvar(features).clamp(-20, 10)

    def decode(self, latent):
        return self.decoder(self.expand(latent))

    def composition(self, means):
        return torch.softmax(self.composition_head(means.mean(dim=1)), dim=-1) * 100
