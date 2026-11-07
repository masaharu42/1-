"""Models for the scratch-trained Euler+GB VAE pilot."""
import torch
from torch import nn
from diffusers import AutoencoderKL


def build_vae():
    return AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=('DownEncoderBlock2D',)*3,
        up_block_types=('UpDecoderBlock2D',)*3,
        block_out_channels=(64,128,256), layers_per_block=2,
        latent_channels=4, norm_num_groups=32, sample_size=256,
    )


class CompositionHead(nn.Module):
    """Equal-weight pooling of K windows into one per-alloy composition."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(16,128),nn.SiLU(),nn.Linear(128,9))

    def forward(self, z):
        # z [B,K,4,H,W]; deterministic posterior mean for stable estimates
        window_mean=z.mean(dim=(-1,-2))
        window_std=z.float().std(dim=(-1,-2),unbiased=False).to(z.dtype)
        window=torch.cat((window_mean,window_std),dim=-1)
        pooled=torch.cat((window.mean(dim=1),window.float().std(dim=1,unbiased=False).to(z.dtype)),dim=-1)
        # Joint non-negative wt.% prediction sums to 100, including nine elements.
        return torch.softmax(self.net(pooled).float(),dim=-1)*100.0


def sobel_magnitude(x):
    """Non-normalised channelwise Sobel magnitude, matching historic baseline."""
    kx=x.new_tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).reshape(1,1,3,3)
    ky=kx.transpose(-1,-2)
    x=x.float(); kx=kx.float(); ky=ky.float(); channels=x.shape[1]
    gx=nn.functional.conv2d(x,kx.repeat(channels,1,1,1),padding=1,groups=channels)
    gy=nn.functional.conv2d(x,ky.repeat(channels,1,1,1),padding=1,groups=channels)
    return (gx.square()+gy.square()+1e-12).sqrt()
