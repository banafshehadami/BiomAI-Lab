import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchGAN1D(nn.Module):
    def __init__(self, in_ch=1, nf=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, nf, 15, stride=4, padding=7), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(nf, nf*2, 15, stride=2, padding=7), nn.BatchNorm1d(nf*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(nf*2, nf*4, 15, stride=2, padding=7), nn.BatchNorm1d(nf*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(nf*4, 1, 7, stride=1, padding=3)
        )
    def forward(self, x):
        return self.net(x)  # (B,1,Lp) logits per patch

class PatchGAN2D(nn.Module):
    def __init__(self, in_ch=1, nf=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, nf, 5, stride=2, padding=2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, 5, stride=2, padding=2), nn.BatchNorm2d(nf*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, 5, stride=2, padding=2), nn.BatchNorm2d(nf*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, 1, 3, stride=1, padding=1)
        )
    def forward(self, x):
        return self.net(x)  # (B,1,Hp,Wp)

class TimeDiscriminator(PatchGAN1D):
    pass

class SDDiscriminator(PatchGAN1D):
    pass

class FreqDiscriminator(PatchGAN2D):
    pass