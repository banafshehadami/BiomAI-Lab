import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import AttentionGate3D, TemporalSwinBlock

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)

class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # align dims (pad if off by one)
        diffD = skip.size(2) - x.size(2)
        diffH = skip.size(3) - x.size(3)
        diffW = skip.size(4) - x.size(4)
        x = F.pad(x, [diffW//2, diffW - diffW//2, diffH//2, diffH - diffH//2, diffD//2, diffD - diffD//2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class SwinAUNet(nn.Module):
    """3D U-Net + attention gates + temporal Swin-style attention. Input (B,3,T,H,W).
    Output rPPG (B,1,T). Spatial GAP after last conv to map to 1D signal.
    """
    def __init__(self, in_channels=3, base=32, temporal_attn=True, t_heads=4, t_window=16):
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(base, base*2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(base*2, base*4)
        self.pool3 = nn.MaxPool3d((1,2,2))  # preserve time depth more aggressively

        self.bottleneck = ConvBlock3D(base*4, base*8)
        self.temporal = TemporalSwinBlock(base*8, heads=t_heads, window=t_window) if temporal_attn else nn.Identity()

        self.ag3 = AttentionGate3D(in_channels_x=base*4, in_channels_g=base*8, inter_channels=base*4)
        self.up3 = UpBlock3D(base*8, base*4)
        self.ag2 = AttentionGate3D(in_channels_x=base*2, in_channels_g=base*4, inter_channels=base*2)
        self.up2 = UpBlock3D(base*4, base*2)
        self.ag1 = AttentionGate3D(in_channels_x=base, in_channels_g=base*2, inter_channels=base)
        self.up1 = UpBlock3D(base*2, base)

        self.out_conv = nn.Conv3d(base, 1, kernel_size=1)

    def forward(self, x):
        # x: (B,3,T,H,W)
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)
        b = self.temporal(b)

        g3 = self.ag3(e3, b)
        d3 = self.up3(b, g3)
        g2 = self.ag2(e2, d3)
        d2 = self.up2(d3, g2)
        g1 = self.ag1(e1, d2)
        d1 = self.up1(d2, g1)

        y3d = self.out_conv(d1)  # (B,1,T',H',W')
        # upsample back to original spatial size and time
        y3d = F.interpolate(y3d, size=(x.shape[2], x.shape[3], x.shape[4]), mode='trilinear', align_corners=False)
        # spatial GAP → (B,1,T)
        y = y3d.mean(dim=(3,4))  # (B,1,T)
        return y