import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate3D(nn.Module):
    """Attention gate for 3D skip connections (Attention U-Net style)."""
    def __init__(self, in_channels_x, in_channels_g, inter_channels):
        super().__init__()
        self.theta_x = nn.Conv3d(in_channels_x, inter_channels, kernel_size=2, stride=2, bias=False)
        self.phi_g = nn.Conv3d(in_channels_g, inter_channels, kernel_size=1, stride=1, bias=True)
        self.psi = nn.Conv3d(inter_channels, 1, kernel_size=1, stride=1, bias=True)
        self.bn = nn.BatchNorm3d(inter_channels)

    def forward(self, x, g):
        # x: skip feat, g: gating (decoder) feat
        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)
        f = F.relu(self.bn(theta_x + phi_g), inplace=True)
        psi = torch.sigmoid(self.psi(f))
        psi_upsampled = F.interpolate(psi, size=x.shape[2:], mode='trilinear', align_corners=False)
        return x * psi_upsampled

class TemporalSwinBlock(nn.Module):
    """A light 'Swin-like' temporal self-attention block over time windows.
    Operates on tensor of shape (B, C, T, H, W). We pool spatially to (B, C, T),
    apply windowed MHSA with cosine attention, then broadcast back.
    """
    def __init__(self, channels, heads=4, window=16):
        super().__init__()
        self.heads = heads
        self.window = window
        self.norm = nn.LayerNorm(channels)
        self.tau = nn.Parameter(torch.ones(heads))  # trainable tau per head
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )

    def _cosine_attn(self, q, k):
        # q,k: (B, heads, W, C//heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logits = torch.einsum('bhwc,bhsc->bhws', q, k)  # (B, heads, W, S)
        # scale by tau (broadcast per head)
        tau = self.tau.view(1, -1, 1, 1) + 1e-6
        return logits / tau

    def forward(self, x):
        B, C, T, H, W = x.shape
        # global spatial pooling → (B, C, T)
        s = x.mean(dim=(3,4))
        s = s.transpose(1,2)  # (B, T, C)
        s = self.norm(s)
        qkv = self.qkv(s).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.heads, C // self.heads).transpose(1,2) for t in qkv]  # (B, H, T, C/H)

        # process by time windows (non-overlapping for simplicity)
        Wsz = self.window
        pad = (Wsz - (T % Wsz)) % Wsz
        if pad:
            q = F.pad(q, (0,0,0,pad))
            k = F.pad(k, (0,0,0,pad))
            v = F.pad(v, (0,0,0,pad))
        Tpad = q.shape[2]
        q = q.view(B, self.heads, Tpad // Wsz, Wsz, -1)
        k = k.view(B, self.heads, Tpad // Wsz, Wsz, -1)
        v = v.view(B, self.heads, Tpad // Wsz, Wsz, -1)

        attn_logits = self._cosine_attn(q, k)  # (B, H, nW, W, W)
        attn = attn_logits.softmax(dim=-1)
        out = torch.einsum('bhntw,bhntc->bhntc', attn, v)
        out = out.reshape(B, self.heads, Tpad, C // self.heads)
        out = out.transpose(1,2).reshape(B, Tpad, C)
        if pad:
            out = out[:, :T, :]

        out = self.proj(out)
        out = out + self.mlp(out)
        # broadcast back to (B, C, T, H, W)
        out = out.transpose(1,2).unsqueeze(-1).unsqueeze(-1)
        out = out.expand(B, C, T, H, W)
        return x + out * 0.5  # residual modulation