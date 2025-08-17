import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import pywt

# --------- basic signal ops ---------

def second_derivative(x):
    """x: (B,1,T) → sd: (B,1,T-2)"""
    return x[..., 2:] - 2*x[..., 1:-1] + x[..., :-2]

@torch.no_grad()
def pearson_r(a, b, eps=1e-8):
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    num = (a*b).sum(dim=-1)
    den = torch.sqrt((a.square().sum(dim=-1)+eps) * (b.square().sum(dim=-1)+eps))
    return (num/den).mean().item()

@torch.no_grad()
def rmse(a, b):
    return torch.sqrt((a - b).square().mean()).item()

@torch.no_grad()
def frechet_distance(a, b):
    # Use a simple curve distance proxy (discrete Fréchet via Hausdorff-like)
    a = a.squeeze().cpu().numpy()
    b = b.squeeze().cpu().numpy()
    # directed Hausdorff in both directions, take max
    d1 = directed_hausdorff(a[:, None], b[:, None])[0]
    d2 = directed_hausdorff(b[:, None], a[:, None])[0]
    return float(max(d1, d2))

# --------- wavelet map (db4) ---------

def db4_wavelet_map(x, levels=5):
    """x: (B,1,T) → coeff map (B, levels, T)
    Build a levels×T map by discrete wavelet decomposition then upsample each level back to T.
    """
    B, _, T = x.shape
    maps = []
    x_np = x.detach().cpu().numpy()
    for b in range(B):
        coeffs = pywt.wavedec(x_np[b,0], 'db4', level=levels)
        # coeffs: [cA_L, cD_L, cD_{L-1}, ..., cD1]
        # Build details-only stack (ignore final approximation or include as level 0)
        stack = []
        for c in coeffs[1:]:  # details
            c_t = torch.tensor(c, dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)  # (1,1,L)
            c_t = F.interpolate(c_t, size=T, mode='linear', align_corners=False)
            stack.append(c_t)
        wmap = torch.cat(stack, dim=1)  # (1, levels, T)
        maps.append(wmap)
    return torch.cat(maps, dim=0)  # (B, levels, T)

# --------- HR estimation helper (simple peak distance at given fs) ---------

def estimate_hr_bpm(sig, fs):
    # sig: (B,1,T); very simple peak detector
    x = sig.squeeze(1)
    x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
    # moving average filter
    kernel = torch.ones(1,1,5, device=x.device)/5
    xf = F.conv1d(x.unsqueeze(1), kernel, padding=2).squeeze(1)
    # detect positive peaks
    left = xf[:, 1:-1] - xf[:, :-2]
    right = xf[:, 1:-1] - xf[:, 2:]
    peaks = (left > 0) & (right > 0) & (xf[:, 1:-1] > 0)
    peak_counts = peaks.sum(dim=1).float()
    duration = (sig.shape[-1] / fs)
    bpm = (peak_counts / duration) * 60.0
    return bpm