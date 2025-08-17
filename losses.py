import torch
import torch.nn as nn
import torch.nn.functional as F
from .signal_utils import second_derivative, db4_wavelet_map

# ---------- Soft-DTW (Cuturi & Blondel) ----------
class SoftDTW(nn.Module):
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, x, y):
        """x,y: (B,1,T) → scalar mean soft-DTW. Uses log-sum-exp DP (differentiable)."""
        B, _, T = x.shape
        S = (x - y).abs()  # L1 distance matrix along time via broadcasting below
        # Build full pairwise distance for each batch: D[b,i,j] = |x[b,i]-y[b,j]|
        x_exp = x.squeeze(1)[:, :, None]  # (B,T,1)
        y_exp = y.squeeze(1)[:, None, :]  # (B,1,T)
        D = (x_exp - y_exp).pow(2)  # (B,T,T)

        gamma = self.gamma
        big = 1e7
        R = x.new_full((B, T+1, T+1), big)
        R[:,0,0] = 0.0
        for i in range(1, T+1):
            Di = D[:, i-1, :]
            for j in range(1, T+1):
                r0 = R[:, i-1, j]
                r1 = R[:, i, j-1]
                r2 = R[:, i-1, j-1]
                r = torch.stack([r0, r1, r2], dim=-1)
                # softmin via -gamma * logsumexp(-r/gamma)
                softmin = -gamma * torch.logsumexp(-r / gamma, dim=-1)
                R[:, i, j] = Di[:, j-1] + softmin
        return R[:, -1, -1].mean()

# ---------- CDF variance loss (differentiable with soft histograms) ----------
class CDFVarianceLoss(nn.Module):
    def __init__(self, bins=64, sigma=0.05):
        super().__init__()
        self.bins = bins
        self.sigma = sigma

    def forward(self, pred, target):
        # pred/target: (B,1,T) or (B,K,T) → flatten last two dims
        B = pred.shape[0]
        x = pred.reshape(B, -1)
        y = target.reshape(B, -1)
        # normalize to [0,1]
        x = (x - x.min(dim=1, keepdim=True).values) / (x.max(dim=1, keepdim=True).values - x.min(dim=1, keepdim=True).values + 1e-6)
        y = (y - y.min(dim=1, keepdim=True).values) / (y.max(dim=1, keepdim=True).values - y.min(dim=1, keepdim=True).values + 1e-6)
        centers = torch.linspace(0, 1, self.bins, device=x.device).view(1, -1)
        def soft_hist(z):
            # Gaussian soft binning
            d = z.unsqueeze(-1) - centers  # (B,N,bins)
            w = torch.exp(-0.5 * (d / self.sigma)**2)
            h = w.sum(dim=1) + 1e-6
            h = h / h.sum(dim=-1, keepdim=True)
            cdf = torch.cumsum(h, dim=-1)
            return cdf
        cdf_x = soft_hist(x)
        cdf_y = soft_hist(y)
        return F.mse_loss(cdf_x, cdf_y)

# ---------- Sparsity losses ----------
class SparsityLoss(nn.Module):
    def forward(self, x):
        return x.abs().mean()

class FreqSparsityLoss(nn.Module):
    def __init__(self, delta_bins=2, levels=5):
        super().__init__()
        self.delta = delta_bins
        self.levels = levels
    def forward(self, freq_map):
        # freq_map: (B, levels, T), compute energy per level and emphasize around spectral peak
        energy = freq_map.abs().mean(dim=-1)  # (B, levels)
        peak = energy.argmax(dim=-1)  # (B,)
        loss = 0.0
        for b in range(freq_map.shape[0]):
            a = max(0, int(peak[b]) - self.delta)
            c = min(self.levels - 1, int(peak[b]) + self.delta)
            sel = energy[b, a:c+1].sum() / (energy[b].sum() + 1e-6)
            # maximize concentration around peak → minimize 1 - sel
            loss = loss + (1.0 - sel)
        return loss / freq_map.shape[0]

# ---------- Domain wrappers ----------
class MultiDomainLoss(nn.Module):
    def __init__(self, alpha=1.5, beta=0.8, gamma=1.2, softdtw_gamma=0.1, cdf_bins=64, cdf_sigma=0.05, freq_delta_bins=2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sdtw = SoftDTW(gamma=softdtw_gamma)
        self.var = CDFVarianceLoss(bins=cdf_bins, sigma=cdf_sigma)
        self.sparsity = SparsityLoss()
        self.freq_sparsity = FreqSparsityLoss(delta_bins=freq_delta_bins)
        self.levels = 5

    def forward(self, pred, target):
        # Time domain
        l_time = self.sdtw(pred, target) + self.sparsity(pred) + self.var(pred, target)
        # SD domain (align lengths)
        sd_p = second_derivative(pred)
        sd_t = second_derivative(target)
        T = min(sd_p.shape[-1], sd_t.shape[-1])
        sd_p = sd_p[..., :T]
        sd_t = sd_t[..., :T]
        l_sd = self.sdtw(sd_p, sd_t) + self.sparsity(sd_p) + self.var(sd_p, sd_t)
        # Freq domain via db4 wavelet map
        fmap_p = db4_wavelet_map(pred, levels=self.levels)
        fmap_t = db4_wavelet_map(target, levels=self.levels)
        l_freq = self.freq_sparsity(fmap_p) + self.var(fmap_p, fmap_t)
        total = self.alpha * l_time + self.beta * l_freq + self.gamma * l_sd
        return total, {
            'L_time': l_time.detach(), 'L_freq': l_freq.detach(), 'L_sd': l_sd.detach()
        }