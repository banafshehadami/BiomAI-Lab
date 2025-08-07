import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.fft


class ConLoss(nn.Module):
    """Contrastive Loss module."""

    def __init__(self, delta_t, K, Fs, high_pass, low_pass):
        super(ConLoss, self).__init__()
        self.ST_sampling = ST_sampling(delta_t, K, Fs, high_pass, low_pass)  # spatiotemporal sampler
        self.distance_func = nn.MSELoss(reduction='mean')  # mean squared error for comparing two PSDs

    def sample_compare(self, list_a, list_b, exclude_same=False):
        """Compare samples."""
        if exclude_same:
            total_distance = 0.
            M = 0
            for i in range(len(list_a)):
                for j in range(len(list_b)):
                    if i != j:
                        total_distance += self.distance_func(list_a[i], list_b[j])
                        M += 1
        else:
            total_distance = 0.
            M = 0
            for i in range(len(list_a)):
                for j in range(len(list_b)):
                    total_distance += self.distance_func(list_a[i], list_b[j])
                    M += 1
        return total_distance / M

    def forward(self, model_output):
        """Forward pass."""
        samples = self.ST_sampling(model_output)
        p_loss = (self.sample_compare(samples[0], samples[0], exclude_same=True) +
                  self.sample_compare(samples[1], samples[1], exclude_same=True)) / 2
        n_loss = -self.sample_compare(samples[0], samples[1])
        loss = p_loss + n_loss
        return loss, p_loss, n_loss


class ST_sampling(nn.Module):
    """Spatiotemporal sampling on ST-rPPG block."""

    def __init__(self, delta_t, K, Fs, high_pass, low_pass):
        super().__init__()
        self.delta_t = delta_t  # time length of each rPPG sample
        self.K = K  # the number of rPPG samples at each spatial position
        self.norm_psd = normPSD(Fs, high_pass, low_pass)

    def forward(self, input):
        """Forward pass."""
        samples = []
        for b in range(input.shape[0]):  # loop over videos (totally 2 videos)
            samples_per_video = []
            for c in range(input.shape[1]):  # loop for sampling over spatial dimension
                for i in range(self.K):  # loop for sampling K samples with time length delta_t along temporal dimension
                    offset = torch.randint(0, input.shape[-1] - self.delta_t + 1, (1,),
                                           device=input.device)  # randomly sample along temporal dimension
                    x = self.norm_psd(input[b, c, offset:offset + self.delta_t])
                    samples_per_video.append(x)
            samples.append(samples_per_video)
        return samples


class normPSD(nn.Module):
    """Normalize PSD module."""

    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=0):
        """Forward pass."""
        x = x - torch.mean(x, dim=-1, keepdim=True)
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad / 2 * L), int(zero_pad / 2 * L)), 'constant', 0)

        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x = torch.add(x[:, 0] ** 2, x[:, 1] ** 2)
        Fn = self.Fs / 2
        freq = torch.linspace(0, Fn, x.shape[0])
        use_freqs = torch.logical_and(freq >= self.high_pass / 60, freq <= self.low_pass / 60)
        x = x[use_freqs]

        # Normalize PSD
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return x
