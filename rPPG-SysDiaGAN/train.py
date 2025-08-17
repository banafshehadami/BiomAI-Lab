import os
import yaml
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.rrppg_gan.datasets import VideoPPGFolder
from src.rrppg_gan.models.generator import SwinAUNet
from src.rrppg_gan.models.discriminators import TimeDiscriminator, SDDiscriminator, FreqDiscriminator
from src.rrppg_gan.losses import MultiDomainLoss
from src.rrppg_gan.signal_utils import second_derivative, db4_wavelet_map


def bce_gan_loss(logits, is_real=True):
    target = torch.ones_like(logits) if is_real else torch.zeros_like(logits)
    return nn.functional.binary_cross_entropy_with_logits(logits, target)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--out', type=str, default='runs/exp')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['train'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(cfg.get('seed', 1337))

    ds = VideoPPGFolder(args.data_root, patch_seconds=cfg['train']['patch_seconds'], sample_rate=cfg['train']['sample_rate'])
    dl = DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['train']['num_workers'])

    G = SwinAUNet(in_channels=cfg['model']['in_channels'], base=cfg['model']['base_channels'],
                   temporal_attn=cfg['model']['temporal_attn']['enabled'],
                   t_heads=cfg['model']['temporal_attn']['heads'], t_window=cfg['model']['temporal_attn']['window']).to(device)

    D_t = TimeDiscriminator().to(device)
    D_sd = SDDiscriminator().to(device)
    D_f = FreqDiscriminator(in_ch=ds.__len__() and 5 or 5).to(device)  # 5 levels by default

    optG = torch.optim.Adam(G.parameters(), lr=cfg['train']['lr'], betas=tuple(cfg['train']['betas']))
    optD = torch.optim.Adam(list(D_t.parameters()) + list(D_sd.parameters()) + list(D_f.parameters()), lr=cfg['train']['lr'], betas=tuple(cfg['train']['betas']))

    md_loss = MultiDomainLoss(alpha=cfg['loss']['alpha_time'], beta=cfg['loss']['beta_freq'], gamma=cfg['loss']['gamma_sd'],
                              softdtw_gamma=cfg['loss']['softdtw_gamma'], cdf_bins=cfg['loss']['cdf_bins'], cdf_sigma=cfg['loss']['cdf_sigma'],
                              freq_delta_bins=cfg['loss']['freq_delta_bins'])

    os.makedirs(args.out, exist_ok=True)
    best_loss = 1e9

    for epoch in range(1, cfg['train']['epochs'] + 1):
        G.train(); D_t.train(); D_sd.train(); D_f.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}")
        running = {'G_total': 0.0, 'L_time': 0.0, 'L_freq': 0.0, 'L_sd': 0.0, 'D': 0.0}
        for batch in pbar:
            vid = batch['video'].to(device)          # (B?, 3, T, H, W) after collation
            ppg = batch['ppg'].to(device)            # (B?, 1, T)
            # ensure proper batch dim
            if vid.dim() == 4:
                vid = vid.unsqueeze(0)
            if ppg.dim() == 2:
                ppg = ppg.unsqueeze(0)

            # ---- Train D ----
            with torch.no_grad():
                fake = G(vid)
            # domains
            sd_r = second_derivative(ppg)
            sd_f = second_derivative(fake)
            fmap_r = db4_wavelet_map(ppg)
            fmap_f = db4_wavelet_map(fake)

            optD.zero_grad()
            # time
            d_real_t = D_t(ppg)
            d_fake_t = D_t(fake)
            loss_dt = bce_gan_loss(d_real_t, True) + bce_gan_loss(d_fake_t, False)
            # sd
            d_real_sd = D_sd(sd_r)
            d_fake_sd = D_sd(sd_f)
            loss_dsd = bce_gan_loss(d_real_sd, True) + bce_gan_loss(d_fake_sd, False)
            # freq (2D): add channel dim for in_ch=1
            d_real_f = D_f(fmap_r.unsqueeze(1))
            d_fake_f = D_f(fmap_f.unsqueeze(1))
            loss_df = bce_gan_loss(d_real_f, True) + bce_gan_loss(d_fake_f, False)
            loss_D = (loss_dt + loss_dsd + loss_df) / 3.0
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(list(D_t.parameters()) + list(D_sd.parameters()) + list(D_f.parameters()), cfg['train']['grad_clip'])
            optD.step()

            # ---- Train G ----
            optG.zero_grad()
            fake = G(vid)
            # adversarial fooling
            g_adv = bce_gan_loss(D_t(fake), True) + bce_gan_loss(D_sd(second_derivative(fake)), True) + bce_gan_loss(D_f(db4_wavelet_map(fake).unsqueeze(1)), True)
            # multi-domain reconstruction
            md_total, md_parts = md_loss(fake, ppg)
            g_total = cfg['loss']['adv_weight'] * g_adv + md_total
            g_total.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), cfg['train']['grad_clip'])
            optG.step()

            running['G_total'] += g_total.item()
            running['L_time'] += md_parts['L_time'].item()
            running['L_freq'] += md_parts['L_freq'].item()
            running['L_sd'] += md_parts['L_sd'].item()
            running['D'] += loss_D.item()

            pbar.set_postfix({k: f"{v/(pbar.n+1):.3f}" for k,v in running.items()})

        # save
        avgG = running['G_total']/max(1,len(dl))
        if avgG < best_loss:
            best_loss = avgG
            torch.save({'G': G.state_dict(), 'epoch': epoch}, os.path.join(args.out, 'best.ckpt'))
        if epoch % cfg['train']['save_every'] == 0:
            torch.save({'G': G.state_dict(), 'epoch': epoch}, os.path.join(args.out, f'epoch_{epoch}.ckpt'))

if __name__ == '__main__':
    main()