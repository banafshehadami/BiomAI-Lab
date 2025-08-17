import os
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.rrppg_gan.datasets import VideoPPGFolder
from src.rrppg_gan.models.generator import SwinAUNet
from src.rrppg_gan.signal_utils import pearson_r, rmse, frechet_distance

@torch.no_grad()
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--config', type=str, default='configs/default.yaml')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = VideoPPGFolder(args.data_root, patch_seconds=cfg['train']['patch_seconds'], sample_rate=cfg['train']['sample_rate'])
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    G = SwinAUNet(in_channels=cfg['model']['in_channels'], base=cfg['model']['base_channels'],
                   temporal_attn=cfg['model']['temporal_attn']['enabled'],
                   t_heads=cfg['model']['temporal_attn']['heads'], t_window=cfg['model']['temporal_attn']['window']).to(device)
    state = torch.load(args.ckpt, map_location=device)
    G.load_state_dict(state['G'])
    G.eval()

    R_list, RMSE_list, FD_list = [], [], []
    for batch in tqdm(dl, desc='Eval'):
        vid = batch['video'].to(device).unsqueeze(0) if batch['video'].dim()==4 else batch['video'].to(device)
        gt = batch['ppg'].to(device).unsqueeze(0) if batch['ppg'].dim()==2 else batch['ppg'].to(device)
        pred = G(vid)
        T = min(pred.shape[-1], gt.shape[-1])
        pred = pred[..., :T]
        gt = gt[..., :T]
        R_list.append(pearson_r(pred, gt))
        RMSE_list.append(rmse(pred, gt))
        FD_list.append(frechet_distance(pred, gt))

    print(f"Pearson R: {sum(R_list)/len(R_list):.4f}")
    print(f"RMSE: {sum(RMSE_list)/len(RMSE_list):.4f}")
    print(f"Frechet distance: {sum(FD_list)/len(FD_list):.4f}")

if __name__ == '__main__':
    main()