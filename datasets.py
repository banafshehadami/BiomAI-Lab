import os
import glob
import torch
from torch.utils.data import Dataset

class VideoPPGFolder(Dataset):
    """Loads .pt files each containing a dict: {"video": (T,3,H,W), "ppg": (T,)}"""
    def __init__(self, root, patch_seconds=4, sample_rate=30):
        self.files = sorted(glob.glob(os.path.join(root, '**', '*.pt'), recursive=True))
        self.patch_len = patch_seconds * sample_rate
        self.sample_rate = sample_rate
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        d = torch.load(self.files[idx])
        vid = d['video'].float()  # (T,3,H,W)
        ppg = d['ppg'].float()    # (T,)
        T = vid.shape[0]
        if T < self.patch_len:
            # pad
            pad = self.patch_len - T
            vid = torch.cat([vid, vid[-1:].repeat(pad,1,1,1)], dim=0)
            ppg = torch.cat([ppg, ppg[-1:].repeat(pad)], dim=0)
        # random crop
        start = torch.randint(0, max(1, T - self.patch_len + 1), (1,)).item()
        vid = vid[start:start+self.patch_len]  # (L,3,H,W)
        ppg = ppg[start:start+self.patch_len]
        # to (B,C,T,H,W) with B=1 for model convenience in collation
        return {
            'video': vid.permute(1,0,2,3),  # (3,T,H,W)
            'ppg': ppg.unsqueeze(0),        # (1,T)
            'fs': self.sample_rate
        }