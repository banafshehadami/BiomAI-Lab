import numpy as np
import os
import h5py
from torch.utils.data import Dataset
import random


def split():
    h5_dir = './h5s'
    train_list = []
    val_list = []
    val_subject = 'create list of subjects you wanna use for validation'
    for subject in range(1, 51):
        if os.path.isfile(h5_dir + '/video_0%d.h5' % (subject)):
            if subject in val_subject:
                val_list.append(h5_dir + '/video_0%d.h5' % (subject))
            else:
                train_list.append(h5_dir + '/video_0%d.h5' % (subject))
    return train_list, val_list


def test_list(dir):
    list = []
    for subject in range(1, 51):
        if os.path.isfile(dir + '/video_0%d.h5' % (subject)):
            list.append(dir + '/video_0%d.h5' % (subject))
    return list
def Give_T(train_list):
    T = float('inf')
    for file in train_list:
        with h5py.File(file, 'r') as f:
            img_length = f['imgs'].shape[0]
            if img_length < T:
                T = img_length
    return T

class H5Dataset(Dataset):

    def __init__(self, train_list, T):
        self.train_list = train_list # list of .h5 file paths for training
        self.T = T # video clip length

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        with h5py.File(self.train_list[idx], 'r') as f:
            img_length = f['imgs'].shape[0]

            idx_start = np.random.choice(img_length-self.T)

            idx_end = idx_start+self.T

            img_seq = f['imgs'][idx_start:idx_end]
            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
        return img_seq
