import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import torch
from Model import Autoencoder
from loss import ConLoss
from power import power

from utils_data import *
from utils_sig import *
from torch import optim
from torch.utils.data import DataLoader
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment('model_train', save_git_info=False, interactive=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

@ex.automain
def my_main():
    # hyperparams for model training
    total_epoch = 100 # total number of epochs for training the model
    lr = 0.00001# learning rate
    in_ch = 3 #number of input video channels, in_ch=3 for RGB videos, in_ch=1 for NIR videos.

    # hyperparams for ST-rPPG block
    fs = 24 # video frame rate, modify it if your video frame rate is not 30 fps.
    S = 2 # spatial dimenion of ST-rPPG block, default is 2x2.
    # hyperparams for rPPG spatiotemporal sampling
    K = 4 # the number of rPPG samples at each spatial position

    result_dir = '.....' # store checkpoints and training recording

    ex.observers.append(FileStorageObserver(result_dir))

    h5_dir = '.......'

    train_list = []
    # for subject in range(0,51):
    #     if os.path.isfile(h5_dir+'video_0%d.h5'%(subject)):
    #         train_list.append(h5_dir+'video_0%d.h5'%(subject))
    exp_dir = result_dir + '/%s'%("run") # store experiment recording to the path


    train_list, test_list = split()
    TotalFrames = Give_T(train_list) -1  # temporal dimension of ST-rPPG block, default is 10 seconds.
    T = 48
    print("TotalFrames" , TotalFrames)
    delta_t = int(T / 2)  # time length of each rPPG sample
    np.save(exp_dir + '/train_list.npy', train_list)
    np.save(exp_dir + '/test_list.npy', test_list)
    # define the dataloader
    dataset = H5Dataset(train_list, TotalFrames) # please read the code about H5Dataset when preparing your dataset
    dataloader = DataLoader(dataset, batch_size=2, # two videos for contrastive learning
                            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    sum =0
    for batch in dataloader:
        print(batch.size())
        sum = sum + 1
    print("sum", sum)
    # define the model and loss
    model = Autoencoder(S, in_ch=in_ch).to(device).train()
    loss_func = ConLoss(delta_t, K, fs, high_pass=40, low_pass=250)
    # define irrelevant power ratio
    IPR = power(Fs=fs, high_pass=40, low_pass=250)
    # define the optimizer
    opt = optim.AdamW(model.parameters(), lr=lr)

    for e in range(total_epoch):
        Total_Iteration = 1
        for it in range(Total_Iteration):
            img_num = 0
            for imgs in dataloader: # dataloader randomly samples a video clip with length T
                print(imgs.size())
                imgs = imgs.to(device)

                # model forward propagation
                model_output = model(imgs)
                rppg = model_output[:,-1] # get rppg

                # define the loss functions
                loss, p_loss, n_loss = loss_func(model_output)

                # optimize
                opt.zero_grad()
                loss.backward()
                opt.step()

                # evaluate irrelevant power ratio during training
                ipr = torch.mean(IPR(rppg.clone().detach()))

                # save loss values and IPR
                ex.log_scalar("loss", loss.item())
                ex.log_scalar("p_loss", p_loss.item())
                ex.log_scalar("n_loss", n_loss.item())
                ex.log_scalar("ipr", ipr.item())
                print('Epoch [%d/%d], Iteration [%d/%d], Batch [%d/%d],loss: %.4f, p_loss: %.4f, n_loss: %.4f'
                    % (e + 1, total_epoch, it ,Total_Iteration , img_num + 1, len(dataloader), loss.item(), p_loss.item(),
                       n_loss.item()))
                img_num = img_num + 1
        # save model checkpoints
        torch.save(model.state_dict(), exp_dir+'/epoch%d.pt'%e)

if __name__ == "__main__":
   my_main()