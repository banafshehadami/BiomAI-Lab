import os
import numpy as np
import cv2
import torch
import json
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Model import Autoencoder
from utils_data import H5Dataset, Give_T
from functions import *
from utils_sig import *

fps = 24
device = torch.device('cpu')
train_exp_dir = './'  # Give your directory
frames_len = 100

with open(train_exp_dir + '/config.json') as f:
    config_train = json.load(f)

def my_main():
    image_folder = input('What is the folder?\n You can input paths like: "./" or "c:/". ')
    count_real = 0
    count_fake = 0
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = cv2.imread(os.path.join(image_folder, filename))
            crop(image, filename, image_folder)
            name = filename.split('.')[0]
            video_path = os.path.join(image_folder, f"video_{name}.avi")
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 24.0, (128, 128))
            image_path = os.path.join(image_folder, filename)
            img = cv2.imread(image_path)
            for i in range(0, frames_len):
                writer.write(img)
            writer.release()
            cap = cv2.VideoCapture(os.path.join(image_folder, f"video_{name}.avi"))
            frames = []
            ret, frame = cap.read()
            cap.release()
            for i in range(0, frames_len):
                frames.append(frame)

            h5_filename = os.path.join(image_folder + '/h5', name + '.h5')
            with h5py.File(h5_filename, 'w') as h5_file:
                h5_file.create_dataset('imgs', data=frames, dtype='uint8', chunks=(1, 128, 128, 3),
                                        compression="gzip", compression_opts=4)

            test_list = [h5_filename]
            TotalFrames = Give_T(test_list) - 1  # Temporal dimension of ST-rPPG block, default is 10 seconds.
            dataset = H5Dataset(test_list, TotalFrames)
            dataloader = DataLoader(dataset, batch_size=1,  # Two videos for contrastive learning
                                    shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
            model = Autoencoder(config_train['S'], config_train['in_ch']).to(device).eval()

            model.load_state_dict(torch.load(train_exp_dir+'/epoch%d.pt' % (99), map_location=device))
            id = 0
            os.remove(video_path)
            for imgs in dataloader:
                imgs = imgs.to(device)

                model_output = model(imgs)
                rppg = model_output[:, -1]  # Get rppg
                rppg = rppg[0].detach().cpu().numpy()
                rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=fps)

                hr, psd_y, psd_x = hr_fft(rppg, fs=fps)

                fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))

                ax1.plot(np.arange(len(rppg)) / fps, rppg)
                ax1.set_xlabel('Time (sec)')
                ax1.grid(True)
                ax1.set_title('rPPG waveform')

                ax2.plot(psd_x, psd_y)
                ax2.set_xlabel('Heart rate (bpm)')
                ax2.set_xlim([40, 200])
                ax2.grid(True)
                ax2.set_title('PSD')

                plt.savefig('./results%d.png' % (id))
                id = id + 1

                print('Heart rate: %.2f bpm' % hr)

if __name__ == "__main__":
    my_main()
